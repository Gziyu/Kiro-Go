package auth

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	kiroOAuthPortalURL = "https://app.kiro.dev/signin"
	kiroOAuthTimeout   = 10 * time.Minute
)

var kiroOAuthTokenURL = func() string {
	return "https://prod.us-east-1.auth.desktop.kiro.dev/oauth/token"
}

// KiroOAuthStart contains the public and private values for one PKCE login.
type KiroOAuthStart struct {
	ID               string
	State            string
	CodeVerifier     string
	CallbackBase     string
	AuthorizationURL string
	ExpiresAt        time.Time
}

// KiroOAuthCallback is the validated callback returned by app.kiro.dev.
type KiroOAuthCallback struct {
	Path        string
	LoginOption string
	Code        string
}

// KiroOAuthToken contains the credential material returned by the token endpoint.
type KiroOAuthToken struct {
	AccessToken  string
	RefreshToken string
	ProfileArn   string
	ExpiresAt    int64
}

// StartKiroOAuth creates a state-bound PKCE login URL. callbackBase must point
// to a local loopback listener; remote callback hosts are deliberately rejected.
func StartKiroOAuth(callbackBase string) (*KiroOAuthStart, error) {
	callbackBase, err := normalizeKiroOAuthCallbackBase(callbackBase)
	if err != nil {
		return nil, err
	}

	state, err := randomBase64URL(32)
	if err != nil {
		return nil, fmt.Errorf("generate OAuth state: %w", err)
	}
	verifier, err := randomBase64URL(32)
	if err != nil {
		return nil, fmt.Errorf("generate PKCE verifier: %w", err)
	}
	challengeSum := sha256.Sum256([]byte(verifier))
	challenge := base64.RawURLEncoding.EncodeToString(challengeSum[:])

	portal, _ := url.Parse(kiroOAuthPortalURL)
	query := portal.Query()
	query.Set("state", state)
	query.Set("code_challenge", challenge)
	query.Set("code_challenge_method", "S256")
	query.Set("redirect_uri", callbackBase)
	query.Set("redirect_from", "KiroIDE")
	portal.RawQuery = query.Encode()

	return &KiroOAuthStart{
		ID:               GenerateAccountID(),
		State:            state,
		CodeVerifier:     verifier,
		CallbackBase:     callbackBase,
		AuthorizationURL: portal.String(),
		ExpiresAt:        time.Now().Add(kiroOAuthTimeout),
	}, nil
}

// ParseKiroOAuthCallback validates state and extracts the authorization code.
func ParseKiroOAuthCallback(rawURL, expectedState string) (*KiroOAuthCallback, error) {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return nil, fmt.Errorf("invalid callback URL: %w", err)
	}
	if parsed.Path != "/oauth/callback" && parsed.Path != "/signin/callback" {
		return nil, fmt.Errorf("callback path must be /oauth/callback or /signin/callback")
	}

	query := parsed.Query()
	if oauthError := strings.TrimSpace(query.Get("error")); oauthError != "" {
		description := strings.TrimSpace(query.Get("error_description"))
		if description != "" {
			return nil, fmt.Errorf("authorization failed: %s (%s)", oauthError, description)
		}
		return nil, fmt.Errorf("authorization failed: %s", oauthError)
	}
	if expectedState == "" || query.Get("state") != expectedState {
		return nil, fmt.Errorf("OAuth state validation failed")
	}

	code := strings.TrimSpace(query.Get("code"))
	if code == "" {
		return nil, fmt.Errorf("callback is missing authorization code")
	}
	loginOption := strings.ToLower(strings.TrimSpace(query.Get("login_option")))
	if loginOption == "" {
		loginOption = strings.ToLower(strings.TrimSpace(query.Get("loginOption")))
	}
	if loginOption != "google" && loginOption != "github" {
		return nil, fmt.Errorf("login option %q is not supported; use Google or GitHub", loginOption)
	}

	return &KiroOAuthCallback{
		Path:        parsed.Path,
		LoginOption: loginOption,
		Code:        code,
	}, nil
}

// ExchangeKiroOAuthCode exchanges a validated PKCE callback for Kiro tokens.
func ExchangeKiroOAuthCode(callback *KiroOAuthCallback, codeVerifier, callbackBase string) (*KiroOAuthToken, error) {
	if callback == nil || strings.TrimSpace(callback.Code) == "" {
		return nil, fmt.Errorf("authorization code is required")
	}
	if strings.TrimSpace(codeVerifier) == "" {
		return nil, fmt.Errorf("PKCE verifier is required")
	}
	callbackBase, err := normalizeKiroOAuthCallbackBase(callbackBase)
	if err != nil {
		return nil, err
	}

	redirectURI := strings.TrimRight(callbackBase, "/") + callback.Path +
		"?login_option=" + url.QueryEscape(callback.LoginOption)
	body, _ := json.Marshal(map[string]string{
		"code":          callback.Code,
		"code_verifier": codeVerifier,
		"redirect_uri":  redirectURI,
	})
	req, err := http.NewRequest(http.MethodPost, kiroOAuthTokenURL(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Kiro")

	resp, err := httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("Kiro token exchange failed: %w", err)
	}
	defer resp.Body.Close()
	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("read Kiro token response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Kiro token exchange returned HTTP %d", resp.StatusCode)
	}

	var envelope struct {
		AccessToken  string          `json:"accessToken"`
		RefreshToken string          `json:"refreshToken"`
		ProfileArn   string          `json:"profileArn"`
		ExpiresIn    int             `json:"expiresIn"`
		Data         json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(responseBody, &envelope); err != nil {
		return nil, fmt.Errorf("parse Kiro token response: %w", err)
	}
	if len(envelope.Data) > 0 && string(envelope.Data) != "null" {
		var nested struct {
			AccessToken  string `json:"accessToken"`
			RefreshToken string `json:"refreshToken"`
			ProfileArn   string `json:"profileArn"`
			ExpiresIn    int    `json:"expiresIn"`
		}
		if err := json.Unmarshal(envelope.Data, &nested); err == nil {
			envelope.AccessToken = nested.AccessToken
			envelope.RefreshToken = nested.RefreshToken
			envelope.ProfileArn = nested.ProfileArn
			envelope.ExpiresIn = nested.ExpiresIn
		}
	}
	if strings.TrimSpace(envelope.AccessToken) == "" || strings.TrimSpace(envelope.RefreshToken) == "" {
		return nil, fmt.Errorf("Kiro token response is missing accessToken or refreshToken")
	}
	if envelope.ExpiresIn <= 0 {
		envelope.ExpiresIn = 3600
	}

	return &KiroOAuthToken{
		AccessToken:  strings.TrimSpace(envelope.AccessToken),
		RefreshToken: strings.TrimSpace(envelope.RefreshToken),
		ProfileArn:   strings.TrimSpace(envelope.ProfileArn),
		ExpiresAt:    time.Now().Unix() + int64(envelope.ExpiresIn),
	}, nil
}

// KiroOAuthProvider returns the persisted display provider for a portal login.
func KiroOAuthProvider(loginOption string) string {
	switch strings.ToLower(strings.TrimSpace(loginOption)) {
	case "github":
		return "GitHub"
	case "google":
		return "Google"
	default:
		return "Social"
	}
}

func normalizeKiroOAuthCallbackBase(raw string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return "", fmt.Errorf("invalid callback base URL: %w", err)
	}
	if parsed.Scheme != "http" {
		return "", fmt.Errorf("callback URL must use http on local loopback")
	}
	hostname := strings.ToLower(parsed.Hostname())
	if hostname != "localhost" && hostname != "127.0.0.1" && hostname != "::1" {
		return "", fmt.Errorf("callback URL must use localhost or a loopback address")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" || (parsed.Path != "" && parsed.Path != "/") {
		return "", fmt.Errorf("callback URL must contain only the local origin")
	}
	port := parsed.Port()
	if port == "" {
		return "", fmt.Errorf("callback URL must include the local tunnel port")
	}
	portNumber, err := strconv.Atoi(port)
	if err != nil || portNumber < 1 || portNumber > 65535 {
		return "", fmt.Errorf("callback URL contains an invalid port")
	}

	host := "localhost"
	if hostname == "127.0.0.1" {
		host = "127.0.0.1"
	} else if hostname == "::1" {
		host = "[::1]"
	}
	return "http://" + host + ":" + port, nil
}

func randomBase64URL(size int) (string, error) {
	buf := make([]byte, size)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}
