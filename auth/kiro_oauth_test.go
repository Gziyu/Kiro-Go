package auth

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"
)

func TestStartKiroOAuthBuildsPKCEPortalURL(t *testing.T) {
	start, err := StartKiroOAuth("http://localhost:8320")
	if err != nil {
		t.Fatalf("StartKiroOAuth: %v", err)
	}
	parsed, err := url.Parse(start.AuthorizationURL)
	if err != nil {
		t.Fatalf("parse authorization URL: %v", err)
	}
	if parsed.Scheme+"://"+parsed.Host+parsed.Path != kiroOAuthPortalURL {
		t.Fatalf("authorization endpoint = %q", parsed.String())
	}
	query := parsed.Query()
	if query.Get("state") != start.State {
		t.Fatalf("state mismatch")
	}
	if query.Get("redirect_uri") != "http://localhost:8320" {
		t.Fatalf("redirect_uri = %q", query.Get("redirect_uri"))
	}
	if query.Get("code_challenge_method") != "S256" || query.Get("redirect_from") != "KiroIDE" {
		t.Fatalf("missing Kiro PKCE parameters: %v", query)
	}
	digest := sha256.Sum256([]byte(start.CodeVerifier))
	expectedChallenge := base64.RawURLEncoding.EncodeToString(digest[:])
	if query.Get("code_challenge") != expectedChallenge {
		t.Fatalf("code_challenge mismatch")
	}
	if start.ExpiresAt.Before(time.Now().Add(9 * time.Minute)) {
		t.Fatalf("session expiry is too short: %v", start.ExpiresAt)
	}
}

func TestStartKiroOAuthRejectsNonLoopbackCallback(t *testing.T) {
	for _, callback := range []string{
		"https://example.com:8320",
		"http://192.168.1.2:8320",
		"http://localhost:8320/oauth/callback",
		"http://localhost",
	} {
		if _, err := StartKiroOAuth(callback); err == nil {
			t.Fatalf("expected callback %q to be rejected", callback)
		}
	}
}

func TestParseKiroOAuthCallbackValidatesStateAndProvider(t *testing.T) {
	callback, err := ParseKiroOAuthCallback(
		"http://localhost:8320/oauth/callback?state=expected&code=abc&login_option=google",
		"expected",
	)
	if err != nil {
		t.Fatalf("ParseKiroOAuthCallback: %v", err)
	}
	if callback.Code != "abc" || callback.LoginOption != "google" || callback.Path != "/oauth/callback" {
		t.Fatalf("unexpected callback: %+v", callback)
	}
	if _, err := ParseKiroOAuthCallback(
		"http://localhost:8320/oauth/callback?state=wrong&code=abc&login_option=google",
		"expected",
	); err == nil {
		t.Fatal("expected state mismatch")
	}
	if _, err := ParseKiroOAuthCallback(
		"http://localhost:8320/oauth/callback?state=expected&code=abc&login_option=builderid",
		"expected",
	); err == nil {
		t.Fatal("expected unsupported login option")
	}
}

func TestExchangeKiroOAuthCode(t *testing.T) {
	var received map[string]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s", r.Method)
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": map[string]interface{}{
				"accessToken":  "access-new",
				"refreshToken": "refresh-new",
				"profileArn":   "arn:aws:codewhisperer:us-east-1:123456789012:profile/test",
				"expiresIn":    3600,
			},
		})
	}))
	defer server.Close()

	oldURL := GetKiroOAuthTokenURLForTest()
	SetKiroOAuthTokenURLForTest(func() string { return server.URL })
	defer SetKiroOAuthTokenURLForTest(oldURL)
	oldClient := SetGlobalAuthClientForTest(server.Client())
	defer SetGlobalAuthClientForTest(oldClient)

	before := time.Now().Unix()
	token, err := ExchangeKiroOAuthCode(&KiroOAuthCallback{
		Path: "/signin/callback", LoginOption: "github", Code: "code-1",
	}, "verifier-1", "http://localhost:8320")
	if err != nil {
		t.Fatalf("ExchangeKiroOAuthCode: %v", err)
	}
	if received["code"] != "code-1" || received["code_verifier"] != "verifier-1" {
		t.Fatalf("unexpected exchange body: %+v", received)
	}
	if received["redirect_uri"] != "http://localhost:8320/signin/callback?login_option=github" {
		t.Fatalf("redirect_uri = %q", received["redirect_uri"])
	}
	if token.AccessToken != "access-new" || token.RefreshToken != "refresh-new" {
		t.Fatalf("unexpected token: %+v", token)
	}
	if token.ExpiresAt < before+3595 {
		t.Fatalf("expiresAt is too early: %d", token.ExpiresAt)
	}
}
