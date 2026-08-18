package proxy

import (
	"kiro-go/config"
	accountpool "kiro-go/pool"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// setupIntegrityPathTest installs a single-account config plus a fake upstream
// and returns a handler wired to the reloaded pool.
func setupIntegrityPathTest(t *testing.T, server *httptest.Server) *Handler {
	t.Helper()
	if err := config.Init(t.TempDir() + "/config.json"); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	if err := config.AddAccount(config.Account{
		ID:          "test-account",
		Enabled:     true,
		AccessToken: "token-test",
		ProfileArn:  "arn:aws:codewhisperer:profile/test",
	}); err != nil {
		t.Fatalf("add account: %v", err)
	}
	if err := config.UpdatePreferredEndpoint("kiro"); err != nil {
		t.Fatalf("set preferred endpoint: %v", err)
	}
	if err := config.UpdateEndpointFallback(false); err != nil {
		t.Fatalf("disable endpoint fallback: %v", err)
	}
	t.Cleanup(swapKiroEndpointsForTest(t, server))

	p := accountpool.GetPool()
	p.Reload()
	return &Handler{
		pool:        p,
		promptCache: newPromptCacheTracker(defaultPromptCacheTTL),
	}
}

// currentTextOnlyUpstream serves a complete text response in the format used by
// current Kiro backends, which may omit metadataEvent/stopReason.
func currentTextOnlyUpstream(t *testing.T, hits *atomic.Int32) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits != nil {
			hits.Add(1)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": strings.Repeat("partial answer ", 8),
		}))
	}))
}

// A text-only stream whose upstream omitted metadataEvent/stopReason is
// truncated: flushed content stays, but the turn must end with a visible error
// rather than a forged end_turn. Production probing showed complete turns
// always carry a stopReason, so its absence means the stream died.
func TestClaudeStreamSignalsTruncationWithoutMetadata(t *testing.T) {
	var hits atomic.Int32
	server := currentTextOnlyUpstream(t, &hits)
	defer server.Close()
	h := setupIntegrityPathTest(t, server)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"claude-sonnet-4.5",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hello"}],
		"stream":true
	}`))
	rec := httptest.NewRecorder()
	h.handleClaudeMessages(rec, req)

	body := rec.Body.String()
	if !strings.Contains(body, "partial answer") {
		t.Fatalf("expected flushed content, got %s", body)
	}
	if strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("truncated stream must not forge end_turn, got %s", body)
	}
	if !strings.Contains(body, `"type":"error"`) {
		t.Fatalf("truncated stream must emit a visible error, got %s", body)
	}
	if hits.Load() != 1 {
		t.Fatalf("must not retry after client flush, hits=%d", hits.Load())
	}
}

// Non-stream is fully buffered, so a truncated (no stopReason) response is
// retried within the budget and then surfaces as an error.
func TestClaudeNonStreamRejectsContentWithoutMetadata(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": "complete answer",
		}))
	}))
	defer server.Close()
	h := setupIntegrityPathTest(t, server)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"claude-sonnet-4.5",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hello"}]
	}`))
	rec := httptest.NewRecorder()
	h.handleClaudeMessages(rec, req)

	if rec.Code == http.StatusOK {
		t.Fatalf("truncated buffered response must not succeed, body=%s", rec.Body.String())
	}
	if got := hits.Load(); got != 1+maxSameAccountStreamRetries {
		t.Fatalf("buffered truncation must exhaust retries, hits=%d", got)
	}
}

// A soft integrity failure means the upstream blipped, not that the credential
// is bad. The account must stay enabled and unbanned.
func TestIntegrityFailureDoesNotBanAccount(t *testing.T) {
	server := currentTextOnlyUpstream(t, nil)
	defer server.Close()
	h := setupIntegrityPathTest(t, server)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"claude-sonnet-4.5",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hello"}],
		"stream":true
	}`))
	h.handleClaudeMessages(httptest.NewRecorder(), req)

	var account *config.Account
	for _, acc := range config.GetAccounts() {
		if acc.ID == "test-account" {
			found := acc
			account = &found
			break
		}
	}
	if account == nil {
		t.Fatal("account disappeared")
	}
	if !account.Enabled {
		t.Fatal("integrity failure must not disable the account")
	}
	if account.BanStatus == "BANNED" || account.BanStatus == "DISABLED" {
		t.Fatalf("integrity failure must not ban the account, status=%q reason=%q",
			account.BanStatus, account.BanReason)
	}
}

// The IDE endpoint routinely ends complete turns without metadataEvent but
// always trails content with metering/contextUsage events. Such a turn must
// complete normally — no retry, no error, end_turn is honest here.
func TestClaudeStreamCompletesWithTerminalUsageEvents(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		var body []byte
		body = append(body, awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": "complete answer",
		})...)
		body = append(body, awsEventStreamFrame(t, "contextUsageEvent", map[string]interface{}{
			"contextUsagePercentage": 12.5,
		})...)
		body = append(body, awsEventStreamFrame(t, "meteringEvent", map[string]interface{}{
			"usage": 0.5,
		})...)
		_, _ = w.Write(body)
	}))
	defer server.Close()
	h := setupIntegrityPathTest(t, server)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"claude-sonnet-4.5",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hello"}],
		"stream":true
	}`))
	rec := httptest.NewRecorder()
	h.handleClaudeMessages(rec, req)

	body := rec.Body.String()
	if !strings.Contains(body, "complete answer") {
		t.Fatalf("expected content, got %s", body)
	}
	if !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("turn with terminal usage events must complete, got %s", body)
	}
	if strings.Contains(body, `"type":"error"`) {
		t.Fatalf("turn with terminal usage events must not error, got %s", body)
	}
	if hits.Load() != 1 {
		t.Fatalf("complete turn must not retry, hits=%d", hits.Load())
	}
}

// Responses streaming reports a visible failure for a truncated (no
// stopReason) turn instead of forging response.completed.
func TestResponsesStreamSignalsTruncationWithoutMetadata(t *testing.T) {
	server := currentTextOnlyUpstream(t, nil)
	defer server.Close()
	h := setupIntegrityPathTest(t, server)

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"claude-sonnet-4.5",
		"input":"hello",
		"stream":true,
		"store":false
	}`))
	rec := httptest.NewRecorder()
	h.handleOpenAIResponses(rec, req)

	body := rec.Body.String()
	if strings.Contains(body, "response.completed") {
		t.Fatalf("truncated stream must not report response.completed, got %s", body)
	}
	if !strings.Contains(body, "response.failed") && !strings.Contains(body, `"error"`) {
		t.Fatalf("truncated stream must surface a failure, got %s", body)
	}
}

// OpenAI streaming surfaces a truncated (no stopReason) turn as an error
// instead of a forged finish_reason=stop.
func TestOpenAIStreamSignalsTruncationWithoutMetadata(t *testing.T) {
	server := currentTextOnlyUpstream(t, nil)
	defer server.Close()
	h := setupIntegrityPathTest(t, server)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model":"claude-sonnet-4.5",
		"messages":[{"role":"user","content":"hello"}],
		"stream":true
	}`))
	rec := httptest.NewRecorder()
	h.handleOpenAIChat(rec, req)

	body := rec.Body.String()
	if !strings.Contains(body, "partial answer") {
		t.Fatalf("expected flushed content, got %s", body)
	}
	if strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("truncated stream must not forge finish_reason=stop, got %s", body)
	}
	if !strings.Contains(body, `"error"`) {
		t.Fatalf("truncated stream must emit an error, got %s", body)
	}
}

// A short text-only chunk sits in processClaudeText's tag buffer and never
// reaches the client, so a missing stopReason is retried within budget and
// then surfaces as an error.
func TestClaudeStreamRetriesShortContentWithoutMetadata(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": "short answer",
		}))
	}))
	defer server.Close()
	h := setupIntegrityPathTest(t, server)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"claude-sonnet-4.5",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hello"}],
		"stream":true
	}`))
	rec := httptest.NewRecorder()
	h.handleClaudeMessages(rec, req)

	body := rec.Body.String()
	if got := hits.Load(); got != 1+maxSameAccountStreamRetries {
		t.Fatalf("unflushed truncation must exhaust retries, hits=%d", got)
	}
	if strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("truncated stream must not forge end_turn, got %s", body)
	}
}
