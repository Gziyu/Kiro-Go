package proxy

import (
	"context"
	"errors"
	"kiro-go/config"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// classifyStreamIntegrity treats a turn as complete on a stopReason, a
// delivered tool call, or answer content followed by terminal usage events
// (metering/contextUsage — the IDE endpoint routinely omits metadataEvent on
// complete turns). Content with neither signal is a truncation.
func TestClassifyStreamIntegrity(t *testing.T) {
	for _, tc := range []struct {
		name          string
		content       int
		tools         int
		stopReason    string
		sawReasoning  bool
		terminalUsage bool
		wantErr       error
	}{
		{"complete with stop", 12, 0, "end_turn", false, false, nil},
		{"complete with tools", 0, 1, "", false, false, nil},
		{"complete with tools despite content", 12, 1, "", false, false, nil},
		{"content with terminal usage is complete", 8, 0, "", false, true, nil},
		{"content without any terminal signal is truncated", 8, 0, "", false, false, errUpstreamTruncatedResponse},
		{"reasoning only stricter than ide", 0, 0, "", true, false, errUpstreamTruncatedResponse},
		{"reasoning only with usage still truncated", 0, 0, "", true, true, errUpstreamTruncatedResponse},
		{"no signal at all", 0, 0, "", false, false, errUpstreamTruncatedResponse},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := classifyStreamIntegrity(tc.content, tc.tools, tc.stopReason, tc.sawReasoning, tc.terminalUsage)
			if tc.wantErr == nil {
				if got != nil {
					t.Fatalf("got %v, want nil", got)
				}
				return
			}
			if got == nil || got.Error() != tc.wantErr.Error() {
				t.Fatalf("got %v, want %v", got, tc.wantErr)
			}
			if !isStreamIntegrityError(got) {
				t.Fatalf("%v must be recognized as a stream integrity error", got)
			}
		})
	}
}

// setupIntegrityTestUpstream points the Kiro endpoint list at a fake upstream
// and returns a restore func. Mirrors the fixture used by handler tests.
func setupIntegrityTestUpstream(t *testing.T, server *httptest.Server) func() {
	t.Helper()
	if err := config.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	if err := config.UpdatePreferredEndpoint("kiro"); err != nil {
		t.Fatalf("set endpoint: %v", err)
	}
	if err := config.UpdateEndpointFallback(false); err != nil {
		t.Fatalf("disable fallback: %v", err)
	}

	oldEndpoints := kiroEndpoints
	kiroEndpoints = []kiroEndpoint{{URL: server.URL, Origin: "AI_EDITOR", Name: "test"}}
	oldClient := kiroHttpStore.Load()
	kiroHttpStore.Store(&http.Client{Timeout: time.Second, Transport: &http.Transport{}})
	return func() {
		kiroEndpoints = oldEndpoints
		kiroHttpStore.Store(oldClient)
	}
}

func integrityTestAccount() *config.Account {
	return &config.Account{
		ID:          "acc",
		Email:       "acc@test",
		AccessToken: "token",
		ProfileArn:  "arn:aws:codewhisperer:profile/test",
	}
}

func integrityTestPayload() *KiroPayload {
	payload := &KiroPayload{}
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: "hi",
		Origin:  "AI_EDITOR",
	}
	return payload
}

// A stream that delivered answer content without stopReason is truncated:
// controlled testing against the production backend showed complete turns
// always carry a metadataEvent stopReason. Before anything is flushed it must
// be retried; if every attempt truncates the error surfaces.
func TestRunKiroWithIntegrityRetryRetriesContentWithoutMetadata(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": "complete answer",
		}))
	}))
	defer server.Close()
	defer setupIntegrityTestUpstream(t, server)()

	var content string
	var stopReason string
	var resets int
	err := runKiroWithIntegrityRetry(context.Background(), integrityTestAccount(), integrityTestPayload(),
		&KiroStreamCallback{
			OnText:       func(s string, _ bool) { content += s },
			OnStopReason: func(r string) { stopReason = r },
		},
		func() (int, int, string, bool) {
			return len(content), 0, stopReason, false
		},
		func() {
			resets++
			content = ""
			stopReason = ""
		},
		nil,
	)
	if !errors.Is(err, errUpstreamTruncatedResponse) {
		t.Fatalf("expected truncation error, got %v", err)
	}
	if got := hits.Load(); got != 1+maxSameAccountStreamRetries {
		t.Fatalf("content without stopReason must exhaust retries, hits=%d", got)
	}
	if resets != maxSameAccountStreamRetries {
		t.Fatalf("expected %d resets, got %d", maxSameAccountStreamRetries, resets)
	}
}

// Once answer content has reached the client, a missing stopReason means the
// stream was truncated mid-turn. Retry would duplicate output, so the error
// must surface visibly instead of forging a successful end_turn.
func TestRunKiroWithIntegrityRetrySignalsTruncationAfterClientFlush(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": "partial",
		}))
		// no metadataEvent/stopReason => truncated
	}))
	defer server.Close()
	defer setupIntegrityTestUpstream(t, server)()

	var content string
	flushed := true
	err := runKiroWithIntegrityRetry(context.Background(), integrityTestAccount(), integrityTestPayload(),
		&KiroStreamCallback{
			OnText: func(s string, _ bool) { content += s },
		},
		func() (int, int, string, bool) { return len(content), 0, "", false },
		func() { content = "" },
		func() bool { return !flushed },
	)
	if !errors.Is(err, errUpstreamTruncatedResponse) {
		t.Fatalf("expected truncation error after flush, got %v", err)
	}
	if hits.Load() != 1 {
		t.Fatalf("must not retry after client flush, hits=%d", hits.Load())
	}
	if content != "partial" {
		t.Fatalf("flushed content must be preserved, content=%q", content)
	}
}

// A canceled client context must not drive an integrity retry. The turn is over,
// so reissuing it would only burn upstream quota.
//
// The upstream here returns content with no stopReason, which classifies as
// truncated and would otherwise be retried; cancellation must suppress that.
func TestRunKiroWithIntegrityRetryStopsOnCanceledContext(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": "partial",
		}))
	}))
	defer server.Close()
	defer setupIntegrityTestUpstream(t, server)()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var content string
	var resets int
	err := runKiroWithIntegrityRetry(ctx, integrityTestAccount(), integrityTestPayload(),
		&KiroStreamCallback{OnText: func(s string, _ bool) { content += s }},
		func() (int, int, string, bool) { return len(content), 0, "", false },
		func() { resets++ },
		nil,
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if isStreamIntegrityError(err) {
		t.Fatalf("cancellation must not be reported as an integrity failure: %v", err)
	}
	if resets != 0 {
		t.Fatalf("cancellation must not trigger a retry reset, got %d", resets)
	}
	if got := hits.Load(); got > 1 {
		t.Fatalf("canceled context must not drive retries, hits=%d", got)
	}
}

// The truncation retry budget stays bounded for reasoning-only turns. Text-only
// turns are valid current-backend completions, while reasoning with no answer or
// tool call remains incomplete and consumes the bounded retry budget.
func TestRunKiroWithIntegrityRetryStopsAfterBudgetExhausted(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		// Always truncated: reasoning with no answer or terminal signal.
		_, _ = w.Write(awsEventStreamFrame(t, "reasoningContentEvent", map[string]interface{}{
			"text": "unfinished reasoning",
		}))
	}))
	defer server.Close()
	defer setupIntegrityTestUpstream(t, server)()

	var reasoning string
	err := runKiroWithIntegrityRetry(context.Background(), integrityTestAccount(), integrityTestPayload(),
		&KiroStreamCallback{OnText: func(s string, isReasoning bool) {
			if isReasoning {
				reasoning += s
			}
		}},
		func() (int, int, string, bool) { return 0, 0, "", reasoning != "" },
		func() { reasoning = "" },
		nil,
	)
	if !isStreamIntegrityError(err) {
		t.Fatalf("expected integrity error once the budget is spent, got %v", err)
	}
	if got := hits.Load(); got != int32(maxSameAccountStreamRetries+1) {
		t.Fatalf("upstream hits=%d, want %d (initial attempt plus %d retry)",
			got, maxSameAccountStreamRetries+1, maxSameAccountStreamRetries)
	}
}
