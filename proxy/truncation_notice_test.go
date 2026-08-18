package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestLooksCutMidSentence(t *testing.T) {
	long := strings.Repeat("内容", 120) // 240 runes
	for _, tc := range []struct {
		name  string
		tail  string
		chars int
		want  bool
	}{
		{"cut mid word", long + "什么质", 243, true},
		{"cut mid phrase", long + "第一步先把图里那", 247, true},
		{"complete cjk period", long + "就这样。", 244, false},
		{"complete ascii period", long + "done.", 245, false},
		{"complete code fence", long + "\n```", 244, false},
		{"short answer without punctuation is fine", "收到", 2, false},
		{"short answer with colon is fine", "好的:", 3, false},
		{"empty tail", "   ", 300, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := looksCutMidSentence(tc.tail, tc.chars); got != tc.want {
				t.Fatalf("looksCutMidSentence(%q, %d) = %v, want %v", tc.tail, tc.chars, got, tc.want)
			}
		})
	}
}

func TestTruncationNoticeConsumeOnce(t *testing.T) {
	recordTruncationNotice("conv-1", "test reason")
	first := consumeTruncationNotice("conv-1")
	if !strings.Contains(first, "test reason") {
		t.Fatalf("expected notice with reason, got %q", first)
	}
	if second := consumeTruncationNotice("conv-1"); second != "" {
		t.Fatalf("notice must be consumed once, got %q", second)
	}
	if other := consumeTruncationNotice("conv-2"); other != "" {
		t.Fatalf("unrelated conversation must have no notice, got %q", other)
	}
}

func TestTruncationNoticeExpires(t *testing.T) {
	truncationNotices.Store("conv-old", truncationNotice{reason: "stale", at: time.Now().Add(-time.Hour)})
	if got := consumeTruncationNotice("conv-old"); got != "" {
		t.Fatalf("stale notice must be dropped, got %q", got)
	}
}

func TestClaudeToKiroInjectsPendingNotice(t *testing.T) {
	req := &ClaudeRequest{
		Model:    "claude-opus-5",
		Messages: []ClaudeMessage{{Role: "user", Content: "继续"}},
	}
	first := ClaudeToKiro(req, false)
	convID := first.ConversationState.ConversationID

	recordTruncationNotice(convID, "cut while streaming tool arguments")

	second := ClaudeToKiro(req, false)
	got := second.ConversationState.CurrentMessage.UserInputMessage.Content
	if !strings.Contains(got, "System note from the API gateway") || !strings.Contains(got, "cut while streaming tool arguments") {
		t.Fatalf("expected advisory prepended to current message, got %q", got)
	}
	if !strings.HasSuffix(got, "继续") {
		t.Fatalf("original content must be preserved after the note, got %q", got)
	}

	third := ClaudeToKiro(req, false)
	if strings.Contains(third.ConversationState.CurrentMessage.UserInputMessage.Content, "System note") {
		t.Fatalf("notice must be consumed once")
	}
}

// A text-only turn that ends mid-sentence (despite terminal usage events)
// completes normally but arms an advisory for the next request.
func TestIntegrityRetryRecordsSoftNoticeForCutText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		var body []byte
		body = append(body, awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": strings.Repeat("这是一段很长的回复。", 30) + "然后我要先",
		})...)
		body = append(body, awsEventStreamFrame(t, "contextUsageEvent", map[string]interface{}{
			"contextUsagePercentage": 40.0,
		})...)
		body = append(body, awsEventStreamFrame(t, "meteringEvent", map[string]interface{}{
			"usage": 0.5,
		})...)
		_, _ = w.Write(body)
	}))
	defer server.Close()
	defer setupIntegrityTestUpstream(t, server)()

	payload := integrityTestPayload()
	payload.ConversationState.ConversationID = "conv-soft-cut"

	var content string
	err := runKiroWithIntegrityRetry(context.Background(), integrityTestAccount(), payload,
		&KiroStreamCallback{OnText: func(s string, _ bool) { content += s }},
		func() (int, int, string, bool) { return len([]rune(content)), 0, "", false },
		nil, nil)
	if err != nil {
		t.Fatalf("turn with terminal usage must complete, got %v", err)
	}
	notice := consumeTruncationNotice("conv-soft-cut")
	if !strings.Contains(notice, "cut off mid-sentence") {
		t.Fatalf("expected soft cut advisory for next request, got %q", notice)
	}
}

// ContentLengthExceededException from the upstream is an output-limit signal,
// not a failure: the turn must end gracefully with stop_reason=max_tokens and
// arm an advisory, instead of erroring after partial content.
func TestContentLengthExceededMapsToMaxTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		var body []byte
		body = append(body, awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": "partial answer",
		})...)
		body = append(body, awsEventStreamFrameWithHeaders(t, map[string]string{
			":message-type": "exception",
			":event-type":   "exception",
		}, map[string]interface{}{
			"message": "ContentLengthExceededException: content length exceeded",
		})...)
		_, _ = w.Write(body)
	}))
	defer server.Close()
	defer setupIntegrityTestUpstream(t, server)()

	payload := integrityTestPayload()
	payload.ConversationState.ConversationID = "conv-cle"

	var content, stopReason string
	err := runKiroWithIntegrityRetry(context.Background(), integrityTestAccount(), payload,
		&KiroStreamCallback{
			OnText:       func(s string, _ bool) { content += s },
			OnStopReason: func(r string) { stopReason = r },
		},
		func() (int, int, string, bool) { return len(content), 0, stopReason, false },
		nil, nil)
	if err != nil {
		t.Fatalf("content-length-exceeded must not error, got %v", err)
	}
	if content != "partial answer" {
		t.Fatalf("partial content must be kept, got %q", content)
	}
	if stopReason != "max_tokens" {
		t.Fatalf("stopReason = %q, want max_tokens", stopReason)
	}
	notice := consumeTruncationNotice("conv-cle")
	if !strings.Contains(notice, "content length limit") {
		t.Fatalf("expected content-limit advisory, got %q", notice)
	}
}

// Other upstream exceptions must still surface as errors.
func TestOtherExceptionStillErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		var body []byte
		body = append(body, awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
			"content": "partial answer",
		})...)
		body = append(body, awsEventStreamFrameWithHeaders(t, map[string]string{
			":message-type": "exception",
			":event-type":   "exception",
		}, map[string]interface{}{
			"message": "ValidationException: something else",
		})...)
		_, _ = w.Write(body)
	}))
	defer server.Close()
	defer setupIntegrityTestUpstream(t, server)()

	err := runKiroWithIntegrityRetry(context.Background(), integrityTestAccount(), integrityTestPayload(),
		&KiroStreamCallback{OnText: func(string, bool) {}},
		func() (int, int, string, bool) { return 0, 0, "", false },
		nil, nil)
	if err == nil || !strings.Contains(err.Error(), "ValidationException") {
		t.Fatalf("non-content-length exceptions must surface, got %v", err)
	}
}

// A stream dying with incomplete tool JSON arms an advisory so the next
// request retries the operation in smaller chunks.
func TestIntegrityRetryRecordsNoticeForIncompleteToolInput(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		// Tool input JSON never closes, then EOF: the classic oversized-Write cut.
		_, _ = w.Write(awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
			"toolUseId": "toolu_1", "name": "Write", "input": `{"file_path":"/tmp/big.go","content":"package main`,
		}))
	}))
	defer server.Close()
	defer setupIntegrityTestUpstream(t, server)()

	payload := integrityTestPayload()
	payload.ConversationState.ConversationID = "conv-tool-cut"

	err := runKiroWithIntegrityRetry(context.Background(), integrityTestAccount(), payload,
		&KiroStreamCallback{},
		func() (int, int, string, bool) { return 0, 0, "", false },
		nil, nil)
	if err == nil {
		t.Fatalf("incomplete tool input must surface an error")
	}
	notice := consumeTruncationNotice("conv-tool-cut")
	if !strings.Contains(notice, "never reached the client") {
		t.Fatalf("expected tool-cut advisory for next request, got %q", notice)
	}
}
