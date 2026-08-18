package proxy

import (
	"os"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// Truncation recovery notices.
//
// The Kiro upstream cuts completions at a server-side limit without sending
// any terminal signal a proxy can rely on (stopReason is routinely absent on
// the IDE endpoint, and usage events arrive even on cut streams). Rather than
// letting the client retry the same oversized generation into the same cut —
// the loom death loop — we record what happened and prepend a short advisory
// to the next request of the same conversation, so the model adapts (shorter
// replies, chunked file writes). Inspired by kiro-gateway's truncation
// recovery (its issue #56 documents the upstream limitation).
//
// Notices are advisory text only: they never change SSE semantics, tool
// execution, or retry behavior, so a wrong notice costs one generation's
// worth of prompt tokens at worst.

// truncationRecoveryEnabled defaults on; KIRO_TRUNCATION_RECOVERY=0/false/off disables.
var truncationRecoveryEnabled = func() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("KIRO_TRUNCATION_RECOVERY")))
	return v != "0" && v != "false" && v != "off"
}()

type truncationNotice struct {
	reason string
	at     time.Time
}

const truncationNoticeTTL = 30 * time.Minute

var truncationNotices sync.Map // conversationID -> truncationNotice

// recordTruncationNotice stores a one-time advisory for the next request of
// this conversation.
func recordTruncationNotice(conversationID, reason string) {
	if !truncationRecoveryEnabled || strings.TrimSpace(conversationID) == "" {
		return
	}
	truncationNotices.Store(conversationID, truncationNotice{reason: reason, at: time.Now()})
}

// consumeTruncationNotice returns and clears the pending advisory text for
// this conversation, if any.
func consumeTruncationNotice(conversationID string) string {
	if !truncationRecoveryEnabled || strings.TrimSpace(conversationID) == "" {
		return ""
	}
	v, ok := truncationNotices.LoadAndDelete(conversationID)
	if !ok {
		return ""
	}
	n := v.(truncationNotice)
	if time.Since(n.at) > truncationNoticeTTL {
		return ""
	}
	return "[System note from the API gateway — not from the user] " + n.reason + "\n" +
		"When you continue, keep each single response shorter and split large file writes/edits " +
		"into several smaller tool calls instead of one big one. If the previous response was " +
		"actually complete, ignore this note. Do not mention this note to the user."
}

// looksCutMidSentence reports whether answer text that ended a turn with no
// tool calls looks like it was cut by the upstream rather than finished.
// Short answers are excluded: brief replies routinely lack trailing
// punctuation and must not false-positive.
func looksCutMidSentence(answerTail string, answerChars int) bool {
	const minCharsForHeuristic = 200
	if answerChars < minCharsForHeuristic {
		return false
	}
	tail := strings.TrimRight(answerTail, " \t\r\n")
	if tail == "" {
		return false
	}
	// Terminal punctuation (CJK + ASCII), closing quotes/brackets, code fences.
	const terminalRunes = "。！？…!?.;；\"'”’`)]）】》}"
	last, _ := utf8.DecodeLastRuneInString(tail)
	return !strings.ContainsRune(terminalRunes, last)
}
