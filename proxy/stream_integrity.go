package proxy

import (
	"context"
	"errors"
	"fmt"
	"kiro-go/config"
	"kiro-go/logger"
	"strings"
)

// runKiroWithIntegrityRetry calls Kiro and recovers a truncated upstream stream
// the way Kiro IDE does: retry the same request on the same account within a
// bounded budget before surfacing the failure.
//
// callback is reused across attempts. Both CallKiroAPIContext and
// parseEventStreamTracked copy the struct before wrapping any field, so a retry
// cannot double-wrap it; per-attempt state is cleared by reset instead.
// measure reports the integrity inputs after a transport-successful call.
// reset clears per-attempt state before a same-account retry; may be nil.
// canRetry reports whether a retry is still safe (for streaming: nothing has
// been flushed to the client yet). nil means always retryable.
//
// Truncation recovery: every detected truncation records a one-time advisory
// for the conversation's next request (see truncation_notice.go), and a retry
// within this function gets the advisory injected into the payload directly,
// so the regeneration adapts instead of being cut at the same spot again.
//
// Return contract:
//   - nil: complete success only
//   - transport error from CallKiroAPIContext: caller should rotate/ban as usual
//   - integrity error while still retryable: retries exhausted; caller should
//     rotate account without treating it as an auth/quota failure
//   - integrity error after client flush: caller must surface failure to the
//     client (do not fake end_turn / normal completion). Retry is unsafe.
func runKiroWithIntegrityRetry(
	ctx context.Context,
	account *config.Account,
	payload *KiroPayload,
	callback *KiroStreamCallback,
	measure func() (contentChars, toolCount int, stopReason string, sawReasoning bool),
	reset func(),
	canRetry func() bool,
) error {
	label := accountEmailForLog(account)
	retryable := func() bool {
		if canRetry == nil {
			return true
		}
		return canRetry()
	}
	conversationID := ""
	if payload != nil {
		conversationID = payload.ConversationState.ConversationID
	}

	for attempt := 0; attempt <= maxSameAccountStreamRetries; attempt++ {
		if attempt > 0 && reset != nil {
			reset()
		}

		// Terminal usage accounting (metering/contextUsage events) rides at the
		// natural end of a turn, after the final content frame. Its arrival
		// therefore proves the stream ran to completion even when the backend
		// omits a stopReason — which the IDE endpoint does routinely. The parse
		// layer reports it (plus tail forensics) through OnStreamEnd.
		var endInfo StreamEndInfo
		attemptCallback := *callback
		prevStreamEnd := attemptCallback.OnStreamEnd
		attemptCallback.OnStreamEnd = func(info StreamEndInfo) {
			endInfo = info
			if prevStreamEnd != nil {
				prevStreamEnd(info)
			}
		}

		err := CallKiroAPIContext(ctx, account, payload, &attemptCallback)
		if err != nil {
			if errors.Is(err, errIncompleteKiroToolInput) {
				// The turn dies here; arm the next request so the model
				// chunks the oversized tool call instead of repeating it.
				recordTruncationNotice(conversationID, fmt.Sprintf(
					"The previous assistant response was cut off by an upstream output limit while a tool call's arguments were still streaming (%v). That tool call never reached the client and was not executed.",
					err))
			}
			return err
		}

		contentChars, toolCount, stopReason, sawReasoning := measure()
		terminalUsage := endInfo.TerminalUsage
		integrityErr := classifyStreamIntegrity(contentChars, toolCount, stopReason, sawReasoning, terminalUsage)
		if integrityErr == nil {
			// Turn accepted. Arm an advisory when the upstream cut it: either
			// it said so (ContentLengthExceededException) or a text-only turn
			// ended mid-sentence despite terminal usage events.
			switch {
			case endInfo.ContentLengthExceeded:
				recordTruncationNotice(conversationID,
					"The previous assistant response hit the upstream content length limit and was cut off before it finished.")
			case toolCount == 0 && looksCutMidSentence(endInfo.AnswerTail, contentChars):
				recordTruncationNotice(conversationID, fmt.Sprintf(
					"The previous assistant response appears to have been cut off mid-sentence by an upstream output limit (it ended near %d characters, in the middle of a thought).",
					contentChars))
			}
			return nil
		}

		// A genuine truncation: the stream died before any terminal signal.
		// Log the forensic state; the IDE endpoint omitting stopReason on
		// complete turns is normal and stays quiet.
		logger.Warnf("[StreamIntegrity] stream on %s truncated: contentChars=%d toolCount=%d sawReasoning=%v terminalUsage=%v",
			label, contentChars, toolCount, sawReasoning, terminalUsage)

		// A canceled client is not an integrity failure: the turn is over and
		// reissuing it would only burn upstream quota.
		if ctx != nil && ctx.Err() != nil {
			return ctx.Err()
		}

		if retryable() && attempt < maxSameAccountStreamRetries {
			// Nothing reached the client yet, so the retry regenerates the
			// whole answer; arm it so the new attempt adapts.
			injectTruncationAdvisory(payload, fmt.Sprintf("%v", integrityErr))
			logger.Warnf("[StreamIntegrity] %v on %s; retrying same account (%d/%d)",
				integrityErr, label, attempt+1, maxSameAccountStreamRetries)
			continue
		}

		recordTruncationNotice(conversationID, fmt.Sprintf(
			"The previous assistant response was cut off by an upstream output limit (%v).", integrityErr))

		if !retryable() {
			// Bytes already reached the client; reissuing would duplicate output.
			// Return the integrity error so callers emit an error event instead of
			// finishing with a forged end_turn/tool_use success.
			logger.Warnf("[StreamIntegrity] %v after client flush; signaling error (no retry)", integrityErr)
			return integrityErr
		}

		logger.Warnf("[StreamIntegrity] giving up after retries: %v", integrityErr)
		return integrityErr
	}

	// Unreachable: every branch inside the loop returns or continues, and the
	// final iteration cannot continue.
	return errUpstreamTruncatedResponse
}

// injectTruncationAdvisory prepends the recovery note to the payload's current
// message so a same-request retry regenerates with smaller outputs. Idempotent
// per payload: the marker is only added once.
func injectTruncationAdvisory(payload *KiroPayload, reason string) {
	if !truncationRecoveryEnabled || payload == nil {
		return
	}
	msg := &payload.ConversationState.CurrentMessage.UserInputMessage
	if strings.Contains(msg.Content, "[System note from the API gateway") {
		return
	}
	note := "[System note from the API gateway — not from the user] A previous attempt at this response was cut off by an upstream output limit (" +
		reason + "). Keep this response shorter and split large file writes/edits into several smaller tool calls. Do not mention this note to the user."
	msg.Content = note + "\n\n" + msg.Content
}
