package main

import (
	"context"
	"errors"
	"testing"
)

func TestROCmpleteEvalGroupCommandReturnsInterruptStatus(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	opts := EncodeOptions{ProbeOptions: defaultProbeOptions(), GroupCRF: true}
	if got := runEncodeCommand(ctx, opts, []string{"a.mkv"}); got != 130 {
		t.Fatalf("exit code = %d, want 130", got)
	}
}

func TestROCmpleteEvalCancellationPreventsFallbackEncoding(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	opts := EncodeOptions{
		ProbeOptions:  defaultProbeOptions(),
		FallbackCRFSet: true,
		FallbackCRF:    32,
	}
	err := groupFallbackOrError(ctx, opts, []groupInput{{File: "a.mkv"}}, errors.New("probe failed"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context cancellation", err)
	}
}
