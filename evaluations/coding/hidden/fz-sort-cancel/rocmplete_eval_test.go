package main

import (
	"context"
	"testing"
	"time"
)

type rocmpleteEvalCancelAfterChecksContext struct {
	checks      int
	cancelAfter int
}

func (c *rocmpleteEvalCancelAfterChecksContext) Deadline() (time.Time, bool) {
	return time.Time{}, false
}

func (c *rocmpleteEvalCancelAfterChecksContext) Done() <-chan struct{} {
	return nil
}

func (c *rocmpleteEvalCancelAfterChecksContext) Value(any) any { return nil }

func (c *rocmpleteEvalCancelAfterChecksContext) Err() error {
	c.checks++
	if c.checks >= c.cancelAfter {
		return context.Canceled
	}
	return nil
}

func TestROCmpleteEvalStableSortCancelsAfterStarting(t *testing.T) {
	items := make([]int, 256*100)
	for i := range items {
		items[i] = len(items) - i
	}
	ctx := &rocmpleteEvalCancelAfterChecksContext{cancelAfter: 5}
	if stableSortContext(ctx, items, func(a, b int) bool { return a < b }) {
		t.Fatal("stableSortContext completed after cancellation")
	}
	if ctx.checks < ctx.cancelAfter {
		t.Fatalf("context checks = %d, want at least %d", ctx.checks, ctx.cancelAfter)
	}
}

func TestROCmpleteEvalStableSortPreservesEqualOrder(t *testing.T) {
	type item struct {
		key   int
		order int
	}
	items := make([]item, 256*20+100)
	for i := range items {
		items[i] = item{key: i % 7, order: i}
	}
	if !stableSortContext(context.Background(), items, func(a, b item) bool {
		return a.key < b.key
	}) {
		t.Fatal("stableSortContext canceled unexpectedly")
	}
	for i := 1; i < len(items); i++ {
		if items[i-1].key > items[i].key {
			t.Fatalf("keys out of order at %d", i)
		}
		if items[i-1].key == items[i].key && items[i-1].order > items[i].order {
			t.Fatalf("equal-key order was not stable at %d", i)
		}
	}
}
