# Logit Temperature Clamp Design

**Date**: 2026-08-05  
**Status**: Approved

## Problem

Step 63 produces maxv=6.3M logit inflation from HC state growth.
This causes degenerate "won won won" repetition — one token overwhelms softmax.

## Fix

After `run_lm_head` produces logits, before argmax:
1. Find max logit (`maxv`)
2. If `maxv > 30.0`, scale all logits by `30.0 / maxv`
3. Proceed with argmax

Threshold 30.0: normal generation has maxv ≈ 13-18, so 30 gives 2x headroom.
