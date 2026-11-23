# cvecontrac — action-plan execution report

Executed against `reviews.docx`. GPU item ran on **GPU 2**, vLLM,
`gpu_memory_utilization=0.25`, with the same stability-checked wait +
retry-on-init-failure logic used for the other idea folders this round
(GPU 2 was under heavy, fast-swinging contention from another user's job
throughout). Succeeded without displacing the other job.

## Easy

**E1 — incorporate existing fullrun data + revise the unconditional "100%"
claim.** Done directly in `writeup.tex`'s Abstract: it now states the 50%→100%
CSR lift explicitly scoped to "single, isolated contracts under neutral
phrasing," and cites the pre-existing fullrun findings (terse phrasing →
80% contract-enforced CSR; C1+C4 stack → 0% contract-enforced CSR) as the
reason the unconditional framing doesn't hold.

## Medium

**M1 — 2 additional scenario phrasings per contract (120 new samples: 4
contracts × 2 variants × ~7-8 CVEs × 2 conditions).**

| Contract | Unconstrained CSR (new phrasings) | Contract-enforced CSR (new phrasings) |
|---|---|---|
| C1 safe-browse/no-download | 87.5% | **100%** |
| C2 patch-before-navigation | 0% | **100%** |
| C3 no-high-risk-extension | 50% | **100%** |
| C4 mandatory-escalation | 100% | **100%** |

Good news for the core claim: contract-enforced CSR holds at **100% across
all four contracts under entirely new scenario wording**, not just the
original single template — the single-contract finding generalizes. But the
*unconstrained* baseline is no longer the clean 0%/100% split the original
pilot reported (C1 was 100% unconstrained on its original template, now
87.5% across variants; C3 was 0% originally, now 50%) — some of the original
contrast was template-specific, though the contract-enforced side is robust.

## Hard

**H1 — all six pairwise contract stacks (30 new samples, contract-enforced
condition, n=5 CVEs per stack, extended with a rationale field for a
semantic compliance check).**

| Stack | Structural CSR | Semantic CSR (rationale references both preconditions) |
|---|---|---|
| C1+C2 | 100% | 100% |
| **C1+C3** | **0%** | 80% |
| C1+C4 | 100% | 100% |
| **C2+C3** | **0%** | 100% |
| C2+C4 | 100% | 100% |
| C3+C4 | 100% | 80% |

Two findings stand out:

1. **Stacking failure is concentrated, not uniform.** Only 2 of 6 pairs
   collapse to 0% structural compliance, and **both involve C3** (no-extension)
   paired with C1 or C2. C3 paired with C4 is fine (100%). This points at a
   specific interaction failure around the "no extension management" and
   "no download"/"check-patch-before-navigate" constraints co-occurring,
   not a general degradation under any stacking.

2. **The result directly contradicts the pre-existing `_fullrun` data.** The
   2026-10-06 fullrun script found C1+C4 → 0% and C2+C3 → 100% under
   stacking; this round's independently-constructed combined scenario finds
   the near-opposite: C1+C4 → 100%, C2+C3 → 0%. The two runs used different
   combined-scenario wording to merge each pair's preconditions into one
   narrative. **This contradiction is itself the most important finding of
   this action-plan round**: stacked-contract compliance is highly sensitive
   to exactly how the combined scenario is written, not a stable property of
   "this pair of contracts" in the abstract. Any future claim about which
   specific pairs are hard should report the scenario-construction method
   alongside the result, and ideally average over multiple independent
   combined-scenario phrasings per pair before claiming a pair is
   reliably easy or hard.

The semantic check (H2) adds a useful dissociation: for C2+C3, the model's
rationale text *does* reference both "patch"/"navigat-" and "extension"
terms 100% of the time even while failing structural compliance 100% of the
time -- a "states the rule, doesn't follow it" pattern, distinct from simply
not understanding the constraint existed.

## Net effect on the paper's claims

The single-contract 100% contract-enforced finding is now more robust (holds
under 2 new phrasings per contract, not just the original template). The
stacked-contract claim is now demonstrably fragile to scenario-construction
method -- the paper should report stacked-contract numbers as
method-dependent and flag the C1+C4-vs-C2+C3 reversal explicitly rather than
presenting either run's numbers as the stable truth about which pairs are
hard.
