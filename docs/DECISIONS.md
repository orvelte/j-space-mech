# DECISIONS

Every deviation from [the spec](claude_jspace-promotion-mechanism-spec.md) is
logged here before proceeding, with the numbers that triggered it.

---

## D1 — Raw-text prompts are the primary arm (deviates from §3.1)

**Spec said:** run with a minimal chat template and `enable_thinking=False`,
noting wikitext-fit-vs-chat-prompt as a stated limitation.

**Decision (Olivia, 2026-09-03):** plain raw-text prompts are the **primary**
arm; chat-templated prompts are secondary, run only if budget allows.

**Why:** the public lens is fit on raw wikitext (`Salesforce-wikitext`,
1000 prompts × 128 tokens). The paper's §3.5 paired-question protocol is a
plain passage anyway, so the chat template buys nothing and adds the largest
single lens-validity confound. Removing it is free.

**Consequence:** the E0.1 "lens readouts on question tokens are sensible" check
now runs on raw text. If the secondary chat arm is ever run, it needs its own
E0.1.

---

## D2 — Pipeline is developed on Qwen3.5-4B, then switched to the 27B

**Spec said:** nothing; §7 hour 0–1 goes straight to the 27B.

**Decision (Olivia, 2026-09-03):** write and debug every piece of plumbing
(per-head hooks, attribution backward, counterfactual ablation, bootstrap
harness) against `Qwen/Qwen3.5-4B` with
`qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt`
(same lens repo, same `qwen-n1000` revision), then change one constant.

**Why:** seconds-per-iteration instead of minutes. Debugging on the 27B spends
the 6–8 GPU-h budget on typos. Cost is human-hours, not GPU-hours.

**Consequence:** all scripts take `MODEL_NAME` / `LENS_FILE` as parameters. No
result from the 4B is reported as a finding — it is plumbing validation only.

---

## D3 — Workspace band for K2 is layers 21–46

**Spec said:** "ℓ* must lie in Neel's reported workspace band for this model."

**Problem found:** Neel's replication post gives no layer indices — only
"two or three somewhat overlapping bands… notably less clean". It also fits its
lens to the **penultimate** layer on 25 Pile prompts, whereas our Neuronpedia
lens targets the **final** layer on 1000 wikitext prompts. So K2 as written had
no number to check against and was ungateable.

**Decision (Olivia, 2026-09-03):** the band is **layers 21–46** (inclusive),
from Olivia's own prior unpublished study on this same Qwen3.6-27B checkpoint.

**Consequence:** K2 fires if ℓ* < 21 or ℓ* > 46. Provenance must be stated in
the write-up as prior unpublished work, not as a citation to Neel's post. E0.3's
own top-25 cell-density curve over layers serves as an independent check on the
band; if it disagrees sharply, that is itself worth reporting.

---

## D4 — A/B templates are token-length-identical, not ±2

**Spec said (§3.2):** "Keep question length within ±2 tokens across A/B so
position indices at the stimulus are comparable."

**Problem found:** E2 patches activations position-by-position between the
clean (B) and corrupted (A) runs. That requires *exactly* equal token counts;
±2 silently misaligns every downstream position and would corrupt all
attribution.

**Decision:** templates are padded to **identical** token length. Every stimulus
asserts `len(tokens_A) == len(tokens_B)` and that the stimulus token IDs are
elementwise equal across conditions, at load time. Any stimulus failing the
assertion is dropped and counted in the E0 report.

---

## D5 — K0's "fit our own lens" fallback is unaffordable; use §8 instead

**Spec said (K0):** "Fit own lens with `jlens.fit` on ≥100 wikitext prompts only
if ≤2 GPU-h."

**Problem found:** `jlens.fitting` costs 1 forward + `ceil(d_model/dim_batch)`
backward passes per prompt. At `dim_batch=32` on d_model 5120 that is 160
backward passes through a 27B model per prompt — order 5–6 GPU-min each, so
~10 GPU-h for 100 prompts. The ≤2 GPU-h condition can never be satisfied at
this model size.

**Decision:** if K0 fires, go directly to the §8 fallback project, or re-target
the whole study at Qwen3.5-4B (which has an equivalent public n1000 lens). Do
not attempt to fit a 27B lens inside this budget.

---

## D6 — The final-norm gain is folded into `W_U` for the lens direction

**Spec said (§3.1):** `Jhat_l[t] = normalise(J_l^T W_U[t])`.

**Problem found:** the model's unembedding is `lm_head(final_norm(x))`, and
`final_norm` is an RMSNorm carrying a learned per-dimension gain `g`. Using the
bare `W_U[t]` therefore gives a direction that does *not* rank tokens the way the
lens's own `apply()` does.

**Decision:** use `Jhat_l[t] = normalise(J_l^T (g * W_U[t]))`.

**Why:** it makes the coordinate agree with `lens.apply`'s ranking exactly — the
remaining `1/rms(J_l h)` factor is constant across tokens and cannot change ranks
— while keeping the coordinate a *linear* functional of `h`, which is what E2's
attribution patching requires. Implemented in `jspace/lens_ops.py`.

---

## D7 — E0.2 uses an answer cue and Qwen's non-thinking prefill

**Spec said (§4 E0.2):** "under (B), model names the correct property >=90% of
stimuli."

**Problem found:** two, both discovered on the 4B dry run. (1) The measurement
prompt is question-first and the passage ends mid-clause, so at the end of the
prompt the model continues the passage rather than answering. (2) Qwen3.x opens
a `<think>` block even from raw text and was still inside it after 48 generated
tokens, so every answer scored as a miss.

**Decision:** the E0.2 *behavioral probe* appends
`"\n\nAnswer:<think>\n\n</think>\n\n"` — an answer cue plus the model's own
non-thinking convention — and greedy-decodes 16 tokens. Condition (A) keeps the
bare prompt, since its check is that the model *continues* the passage.

**Consequence:** the behavioral probe is a **different prompt** from the one used
for all internals measurements, which are unchanged and carry no cue. E0.2 tests
whether the model knows the property; it is not a measurement of the J-space.
Per spec, stimuli failing E0.2 are dropped and the count reported.

---

## D8 — E1 reports both the unrestricted l* and the first in-band l*

**Spec said (§3.3):** l* is the first layer where the (B)-(A) gap exceeds 2 SE and
stays there for >=3 consecutive layers; K2 fires if it falls outside the band.

**Problem found:** on the dry run the unrestricted rule selected layers 5, 6 and
13 — exactly the noisy early-layer regime the R-lens post warns about and that
the bad-null checklist lists as a way to make a null uninterpretable.

**Decision:** report both `l_star` (the spec's unrestricted rule, primary) and
`l_star_in_band` (the same rule restricted to layers 21-46). K2 is still
evaluated on the unrestricted value; the in-band value is the pre-registered
fallback so that a K2 firing has a principled continuation rather than an
ad-hoc one chosen after seeing the data.

---

## G1 — Gate decision at E0 (spec §7 hour 1–2 decision point): **proceed**

Numbers, Qwen3.6-27B, n=30 stimuli per property, raw text, p*=22:

| property | E0.2 naming | E0.3 ratio (all layers) | per-passage ratio CI95 | l* (spec rule) | l* in band | E0.4 at in-band l* |
|---|---|---|---|---|---|---|
| language | 100% | 1.43 | [1.43, 2.57] | 0 | 24 | J 0.223 vs LL 0.039 → **5.72x, pass** |
| tense | 100% | 3.31 | [1.32, 2.51] | 2 | 21 | J 0.077 vs LL 0.082 → **0.94x, fail** |
| pos | 73% | 47.50 | [14.24, 23.31] | 2 | 29 | J 0.155 vs LL 0.073 → **2.14x, pass** |

**K1 does not fire.** It requires E0.3 to fail for all three *or* E0.4 to fail for
all three. E0.3 passes for tense and pos; E0.4 passes for language and pos.

**K2 fires on the spec's unrestricted rule** — l* is layer 0/2/2 for all three,
i.e. the noisy early-layer regime, outside the 21–46 band. Per D8 the
pre-registered in-band fallback is used: l* = 24 (language), 21 (tense),
29 (pos). K2's "restrict to the property(ies) that pass" is applied on the
in-band values.

**Contiguous >2 SE runs in band:** language 24–32; tense 21–24 and 26–29;
pos 29–46 (continuing to 62).

**Per-property reading:**
- **language — the strongest, and the primary property for E2 onward.** One
  isolated 9-layer block (24–32) inside the band, nothing elsewhere, J/LL
  separation 5.7x at l* and >=1.6x across the whole run, E0.2 100%.
- **tense — fails E0.4 at its l\***. The logit lens shows the *same* gap at layer
  21 (ratio 0.94), so at l* this is output preparation, not workspace content
  (deflationary explanation #2). The J/LL ratio only clears 1.5x at layers
  27–29 (2.3, 2.1). Carried forward as secondary, and the write-up must state
  that its l* is the one layer where the deflationary reading is *not* excluded.
- **pos — passes E0.4 but is the most suspect.** Its >2 SE run extends from 29
  all the way to 62, which is the signature of output preparation rather than a
  localised workspace event, and E0.2 is 73%, below the spec's 90% bar. The
  8 failing stimuli are dropped per E0.2's own instruction and the property is
  re-measured before use.

**Action:** proceed to E1/E2 with **language as the headline property**, tense and
pos carried as the H3 generality test. E0.3 is additionally re-reported
restricted to the 21–46 band, which is the convention the paper's own release
uses ("experiments report over this band, not individual layers",
`data/experiments/README.md`); the all-layer aggregate dilutes a 9-layer effect
across 63 layers and is what drove language's 1.43.

---

## G1a — Correction to G1: the band does not rescue language's E0.3

G1 asserted that language's 1.43 all-layer E0.3 ratio was dilution — a 9-layer
effect averaged over 63 layers — and would rise when restricted to the band.
**That was wrong.** Measured over layers 21–46 the ratio is 1.47, and over
language's own divergence window 24–32 it is 1.49.

| property | all layers | band 21–46 | own window |
|---|---|---|---|
| language | 1.43 (1037 vs 1480 cells) | 1.47 (722 vs 1063) | 1.49, window 24–32 (297 vs 443) |
| tense | 3.31 (13 vs 43) | 0 vs 5 cells — undefined | 0 vs 5, window 21–29 |
| pos | 35.17 (12 vs 422) | 39.67 (9 vs 357) | 39.67, window 29–46 |

**The actual explanation is more interesting.** For language the top-25 rank
criterion is *at ceiling under both conditions*: the passage is in Spanish, so
` Spanish` is already in the top 25 at 722 in-band cells under the (A)
"predict the next word" question. The instruction cannot raise a rank that is
already high. What it moves is the **magnitude** — the raw coordinate gap is
0.223 with a 5.72x J/logit-lens separation. Spec §3.1 makes the raw coordinate
the primary metric and rank the secondary one precisely because rank is coarse;
language is the case that shows why.

**Consequence:** language passes on the primary metric and fails on a saturated
secondary one. It stays the headline property. E0.3 is reported for all three
ranges with this ceiling effect stated explicitly — reporting only the ratio
would misrepresent language as the weak property when it is the clean one.

Tense's in-band cell counts (0 vs 5) are too small to support any ratio; its
all-layer 3.31 is driven entirely by out-of-band layers. Combined with its E0.4
failure at l*=21, tense is the weakest of the three, not the second strongest.

*(Implementation note: the ratio helper divides by `sum + 1e-9`, which prints a
nonsense 5e9 when the denominator is truly zero. Zero-denominator cases are
reported as raw counts, never as a ratio.)*

---

## G2 — The workspace band stays at 21–46, unadjusted

**Question:** should the band be re-fitted now that E0 data exists?

**Decision: no.** Keep Olivia's prior 21–46 exactly as pre-registered.

**Why:**
1. **It is not binding on any live decision.** All three in-band l* values land
   at 21–29, nowhere near the upper edge. Only the lower edge does any work, by
   excluding the layer 0–13 early-layer artifact.
2. **The lower edge is independently corroborated by this data.** Every
   property's in-band divergence onset is >=21 (language 24, tense 21, pos 29),
   and the boot-currency probe reads `Italy` from L23. The only sub-21
   significance is scattered and non-contiguous — the R-lens early-layer artifact.
3. **Re-fitting would make K2 circular.** The band's entire value right now is
   that it is an *external* prior which rescued l* from the early-layer regime.
   Tuning it on the same data used to select l* would destroy that independence
   and the write-up would lose the defence.

**Recorded caveats, to be stated in the write-up rather than fixed by moving the
band:**
- Tense's l*=21 sits exactly on the lower edge, so it is edge-sensitive: a true
  edge of 22–23 would move tense's l* to 26. This compounds its E0.4 failure.
- POS's divergence runs from 29 to 62, past the band's upper edge and into the
  output. The upper region of the band blends into output preparation for that
  property; that is a caveat on POS, not evidence the edge is misplaced.
- The per-property **divergence window** (language 24–32, tense 21–29, pos 29–46)
  is reported as a derived quantity. It is narrower than the band and should not
  be conflated with it.

---

## D10 — Tense's l* is moved 21 → 28. **This is a post-hoc change to a pre-registered rule.**

**What the pre-registered rule said (spec §3.3):** l* is the *first* layer where
the (B)−(A) coordinate gap exceeds 2 SE and stays above 2 SE for at least three
consecutive layers, subject to lying in the workspace band.

**What it selected for tense:** layer 21, where E0.4 **fails** — the logit lens
shows a gap of 0.082 against the J-lens's 0.077, a ratio of 0.94. At l*=21 the
deflationary "this is output preparation, not workspace content" reading cannot
be excluded, so E2 run there would be uninterpretable whatever it found.

**Decision (Olivia, 2026-09-03): move tense's l* to 28.** At 28 the J/logit-lens
ratio is 4.56, the layer is inside the band, and it sits inside a genuine >2 SE
run (26–29).

**Justification.** The rule's *form* is sound, but for tense its output was
determined by the workspace band's lower edge rather than by the divergence
structure of the data:

1. **The band edge is the unvalidated input.** 21–46 comes from Olivia's prior
   study — loose, unpublished, and never validated against this measurement. The
   value 21 is exactly that edge. Tense has two separate >2 SE runs in band,
   21–24 and 26–29, and "first" picks the earlier one *only because* the edge was
   drawn at 21. Had the edge been drawn at 22 or 23 — equally defensible given
   its provenance — the same unmodified rule would have returned 26. An l* that
   flips on an arbitrary boundary is not carrying the evidential weight
   pre-registration is supposed to protect.
2. **The rule was written for a single clean divergence, which is what language
   has** (one isolated run, 24–32; the rule returns 24 and E0.4 passes at 5.72x,
   unmodified). Tense has two runs and the rule has no tie-break for that case.
3. **E0.4 is itself a pre-registered gate**, and it disqualifies 21. Choosing
   between two runs by using the other pre-registered control is a narrower move
   than free selection.

**What this change does *not* license, and must appear in the write-up:**

- It was made **after seeing that E0.4 failed at 21**. That is post-hoc, and no
  amount of justification converts it into a pre-registered choice. State it
  plainly rather than presenting 28 as the rule's output.
- **Tense cannot carry confirmatory weight equal to language.** Language's
  l*=24 came from the unmodified rule and passes E0.4 on its own; tense's did
  not. For H3, language is the anchor and tense is corroboration whose l* was
  chosen with knowledge of the outcome.
- The full in-band J/logit-lens profile for tense is reported (21: 0.9, 23: 1.1,
  25: 0.6, 27: 2.3, 29: 2.1, and 4.56 at 28) so a reader sees the whole selection
  landscape rather than the chosen point alone.

**Recorded but not run:** the cheap robustness check is to repeat E2 for tense at
27 and 29 — adjacent layers in the same run — and show the promoter set is stable
across them. If it is, the specific choice of 28 stops mattering. ~40 GPU-min.

---

## D11 — E3 behavioral metrics, and a correction to the H4 criterion

**Spec said (§4 E3):** continuation quality is "perplexity of the ground-truth
next 10 tokens", and for language "whether the continuation is still in Spanish
(langdetect on 20 sampled tokens)"; the dissociation succeeds if the coordinate
drops >=70% "while both behavioral metrics stay within 10% of baseline".

**Deviation 1 — no ground-truth continuations exist.** The stimuli are authored
and end mid-clause with no reference continuation. Substituted: the *baseline
model's own greedy 10-token continuation* is taken as the reference and its
perplexity is measured under ablation. This answers the intended question —
does the ablated model still find the unablated model's continuation likely —
without inventing a ground truth. Absolute values are not comparable across
properties (the reference is re-tokenized from decoded text, which does not
round-trip exactly); only the baseline→ablated ratio is used.

**Correction — the langdetect metric was initially left out of the pass rule.**
The first E3 run measured langdetect only *under ablation* (language: 0.53) and
did not measure a baseline, while `behaviour_within_10pct` checked only naming
accuracy and perplexity. A 0.53 ablated rate could have meant a large behavioural
degradation that the criterion would have missed, so "dissociation holds" for
language was not established by that run. Fixed: the baseline rate is now
measured too and included in the criterion.

**Outcome:** language's baseline langdetect rate is **also 0.533** — identical to
the ablated rate. The 53% is the metric's own floor, not ablation damage:
20 tokens is a very short sample for language identification. The dissociation
does hold for language, but the langdetect metric is weak and can only detect
large degradations. State that limit rather than presenting 0.53 → 0.53 as
strong evidence.

---

## D12 — The "fraction of the gap closed" estimator is changed to a ratio of means

**What was used through E3/E4:** the mean over passages of
`(m_B - m_ablated) / (m_B - m_A)` — a mean of per-passage ratios.

**Problem found (in E4b smoke testing):** the denominator is each passage's *own*
(B)−(A) gap, which is near zero for some passages. Those passages dominate the
mean and produce values that are not interpretable as a fraction: one sampled
set scored **357%**. The >100% overshoots reported in E3 (language 105.7%,
pos 150.9%) are partly this artifact rather than a real over-ablation.

**Decision:** the primary estimator becomes the **ratio of means**,
`(mean(m_B) - mean(m_ablated)) / (mean(m_B) - mean(m_A))`, which is bounded by the
aggregate effect and cannot be blown up by a single small-gap passage. The old
per-passage estimator is still computed and reported alongside it so the two are
comparable and the change is auditable.

**Consequence:** E3 was re-run for all three properties under the new estimator.
The qualitative E3 conclusions — a 2–4 node set suffices, random size- and
kind-matched sets do not, broadcast-band heads do not — do not depend on the
estimator, but the specific percentages do, and only the re-run numbers should be
quoted. E4's transfer matrix is a ratio *of* these fractions, so it is affected in
magnitude but not in sign or in the asymmetry it shows.

---

## G3 — E4b (sufficient-set family search) killed mid-run; pivot

Killed after ~13/18 free-mode runs on language. Partial, not to be quoted:
sampled sufficient sets were small (|S| = 2-4) and the smoke run had already found
a sufficient set *without* `gdn/24`. Superseded by a four-experiment plan
(Olivia) that attacks the granularity confound in E2 directly rather than
elaborating on sets built from it.

## D13 — E2's node granularity was not matched, and the comparison was unfair

A Gated DeltaNet block is a layer's **entire** token mixer. A single attention
head is **1/24** of one. E2 ranked them against each other as peers, so:

- the **top-promoter table** understated attention (one head vs a whole block);
- the **mass breakdown** ran the other way, summing `|attribution|` over 144
  separate head nodes against 19 whole GDN blocks, and since
  `|a| + |b| >= |a + b|`, splitting a block into 24 absolute-valued pieces
  inflates its total.

Both distortions are corrected by aggregating heads within a layer into one
attention block. Attribution is an inner product over disjoint slices of the same
tensor, so a layer's block attribution is exactly the **sum of its per-head
attributions** — recomputable from the cached arrays with no new forward passes.
Block-level *true* effects still need patching and are measured fresh.

**The hybrid-architecture claim in the write-up is provisional until this is
re-run at matched granularity, and is deleted if it does not survive.**

## G4 — Granularity gate: **the hybrid-architecture claim survives**

Re-done at matched block granularity for language (l*=24):

| | per-head (E2, unfair) | block-level (matched) |
|---|---|---|
| attn @ stimulus | 22.4% | **8.7%** |
| gdn @ stimulus | 23.7% | **31.3%** |
| mlp @ stimulus | 19.3% | **25.4%** |

Correcting the granularity moves mass **away** from attention, not toward it: the
old figure summed `|attribution|` over 24 head-sized pieces, which inflated it.
Totals are GDN 45.6%, MLP 44.0%, attention 10.5%.

Block-level true effects agree. Top 10 blocks: 5 GDN, 4 MLP, **1 attention**
(`attn/15`, −29%, rank 5). The top four (`gdn/20` −52%, `gdn/24` −36%, `gdn/9`
−33%, `mlp/24` −32%) are unchanged. Giving attention its whole layer roughly
doubles its best effect (`attn/15/13` −16% → `attn/15` −29%) and it is still
behind three GDN blocks.

**Verdict: keep the claim, now stated at block level.** Promotion routes
predominantly through GDN and MLP blocks, not attention.

---

## D14 — Pre-commitment for the direct-effect test (stated before running)

The "promoter" framing is only meaningful if these nodes *move the fact into*
the workspace rather than merely *writing the readout direction at l\**. Two
diagnostics, with the decision fixed in advance:

1. **Direct fraction.** Split each block's Δm into the *direct* path — the change
   in its own additive contribution to `h_l*(p*)`, projected on `Jhat_l*` — and
   the indirect remainder routed through later layers.
2. **Target mobility.** Re-run block attribution against a metric integrated over
   language's whole divergence window (sum of the coordinate at p* over layers
   24–32), so writers at layer 24 are not privileged.

**Pre-committed decision rule:** if the median direct fraction of the top-10
blocks is **>0.70** *and* the top-10 Jaccard between the l*=24 metric and the
integrated metric is **<0.30**, then these nodes are readout writers, the word
"promoter" is dropped from the write-up, and the project is re-framed as a
characterization of who writes the J-lens direction at the readout layer. If only
one of the two fires, the framing is qualified rather than dropped, and the
failing diagnostic is reported in full.

## G5 — Direct-effect gate: **the "promoter" framing survives** (D14 rule)

Neither pre-committed condition fired for language:

- **median direct fraction of the top-10 blocks = 15%** (threshold >70%). Only
  `mlp/24` is a pure readout writer (100% direct, writing at exactly l*=24);
  `gdn/20` and `gdn/24` are mixed (62%, 59%); and `gdn/9`, `gdn/21`, `gdn/14`,
  `mlp/17`, `mlp/10` are **~0% direct** — their entire effect is routed through
  later layers, which is what an upstream cause looks like.
- **top-10 Jaccard between the l*=24 metric and the integrated 24–32 metric =
  0.67** (threshold <0.30). Eight of ten blocks are shared.

**The two diagnostics agree with each other, which is the strongest part of this
result.** The blocks that were *direct* writers at layer 24 are exactly the ones
that lose their rank when the target moves off layer 24: `mlp/24` (100% direct)
and `gdn/24` (59% direct) both drop out of the top 10 under the integrated
metric, replaced by `attn/19` and `mlp/27`. The ~0%-direct blocks — `gdn/9`,
`gdn/14`, `gdn/21`, `mlp/17`, `mlp/10` — are stable across both targets.

**Consequences for the write-up:**
1. Keep "promoter", but restrict it to the **target-stable, mostly-indirect**
   blocks: `gdn/20`, `gdn/9`, `gdn/14`, `gdn/21`, `mlp/17`, `mlp/10`, `attn/15`.
2. Describe `mlp/24` as a **readout writer**, not a promoter. It writes the lens
   direction at l* and does nothing else.
3. **This weakens the earlier `gdn/24` "shared gate" claim.** `gdn/24` is 59%
   direct and drops out of the top 10 when the target moves off its own layer, so
   part of its apparent importance was privilege from measuring at l*=24. Any
   shared-gate claim must be re-checked against the target-stable set.

## G6 — E9: the boundary carrier is generic; the (B)-(A) contrast is not what it seemed

Unpatched Spanish coordinate at (l*=24, p*) on the same 30 Spanish passages:
A 1.7324 (0%), tense 1.9116 (**80%**), pos 2.1309 (**178%**), language 1.9557 (100%).

Boundary patch at L9: from A kills 28.0%, from tense **−14.5%**, from pos −8.0%,
identity 0.0% (exact-intervention control).

**The contrast this project measures is "asked to identify any property" vs
"predict the next word", not "asked about the language" vs not.** The carrier at
L9 transmits instruction-type, not property identity.

Not excluded: identifying the tense/POS *of a Spanish passage* may instrumentally
require recognising Spanish. Decisive follow-up is a linguistically contentless
property question (`selectivity-linecount` from the paper's release), ~30 GPU-min.
Until that is run, the generic-carrier claim is stated as the *leading* reading,
not as established.

---

## D15 — Pre-commitment for the contentless-question test (stated before running)

E9 showed that asking about tense or part of speech puts *Spanish* in the
workspace at 80% / 178% of what asking about the language does. Two readings
survive that result and it cannot separate them:

- **generic task-set** — any "identify a property" instruction opens the
  workspace, and the property asked about does not determine what enters;
- **linguistic analysis** — identifying the tense or part of speech *of a Spanish
  passage* instrumentally requires recognising that it is Spanish, so the carrier
  could still be property-specific.

Test: four questions that ask for a property requiring **no linguistic analysis**,
all exactly 13 tokens so they stay aligned with the existing prefixes:

    wordcount    "...identify how many words are in it."
    firstletter  "...identify the letter that it begins with."
    linecount    "...identify the number of lines it has."
    linewidth    "...identify the width of its longest line."   (nearest to the
                  paper's own selectivity-linecount task)

Measured: the Spanish coordinate at (l*=24, p*), as a share of the
language-vs-(A) gap; and the L9 boundary patch from each source.

**Pre-committed decision rule**, on the mean share across the four:

- **>= 0.50** — the generic task-set reading is established. §23 stands as
  written and strengthens: even a question with no linguistic content admits
  Spanish to the workspace.
- **< 0.20** — the generic reading is **retracted**. The E9 result then means
  the carrier serves linguistic analysis broadly, and "generic" overstates it.
- **0.20-0.50** — graded: report as a partial effect with the share, and state
  that neither pure reading holds.

The L9 patch is secondary evidence: if a contentless question's boundary state
substitutes for the language question's at no cost (as tense's did, -14.5%), that
supports the generic reading independently of the unpatched shares.

## G7 — D15 fires: **the "generic carrier" claim is retracted**

Spanish coordinate at (l*=24, p*), same 30 Spanish passages, as a share of the
language-vs-(A) gap:

| question | share |
|---|---|
| identify the **part of speech** | +178% |
| identify the **language** | +100% |
| identify the **tense** | +80% |
| (A) predict the next word | 0% |
| how many **words** | **−11%** |
| longest **line width** | **−63%** |
| how many **lines** | **−75%** |
| **first letter** | **−93%** |

Mean over the four contentless questions = **−60%**, far below D15's 0.20
retraction threshold. The L9 boundary patch agrees: patching from *first letter*
kills 25.7% where patching from (A) kills 28.0% — i.e. a contentless question's
boundary state is nearly as useless as no question at all — while patching from
tense (−14.5%) or part of speech (−8.0%) costs nothing.

**Corrected claim.** The carrier is neither generic task-set nor
property-specific. It is a **linguistic-analysis** signal: it is shared across
questions that require analysing the passage *as language* (language, tense,
part of speech, freely interchangeable at L9) and is absent from questions about
surface or orthographic properties, which actively *suppress* Spanish below the
no-question baseline.

§23 of the write-up is rewritten accordingly. The alternative reading E9 could
not exclude turns out to be the right one: asking for the tense or part of speech
of a Spanish passage elevates Spanish because that analysis requires recognising
the language, not because any instruction opens the workspace.

---

## G8 — E10 discarded; E11 supersedes it (arms + graded ladder + label matrix)

E10's binary contentless-vs-linguistic framing is thrown out at Olivia's
direction. Its linecount/wordcount cells are re-collected inside E11's graded
ladder, on the same passages and position, under a better design. The E10 files
remain in `results/` marked superseded; nothing is built on them.

## D16 — Two measurement errors found and fixed in E11

1. **Unpaired error bars.** All arms run on the same 30 passages and
   between-passage variance in the baseline coordinate is large and common to
   every arm. The first E11 pass reported the SE of each arm's raw mean (~0.24,
   about 90% of the whole (A)->(B_language) gap), which made every arm look
   null. Corrected to the SE of the **per-passage paired difference from (A)**,
   which is ~4x tighter and is the only correct comparison here.

2. **Wrong measurement position.** E11 first located the stimulus by
   `len(prompt) - len(passage)`. The passage retokenises at the prefix boundary,
   so this gave start=14 and p*=**24**, two tokens off the p*=**22** used in
   E0-E10. Corrected to the same diff-based rule `stimuli.load` uses. After the
   fix E11's shares reproduce E9/E10 exactly (linewidth -63%, wordcount -11%,
   tense 80%, pos 178%, language 100%), which cross-validates both.

Both errors were mine and both changed the numbers materially; the position bug
would have made the whole experiment incomparable with the rest of the project.
