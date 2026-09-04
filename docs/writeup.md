# What promotes a fact into the J-space? — running record

Working notes: methods, results, and reasoning as they accumulate. Structured to
follow the spec's §10 write-up skeleton so it feeds Olivia's prose directly.
**Not prose** — claims here are stated flatly with the numbers attached.
Gitignored. Every deviation and gate decision is in `DECISIONS.md`; this file
records what the numbers *mean*.

Last updated: after E11 (baseline arms, graded ladder, question x label matrix).

---

## 0. Status

| stage | state |
|---|---|
| Setup, lens, stimuli | done |
| E0.0 autograd through GDN | **pass** |
| E0.1 lens sanity | **pass** |
| E0.2 behavioral | pass (language, tense); POS 73% → 8 stimuli dropped |
| E0.3 workspace difference | pass for POS; language at ceiling; tense in-band negligible |
| E0.4 J-lens vs logit lens | pass for language, POS; **fail for tense** |
| E1 layer localisation | l* selected, see §4 |
| E2 attribution | **all three done**, K3 clear everywhere (r = 0.907 / 0.921 / 0.975) |
| E2 l* robustness | tense stable across l* = 27/28/29 (D10 check) |
| E4 overlap | above chance for all pairs, but **below H3's 3x prediction** |
| E3 causal + H4 | **done**; K4 does not fire; dissociation holds for language and tense, **fails for pos** |
| E4 cross-ablation | **done**; median off-diagonal transfer ~0 → **H3 not supported** |
| Figures | E0.1–E0.4, E1, E2 in `figures/` |

**Gate: K1 does not fire. Proceeding, with language as the headline property.**

---

## 1. Setup

**Model.** `Qwen/Qwen3.6-27B`, bf16, `attn_implementation="eager"`, loaded as
`Qwen3_5ForCausalLM` (the text tower; drops `model.visual.*` and `mtp.*`).
64 layers, d_model 5120, vocab 248320, 24 attention heads, head_dim 256, 4 KV
heads. **Hybrid**: 16 full-attention layers at indices 3, 7, …, 63 (`i % 4 == 3`)
and 48 Gated DeltaNet linear-attention layers. H1 is only testable at the 16
full-attention layers; elsewhere the question→stimulus route can only be a
recurrent GDN carry, which has no attention pattern to inspect. This is a
question the paper could not ask about Claude.

**Lens.** `neuronpedia/jacobian-lens`, revision `qwen-n1000`, commit
`16a01f30`, file `Qwen3.6-27B_jacobian_lens_n1000.pt` (3,303,032,772 bytes).
Verified on load: `n_prompts == 1000`, `d_model == 5120`, 63 source layers
(0–62) of shape [5120, 5120] — so the transport target is the **final** layer
(Neuronpedia convention), not the penultimate one the paper and Neel's
replication use. Must be stated in the write-up.

**Prompts are raw text, not chat-templated** (DECISIONS D1) — the lens is fit on
raw wikitext and the paper's §3.5 protocol is a plain passage, so the chat
template would add the single largest lens-validity confound for no gain.

**Metric.** `c_t(l,p) = <h_l(p), Jhat_l[t]>` with
`Jhat_l[t] = normalise(J_l^T (g * W_U[t]))`, where `g` is the final RMSNorm gain
(DECISIONS D6). Folding in `g` makes the coordinate rank tokens exactly as the
lens's own `apply()` does while keeping it **linear in h** — which is what E2's
attribution backward requires. Rank is the secondary metric; see §5 for why that
distinction turned out to matter more than expected.

**Stimuli.** 30 passages per property, authored for this project, each ending
mid-clause in the paper's style. Question wordings came out naturally parallel
at **exactly 13 tokens each**, so no filler padding was needed:

```
A         Read the following passage and predict the next word that it requires.
language  Read the following passage and identify the language it is written in.
tense     Read the following passage and identify the tense it is written in.
pos       Read the following passage and identify the part of speech required next.
```

All 90 pairs pass the alignment assertion (identical token counts *and*
elementwise-identical stimulus token ids — DECISIONS D4, required because E2
patches position-by-position). p* = 22 for every stimulus, comfortably above the
lens's unfitted-below-16 boundary (`jlens.fitting.SKIP_FIRST_N_POSITIONS`).

---

## 2. Positive controls (E0.0, E0.1)

**E0.0 — autograd reaches layer 0 through 24 GDN layers**, gradient norm 12.6,
all finite. This was the single most likely silent failure: attribution patching
requires a backward pass through the recurrent linear-attention stack, and the
GDN kernels resolve through a hub-fetch-with-torch-fallback path. It works.

**E0.1a — the J-lens leads the logit lens by 16 layers.** On the walkthrough's
two-hop prompt (*"the currency used in the country shaped like a boot"*), read at
the `boot` token: the J-lens surfaces ` Italy` at **L23** and holds it through
L51; the logit lens does not produce it until **L39**, and at mid layers reads
pure noise (`'shaw'`, `'举世'`, `'-shaped'`). This is the paper's core
qualitative claim, reproduced on the open model.

**E0.1b — late-layer agreement 0.62** mean top-5 overlap with the logit lens
across 10 wikitext snippets, above the 0.60 bar and well clear of K0's 0.40.

The 4B dry run (DECISIONS D2) reproduced all three on `Qwen3.5-4B` — same hybrid
architecture, `Italy` at L9 vs never for the logit lens — and caught three bugs
before they cost 27B time.

---

## 3. Does the workspace difference exist? (E0.2, E0.3, E0.4)

n = 30 per property (POS 22 after E0.2 filtering), p* = 22, raw text.

| property | E0.2 naming | E0.3 all layers | E0.3 band 21–46 | l* in band | E0.4 at l* |
|---|---|---|---|---|---|
| **language** | 100% | 1.43 | 1.47 | **24** | J 0.223 vs LL 0.039 → **5.72x pass** |
| tense | 100% | 3.31 | 0 vs 5 cells | 21 → **28** (D10) | at 21: J 0.077 vs LL 0.082 → **0.94x fail**; at 28: **4.56x** |
| pos | 73% → 22 kept | 35.17 | 39.67 | **30** | J 0.291 vs LL 0.103 → **2.82x pass** |

*(POS's l* and E0.4 both move once the 8 E0.2 failures are dropped: l* 29→30,
n 30→22, ratio 2.14x→2.82x. Any POS number quoted from before the filtering is
stale.)*

**K1 does not fire**: it requires E0.3 to fail for all three *or* E0.4 to fail for
all three. Neither holds.

**K2 fires on the spec's unrestricted l* rule** — it selected layers 0, 2, 2, the
noisy early-layer regime the R-lens post warns about. The pre-registered in-band
fallback (DECISIONS D8) gives l* = 24 / 21 / 29.

---

## 4. Where the fact enters (E1)

Contiguous runs where the (B)−(A) coordinate gap at p* exceeds 2 SE for >=3
consecutive layers:

| property | in-band divergence window | out-of-band runs |
|---|---|---|
| language | **24–32** (9 layers, isolated) | 0–2, 5–7 (early-layer artifact) |
| tense | 21–24, 26–29 | 2–4, 11–13 |
| pos | 29–46, continuing to 62 | 2–5 |

Language is the clean case: one isolated 9-layer block inside the band and
nothing anywhere else in the stack. J/logit-lens separation is 5.72x at l*=24 and
stays >=1.6x across the whole run.

POS's run extends to layer 62 — to the output. That is the signature of output
preparation rather than a localised workspace event (deflationary explanation
#2), even though its E0.4 ratio passes at l*=29.

---

## 5. The finding that changed the reading: rank saturates for language

The all-layer E0.3 ratio of 1.43 looks like language failing. It is not, and the
first explanation offered — dilution of a 9-layer effect across 63 layers — was
**wrong**: restricting to the band gives 1.47, and to language's own 24–32
window, 1.49.

The actual cause is a **ceiling effect in the rank criterion**. The passage is in
Spanish, so ` Spanish` is already inside the top 25 at 722 in-band (layer,
position) cells under the (A) "predict the next word" question. An instruction
cannot raise a rank that is already high. What the instruction moves is the
*magnitude*: coordinate gap 0.223 [0.117, 0.343] against a logit-lens gap of
0.039 [−0.014, 0.094] whose CI includes zero.

This is why spec §3.1 makes the raw coordinate primary and rank secondary, and
language is the case that demonstrates it. **Reporting only the top-25 ratio
would misrepresent the cleanest property as the weakest one.** Both must be
reported, with the ceiling stated.

The mirror-image caution applies to POS: its rank ratio is enormous (39.67) but
its coordinate gap is mid-sized and its divergence runs to the output — a large
rank effect is not automatically a workspace effect either.

---

## 6. Per-property standing going into E2

- **language — headline.** E0.2 100%, isolated 24–32 window, strongest J/LL
  separation, rank at ceiling for a comprehensible reason. l* = 24.
- **pos — secondary, with a caveat.** Largest rank effect and E0.4 passes, but
  the run to layer 62 and the 73% naming accuracy both point at output
  preparation. Carried for H3 generality; conclusions about it need the E0.4
  control re-stated at every layer used.
- **tense — weakest.** Fails E0.4 at its own l* (the logit lens shows the *same*
  gap at layer 21), its in-band rank counts are 0 vs 5 (too small for a ratio),
  and its all-layer 3.31 comes entirely from out-of-band layers. Its l* also sits
  exactly on the band's lower edge, so it is edge-sensitive.

---

## 7. Deflationary explanations — status

| # | explanation | status |
|---|---|---|
| 1 | Just in-context conditioning | open — this is what E2's H1-vs-H2 attribution split decides |
| 2 | Lens artifact / output preparation | **controlled for language** (E0.4 5.72x, p* is 10+ tokens before any answer position); **not excluded for tense** at l*=21; **live concern for POS** given the run to L62 |
| 3 | Ablation collateral | pending E3 — 5-seed layer-matched random node sets |
| 4 | Attribution is first-order only | pending E2 — every node verified by activation patching, K3 checks the correlation |

---

## 8. Open questions

1. ~~Tense's l\*~~ — **resolved: moved 21 → 28** (DECISIONS D10). The write-up
   must state that this was post-hoc, made after seeing E0.4 fail at 21, and that
   tense therefore cannot carry confirmatory weight equal to language, whose
   l*=24 came from the unmodified rule. The justification is that the rule's
   output for tense was fixed by the *unvalidated* band edge, not by the data:
   tense has two in-band >2 SE runs (21–24, 26–29) and "first" picks the earlier
   only because the edge was drawn at 21. An edge of 22–23 would have returned 26
   from the same unmodified rule. Outstanding robustness check: repeat E2 at 27
   and 29 to show the promoter set is stable across the run (~40 GPU-min).
2. **Whether POS survives as a generality test** given its output-preparation
   signature, or whether H3 rests on language + tense only.
3. Band stays 21–46, unadjusted — reasoning in DECISIONS G2.


---

## 9. E2 — what promotes the fact (language, l*=24, n=30)

`figures/e2_attribution_language.png`. m(B) = 1.956, m(A) = 1.732, gap = 0.223.

**Attribution is reliable here.** Attribution-patching estimate vs true
activation-patching effect: **r = 0.907** across 46 candidates. K3 (fires below
0.4) does not fire, and 42/46 candidates survive the sign and 25%-magnitude
filters. Deflationary explanation #4 is addressed: no node is called a promoter
on a first-order estimate alone.

### The H1-vs-H2 breakdown

| node type | at stimulus positions | at question positions |
|---|---|---|
| Gated DeltaNet block | **23.7%** | 10.8% |
| full-attention head | 22.4% | 9.5% |
| MLP block | 19.3% | 14.2% |

**No node type dominates.** This is the headline mechanistic result so far, and
it cuts against H1's clean prediction.

### Top verified promoters (drop in the coordinate when counterfactually ablated)

| node | true Δm | ±SE | share of the (B)−(A) gap |
|---|---|---|---|
| `gdn/20` | −0.116 | 0.022 | 52% |
| `gdn/24` | −0.080 | 0.021 | 36% |
| `gdn/9` | −0.073 | 0.022 | 33% |
| `mlp/24` | −0.072 | 0.038 | 32% |
| `mlp/17` | −0.061 | 0.028 | 27% |
| `gdn/14` | −0.047 | 0.025 | 21% |
| `attn/11/0` | −0.046 | 0.014 | 21% |

**The strongest promoter is not an attention head.** It is a Gated DeltaNet block
at layer 20, and the top six are GDN and MLP blocks; the best full-attention head
(`attn/11/0`) ranks seventh. Reading against the pre-registered hypotheses:

- **H1 as stated is not supported.** Its prediction was "top attribution nodes are
  attention heads". They are not, and only 32% of the attribution mass sits on
  full-attention heads at all — barely more than GDN blocks (35%) and about the
  same as MLPs (34%).
- **H2 gets partial support.** `mlp/24` — an MLP block exactly at l* — is the
  fourth-strongest promoter at 32% of the gap, with `mlp/17`, `mlp/18`, `mlp/19`
  behind it. The MLP-side transform H2 predicts is present, but it is not the
  whole story either.
- **The hybrid-architecture question the paper could not ask has an answer, and it
  is not the inspectable one.** Roughly as much promotion mass routes through the
  *recurrent* GDN carry as through full-attention heads. Whatever moves the fact
  into the J-space is substantially not a question→stimulus attention read; it is
  a recurrent state carry with no attention pattern to inspect. Attribution
  graphs would be the natural follow-up, and this is the concrete reason why.

### Caveats that constrain how far this can be pushed

1. **Individual effects are modest and overlapping.** The largest single node
   accounts for 52% of the gap and the top seven sum to well over 200%, so
   effects are not additive. Whether a *small set* suffices is exactly what E3's
   greedy construction tests; this table cannot answer it.
2. **The verified-by-kind counts are an artifact of the candidate caps** (top-30
   heads, top-8 MLPs, top-8 GDN). "27 attention heads verified" reflects the cap
   allocation, not relative importance. The mass breakdown is the unbiased
   statement; the counts are not.
3. Every number here is for **language only**. H3 needs POS (running) and tense
   (blocked on the l* question in §8.1).

---

## 10. Figures

| file | shows |
|---|---|
| `e0_1_lens_sanity.png` | J-lens surfaces ` Italy` at L23 vs L39 for the logit lens; per-snippet top-5 overlap against the 0.60 bar and K0's 0.40 |
| `e0_2_behavioral.png` | naming accuracy per property against the 90% bar, with stimuli kept |
| `e0_3_cells.png` | top-25 cell counts, (A) vs (B), all layers and band; and where in the stack the cells are |
| `e0_4_lens_vs_logitlens.png` | the J-lens vs logit-lens gap at each l*, bootstrap CIs, with pass/FAIL ratios |
| `e1_layer_curves.png` | **the E1 figure**: coordinate curves for (A) and (B) with per-passage SE, and the gap-over-SE curve, band and l* marked |
| `e2_attribution_language.png` | attribution mass by node type, the attribution-vs-patching diagnostic, and verified promoters |

All figures regenerate from saved results with `python3 scripts/figures.py` — no
GPU needed. Annotations are computed from the arrays being plotted, never read
from a previously written summary (an earlier version annotated POS with a stale
pre-filtering ratio).


---

## 11. E2 across all three properties

| property | l* | n | (B)−(A) gap | attr-vs-patching r | verified |
|---|---|---|---|---|---|
| language | 24 | 30 | 0.223 | 0.907 | 42/46 |
| tense | 28 (D10) | 30 | 0.204 | 0.921 | 45/46 |
| pos | 30 | 22 | 0.291 | 0.975 | 41/46 |

**K3 does not fire anywhere.** Attribution patching is a good approximation in
this model, r = 0.91–0.98 across 138 verified candidate nodes.

### The attribution mass ordering replicates across all three properties

| at stimulus positions | language | tense | pos |
|---|---|---|---|
| Gated DeltaNet | **24%** | **27%** | **25%** |
| MLP | 19% | 21% | 22% |
| full-attention head | 22% | 21% | 18% |

Three properties, three different l*, three different stimulus sets — and the
same ordering every time, with GDN blocks carrying the largest single share of
promotion mass and full-attention heads never leading. **H1 as pre-registered
("top attribution nodes are attention heads") is not supported for any property.**
Top-8 promoter lists are GDN- and MLP-dominated throughout; the best attention
head ranks 7th (language), 8th (tense), and outside the top 8 (pos).

The strongest single reading of E2 so far: promotion in this model routes
substantially through the **recurrent** linear-attention state, not through an
inspectable question→stimulus attention read. That is a claim the paper could not
make about Claude, and it is the concrete argument for attribution graphs as the
follow-up — the mechanism sits where per-head circuit analysis cannot reach.

---

## 12. Is tense's post-hoc l* load-bearing? (D10 robustness check)

E2 repeated at l* = 27, 28 and 29 — adjacent layers inside the same >2 SE run.

| pair | Jaccard | top-5 Jaccard | shared | Spearman ρ of effects |
|---|---|---|---|---|
| 27 vs 28 | 0.69 | 0.67 | 37 | 0.91 |
| 27 vs 29 | 0.57 | 0.43 | 32 | 0.86 |
| 28 vs 29 | 0.74 | 0.25 | 37 | 0.87 |

Mass breakdown is essentially invariant (GDN at stimulus 25–27%, MLP 21–24%,
attention 20–24%), and `gdn/24` is the top promoter at all three layers, with
`gdn/21`, `mlp/10` and `mlp/17` in every top-8.

**Verdict: the promoter *set* is stable; the fine *ranking* is not.** All pairwise
Jaccards are >= 0.57 and effect-size rank correlations are 0.86–0.91, so the
conclusions drawn from tense do not depend on having picked 28. But top-5 Jaccard
falls to 0.25 for 28 vs 29, so claims about *which node is second or third*
should not be made from tense. The measured gap also shrinks with depth
(0.234 → 0.204 → 0.121), making 29 the weakest measurement point of the three.

This substantially defuses the post-hoc concern in D10 — but it does not erase
it, and the write-up should present the sweep rather than only l* = 28.

---

## 13. H3 preview — one shared gate, or many? (overlap half of E4)

Pools differ across properties because l* differs, so every comparison is
restricted to nodes at layers <= min(l*_a, l*_b), with the random baseline drawn
from that same restricted pool (1000 draws).

| pair | shared | Jaccard | chance | ratio | p |
|---|---|---|---|---|---|
| language vs tense | 17 | 0.288 | 0.110 | **2.6x** | <0.001 |
| language vs pos | 12 | 0.207 | 0.101 | **2.1x** | 0.004 |
| tense vs pos | 13 | 0.183 | 0.106 | **1.7x** | 0.024 |

**Overlap is reliably above chance for all three pairs, but every ratio falls
below H3's pre-registered >=3x prediction.** On the overlap criterion alone, H3
as stated is not met — though the "property-specific promoters, overlap at
chance" alternative is also clearly refuted. This is the middle outcome the spec
anticipated in E4: a small shared core plus property-specific machinery.

Six nodes are verified promoters for all three properties, and they are not
equally interesting:

| node | language | tense | pos |
|---|---|---|---|
| **`gdn/24`** | **−36%** | **−41%** | **−30%** |
| `gdn/9` | −33% | −11% | −8% |
| `gdn/22` | −20% | −4% | −10% |
| `attn/15/13` | −16% | −3% | −9% |
| `attn/11/0` | −21% | −9% | −1% |
| `attn/11/1` | −7% | −3% | −1% |

`gdn/24` is the one genuinely shared *strong* promoter: a top-3 node for every
property, accounting for 30–41% of the (B)−(A) gap in each. The others are strong
for language and marginal elsewhere, so they are shared in the set-membership
sense without being shared in any load-bearing sense.

**This is the single most promising thread for the write-up**, and it is not yet
established: H3's other half is cross-ablation (ablate P_i, measure property j,
require >=50% transfer). Overlap of *membership* is much weaker evidence than
transfer of *effect*. Whether `gdn/24` is a shared gate or three coincident
property-specific uses of one block is exactly what E3/E4 must decide.

---

## 14. E3 — a very small set suffices, and it dissociates from behaviour

| property | P_prop | gap closed | random matched (5 seeds) | broadcast-band heads |
|---|---|---|---|---|
| language | `gdn/20, gdn/24, gdn/9` | **105.7%** | 19.2% | −1.1% |
| tense | `gdn/24, gdn/28, gdn/21, mlp/10` | **76.6%** | −6.9% | −13.4% |
| pos | `gdn/25, mlp/25` | **150.9%** | −17.6% | 1.3% |

**K4 does not fire.** Two to four nodes drive the coordinate to the (A) level in
every property, and the mandatory size- and kind-matched random baselines get
nowhere near it — two of three are *negative*, i.e. random counterfactual
ablation moves the coordinate the wrong way. Broadcast-head sets from the band do
essentially nothing, so the promoters are **not** just the paper's broadcast
heads (§4.3.2). Every P_prop is GDN-dominated, consistent with E2.

Overshoot (105.7%, 150.9%) means ablation pushes the coordinate *below* the (A)
level. The intervention is not a clean "restore to (A)": these nodes carry other
signal too, and effects are non-additive.

### H4 — the dissociation

| property | coordinate drop | naming accuracy | ref-continuation ppl | still Spanish | dissociation |
|---|---|---|---|---|---|
| language | 105.7% | 100% → 100% | 162.8 → 162.7 | 0.533 → 0.533 | **holds** |
| tense | 76.6% | 100% → 100% | 15.3 → 15.4 | — | **holds** |
| pos | 150.9% | **100% → 82%** | 10.8 → 10.8 | — | **fails** |

For language and tense the fact is removed from the J-space at p* while the model
still names the property and still assigns the same likelihood to its own
continuation — H4's prediction, and the result that separates "promotion" from
"use". For **pos** naming accuracy drops 18 points, past the 10% bar: its
"promoters" are partly the fact circuit itself, which is what its
output-preparation signature predicted from E0.4 onward.

Caveat on the langdetect metric (D11): language's baseline rate is 0.533, the
same as ablated, so 20 tokens is too short a sample for it to be a sensitive
test. It rules out large degradation, nothing finer.

---

## 15. E4 — H3 is not supported

Transfer matrix, normalised by each property's own within-property drop:

| measured ↓ / ablated → | P_language | P_tense | P_pos |
|---|---|---|---|
| **language** | 1.00 | **1.72** | n/a |
| **tense** | −0.11 | 1.00 | −0.28 |
| **pos** | 0.22 | −0.31 | 1.00 |

`n/a`: both nodes of P_pos sit at layer 25, above language's l* = 24, so they are
downstream of the readout and the cell is vacuous by construction — not a null
result. Excluding it, the off-diagonals are **1.72, 0.22, −0.11, −0.28, −0.31**:
mean 0.25, **median −0.11**.

**H3's prediction (>=50% transfer) fails; the mean is carried entirely by one
cell.** Combined with overlap ratios of 2.6x / 2.1x / 1.7x — all below the
pre-registered >=3x — the "one shared gate" reading is not supported. The strict
"property-specific, overlap at chance" alternative is not right either: overlap
was reliably above chance. The honest summary is **largely property-specific
promotion with a small shared core**.

### Two findings inside the matrix that matter more than the headline

1. **The transfer is asymmetric.** Ablating P_tense closes 172% of *language's*
   gap — more than language's own set does — while ablating P_language closes
   −11% of tense's. A shared gate would be symmetric. This is not that.
2. **P_prop is *a* sufficient set, not *the* promotion set.** That asymmetry is
   only possible because several different small GDN sets each suffice to
   collapse the coordinate: `{gdn/24, gdn/21, mlp/10}` disrupts language more
   than `{gdn/20, gdn/24, gdn/9}` does. The greedy prefix returns one arbitrary
   representative of a family of sufficient sets, so **overlap and transfer
   between P_i and P_j are comparisons between arbitrary representatives**, and
   both statistics understate any true sharing. This is the most important
   methodological caveat in the project and it limits how hard H3 can be pushed
   in either direction.

`gdn/24` remains the one node with a genuine claim to being shared: top-3 by E2
attribution in all three properties, present in both P_language and P_tense, and
the likely cause of the one large off-diagonal.

---

## 16. Hypothesis scorecard

| ID | prediction | outcome |
|---|---|---|
| H1 | top attribution nodes are attention heads reading the question | **not supported** — GDN blocks lead in all three properties; best head ranks 7th/8th/outside top 8 |
| H2 | an MLP at/below workspace onset is required | **partial** — MLPs carry 19–22% of mass and `mlp/25` is half of P_pos, but GDN blocks dominate every P_prop |
| H3 | shared promotion circuit (>=3x overlap, >=50% transfer) | **not supported** — overlap 1.7–2.6x, median transfer ≈ 0, transfer asymmetric |
| H4 | promotion separable from use | **holds for language and tense**, fails for pos |

The GDN result is the finding with the most weight: promotion in this hybrid model
routes primarily through recurrent linear-attention state, where no attention
pattern exists to inspect. That is why H1 could not be confirmed and why
attribution graphs are the natural next step rather than a nicety.

---

## 17. Not run

- **E5 path patching** (optional in the spec) — would test H1's "reads the
  instruction" claim directly. Lower value now that H1 is unsupported, though
  path-patching the question tokens into `gdn/24`'s input would sharpen §15.
- Chat-template arm (D1 secondary).
- The full E4 overlap × transfer analysis over a *family* of sufficient sets
  rather than one greedy representative — the fix for the §15.2 caveat.


---

# PIVOT — four experiments that re-found the result

E4b (sufficient-set family sampling) was killed mid-run. The four experiments
below replaced it, each with a kill gate. Figure: `figures/pivot_experiments.png`.

## 18. Granularity fix — the comparison in E2 was unfair, and correcting it
## strengthens the finding

A GDN block is a layer's **entire** token mixer; an attention head is **1/24** of
one. E2 ranked them as peers. Both of E2's statistics were distorted, in opposite
directions (D13): the promoter table understated attention (one head vs a whole
block), while the mass breakdown *overstated* it, because summing `|attribution|`
over 24 head-sized pieces inflates a block's total (`|a|+|b| >= |a+b|`).

Attention aggregated to block level, language, l*=24:

| @ stimulus positions | per-head (E2) | block level (matched) |
|---|---|---|
| attention | 22.4% | **8.7%** |
| GDN | 23.7% | **31.3%** |
| MLP | 19.3% | **25.4%** |

Totals: GDN 45.6%, MLP 44.0%, attention 10.5%. Block-level *true* effects agree:
the top 10 blocks are 5 GDN, 4 MLP and **one** attention block (`attn/15`, −29%,
rank 5). Giving attention its whole layer roughly doubles its best effect
(`attn/15/13` −16% → `attn/15` −29%) and it still trails three GDN blocks.

**Gate: the hybrid-architecture claim survives and is now stated at block level.**

## 19. Direct-effect decomposition — the "promoter" framing survives, but the set
## splits in two

Pre-committed rule (D14): drop "promoter" if the top blocks are >70% direct path
**and** the set changes wholesale when the readout target moves. Neither fired.

**Median direct fraction of the top 10 = 15%.** The split is the interesting part:

| block | Δm | direct | fraction direct |
|---|---|---|---|
| `mlp/24` | −0.072 | −0.072 | **100%** |
| `gdn/20` | −0.116 | −0.072 | 62% |
| `gdn/24` | −0.080 | −0.047 | 59% |
| `gdn/22` | −0.045 | −0.018 | 41% |
| `attn/15` | −0.065 | −0.016 | 24% |
| `mlp/10`, `gdn/21`, `gdn/9`, `mlp/17`, `gdn/14` | −0.047 … −0.073 | ~0 | **≈0%** |

**Top-10 Jaccard between the l*=24 metric and a metric integrated over the whole
24–32 window = 0.67** (8 of 10 shared).

**The two diagnostics agree, which is the strongest part of this result.** The
blocks that are *direct* writers at layer 24 are exactly the ones that lose rank
when the target moves off layer 24: `mlp/24` (100% direct) and `gdn/24` (59%)
both fall out of the top 10 under the integrated metric, replaced by `attn/19`
and `mlp/27`. The ≈0%-direct blocks — `gdn/9`, `gdn/14`, `gdn/21`, `mlp/17`,
`mlp/10` — are stable under both targets.

So the set is two different things:
- **readout writers** — `mlp/24` above all: they write the lens direction at l*
  and do nothing else. Calling them promoters was wrong.
- **upstream promoters** — `gdn/20`, `gdn/9`, `gdn/14`, `gdn/21`, `mlp/17`,
  `mlp/10`, `attn/15`: target-stable, and acting almost entirely through later
  layers.

**This retracts part of the `gdn/24` shared-gate claim in §13/§15.** `gdn/24` is
59% direct and drops out when the target moves, so some of its apparent
importance was privilege from measuring at its own layer.

## 20. The mechanism experiment — where "report this" crosses into the stimulus

The instruction can only reach stimulus positions by crossing from question
positions, and in this hybrid that crossing happens inside either a
full-attention layer (K/V at question positions) or a GDN recurrent state. One
uniform intervention per layer (`scripts/e7_boundary.py`): replace the token
mixer's **input at question positions** with the (A) run's — so the GDN boundary
state, or the K/V the stimulus reads, comes from (A) — then **restore the mixer's
output at question positions** to its (B) values, so the question-position
residual is untouched downstream and the claim is about layer *l* alone.

Fraction of the (B)−(A) gap killed, language, layers 0–24:

| layer | kind | kills |
|---|---|---|
| **L9** | **GDN** | **28.0%** |
| L24 | GDN | 15.7% |
| L15 | attention | 10.4% |
| L8 | GDN | 10.3% |
| L10 | GDN | 10.0% |
| L12, L18 | GDN | 9.1% |
| L14 | GDN | 6.9% |

Best single attention layer **10.4%**; best single GDN layer **28.0%**; seven of
the top eight are GDN.

**The crossing is distributed, and it is carried mainly by recurrent state.** No
single layer carries the instruction — the largest contribution is 28% — but the
carriers are overwhelmingly GDN layers, concentrated at **L8–L12, well below the
workspace band and well below l\***, with a second contribution at L24.

This converges with §19 in a way neither experiment could establish alone:
**`gdn/9` is both the single largest boundary carrier (28%) and a ≈0%-direct
promoter.** Its whole causal role is to carry the instruction across the
question→stimulus boundary at layer 9; its effect on the coordinate at layer 24
is entirely indirect. That is a mechanism, not a correlation, and it is the
answer to the question §9.1 of the paper left open — for this model.

## 21. Re-entry — the fact is removed at l*, and repaired eight layers later

Ablating P_language at stimulus positions and reading the full (layer × position)
grid:

| layer at p* | 24 | 28 | 32 | 36 | 40 | 44 | 52 |
|---|---|---|---|---|---|---|---|
| fraction of the (B)−(A) coordinate retained | **−6%** | 58% | **78%** | 77% | 81% | 81% | 40% |

- **Re-entry by depth: layer 32.** The coordinate is fully removed at l*=24
  (−6%, marginally below the (A) level), recovers to 58% by L28 and crosses 70%
  at **L32** — the exact top of language's divergence window (24–32).
- **Re-entry by position: never.** At layer 24 the coordinate does not recover at
  any later stimulus position.

So the repair is **vertical, not horizontal**: downstream layers re-derive the
language from stimulus content that was never ablated, rather than the fact
re-entering at a later token. This explains H4 cleanly — the model still names
Spanish (100% under ablation) because the fact is back in the workspace by L32,
long before the output. The dissociation in §14 is not evidence that the
workspace is behaviourally inert; it is evidence that **ablating one window of a
redundant pipeline gets repaired downstream.** That is a materially weaker claim
than "promotion is separable from use", and the write-up must say so.

## 22. Revised scorecard

| ID | outcome after the pivot |
|---|---|
| H1 | **not supported** — at matched block granularity attention holds 10.5% of mass and one top-10 slot; the boundary crossing is carried by GDN layers (28% vs 10.4% best-single) |
| H2 | **partial, and sharpened** — `mlp/24` is a pure readout writer (100% direct), not a promoter; `mlp/17` and `mlp/10` are genuine ≈0%-direct upstream promoters |
| H3 | **not supported, and partly retracted** — `gdn/24`'s shared-gate status is undercut by §19 |
| H4 | **holds, but reinterpreted** — the dissociation exists because the fact is repaired by L32, not because the workspace is inert |

**The headline is §20.** Promotion in this model is a distributed, recurrent
carry from the instruction into stimulus positions, concentrated in GDN layers
8–12, invisible to per-head attention analysis, and followed by downstream
repair if it is removed.

---

## 23. E11 — baseline arms, the graded ladder, and the question × label matrix

Supersedes E9's "generic carrier" claim and E10's binary contentless test (both
discarded, G8). Nine question arms, same 30 Spanish passages, no patching, one
forward pass each, all coordinates read at (l*=24, p*=22). Errors are the SE of
the **per-passage paired difference from (A)** — the arms share passages, so the
large between-passage variance cancels and the unpaired SE is meaningless (D16).

### The positive control: **promotion, not suppression** — but underpowered

| arm | Δ Spanish vs (A) | ±SE | t | share of the language arm |
|---|---|---|---|---|
| passage only\* | −0.185 | 0.369 | −0.5 | −83% |
| **neutral instruction** | **+0.091** | 0.064 | **1.4** | **41%** |
| (A) predict the next word | 0 | — | — | 0% |
| **language** | +0.223 | 0.059 | **3.8** | 100% |

\* no prefix, so its absolute position differs from every other arm; never
significant at either offset and not interpretable.

A task-free instruction ("Read the following passage.") sits **41% of the way**
from (A) to the language question and is not significantly above (A) (t=1.4).
So next-word prediction is **not** actively evicting the fact — the story is
promotion, and the paper's framing survives with the generality qualification.
But 41% is not zero either: by the pre-registered reading this is the
"in between → it's both" case, and it is **underpowered**, not resolved. n=30
gives no power to separate 41% from either end. Resolving it needs more passages,
not more analysis.

### The ladder is a step, not a dose-response

| arm | Δ Spanish | t | share |
|---|---|---|---|
| longest line width (contentless) | −0.140 | −1.5 | −63% |
| how many words (contentless, textual) | −0.025 | −0.3 | −11% |
| register (linguistic, not language-dependent) | −0.108 | −1.2 | −49% |
| tense (language-adjacent) | +0.179 | 1.3 | 80% |
| language | +0.223 | **3.8** | 100% |
| part of speech | +0.398 | **3.2** | 178% |

Only **language** and **part of speech** clear |t| ≥ 2. There is no monotonic
dose-response: the three low arms are flat-to-negative and statistically
indistinguishable from each other, then the grammatical arms jump. **`register`
is the informative one** — it is a genuinely linguistic question that is not
about the language, and it sits at −49%, with the contentless arms. So the
dividing line is not "linguistic vs not"; it is something narrower, and tense's
80% (t=1.3) is suggestive but unproven.

**The POS puzzle you flagged holds up and sharpens.** Its wording is the closest
of any arm to (A)'s ("...required next" vs "...the next word that it requires"),
yet it produces the largest effect of all (178%, t=3.2). That is evidence against
a wording-similarity account and for task demand scaling the effect.

### The matrix — the anchor result

Paired Δ vs (A) at (l*, p*), `figures/e11_arms_matrix.png`:

| arm | Spanish | past | formal | informal | English | table | purple | **adjective** | random (24 tok) |
|---|---|---|---|---|---|---|---|---|---|
| neutral | 0.09 | −0.08 | 0.00 | −0.02 | −0.06 | −0.10 | −0.12 | −0.10 | −0.02 |
| line width | −0.14 | −0.10 | 0.02 | 0.04 | −0.07 | −0.06 | −0.08 | −0.04 | −0.01 |
| register | −0.11 | −0.03 | **0.16** | **0.27** | 0.03 | 0.00 | 0.00 | 0.05 | −0.01 |
| tense | 0.18 | **0.21** | 0.24 | 0.18 | 0.20 | 0.16 | −0.01 | −0.04 | −0.11 |
| language | **0.22** | 0.11 | 0.17 | 0.16 | 0.20 | 0.10 | 0.08 | −0.04 | −0.08 |
| part of speech | **0.40** | 0.29 | 0.36 | 0.25 | 0.34 | 0.27 | 0.17 | **−0.03** | −0.15 |

Four things, in order of how much they change the story:

1. **Under a grammatical question, every passage-global property rises together.**
   The tense arm raises *Spanish* (0.18) as much as it raises *past* (0.21). The
   language arm raises *past*, *formal* and *informal* alongside *Spanish*. The
   workspace is not being loaded selectively with the property asked about.
2. **The answer token never rises.** ` adjective` is flat or negative under every
   arm including its own (−0.03 under the POS question). *But this is expected
   and is not evidence against selection*: p* is mid-passage and the required
   part of speech is a property of the word **after** the passage, undetermined
   at p*. It should be read as a scope limit on the design, not a finding.
3. **The rise is content, not rescaling.** The residual norm moves ≤3.4% across
   all arms, and the 24 random control tokens move the **opposite** way (−0.15
   under POS, −0.08 under language) while Spanish rises. Spanish − random is
   +0.55 (POS), +0.30 (language). The scale confound is excluded.
4. **Selectivity is real but weak.** Spanish rises more than the irrelevant
   content words, but not by much — 0.22 vs 0.20 (` English`) and 0.10 (` table`)
   under the language arm. ` English` is a *foil* for a Spanish passage and it
   tracks Spanish closely throughout.

### What this does to the paper's claim

On this model, "the specific property enters the workspace under a name-it
instruction" is too strong. What the evidence supports is: **a grammatical
question raises a broad set of passage-global properties together, along with
unrelated content words, while pushing random tokens down — and the question
selects the output rather than the workspace content.** Register and the
contentless questions do not do this at all, so the effect is not generic to
instructions either.

### Limits that bound all of the above

- Only `language` and `pos` are individually significant; the ladder's middle
  (tense, register) is unresolved at n=30.
- One passage set, one property label per column, one model, single-token labels.
- The neutral arm's 41% is the open question the write-up must not paper over.

