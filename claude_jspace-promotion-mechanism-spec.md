# Project Spec: What Promotes a Fact into the J-Space?

**Budget:** 6–10 GPU-hours on one A100-80GB, ~10 human-hours
**Model:** Qwen3.6-27B (the checkpoint the public pre-fitted J-lens was built on — verify hash before anything else, see E0.1)
**One-line goal:** Causally identify the nodes (attention heads and/or MLPs) that move a task-relevant fact into the J-space when an instruction asks the model to report it, and test whether those nodes are *shared* across different facts.

**What an expert learns from this if it works:** Gurnee et al. showed the *same* fact enters the J-space under a "name it" instruction and not under a "just continue" instruction (§3.2, §3.5), and explicitly left the mechanism open (§9.1). Nobody has published a causal account since. The interesting question is not "which heads promote *Spanish*" but whether there is one gate or many: a shared promotion circuit across unrelated properties is the strong global-workspace prediction; property-specific promoters cut against it. Either outcome is a relative claim across conditions and properties, which is the kind that is hard to dismiss.

---

## 1. Background and prior work

### 1.1 What the paper established
- **J-lens:** for each vocab token and layer, a direction whose activation encodes the model's potential to verbalize that token later. Fit by averaging Jacobians from residual stream at layer ℓ to the final-layer logit direction over a corpus. Reads *content*; says nothing about *entry*.
- **Task-dependence of workspace contents (§3.5, Fig. 11):** paired-question protocol. Same stimulus passage, two preceding questions: (A) "predict the next word" vs (B) "name the property" (language, tense, register, part of speech). Property label appears in top lens tokens at far more stimulus positions under (B). Since stimulus tokens are identical, the difference is caused by the question alone.
- **Directed modulation (§3.2):** "hold X in mind" instructions put X in the J-space. Metacognitive tokens (*thinking, imagine, focused*) appear at *earlier* layers than the content itself. Hint: an instruction-derived "performing a mental operation" representation precedes content arrival.
- **Broadcast heads (§4.3.2, §A.19):** a top-1% subset of workspace-layer heads whose OV circuits selectively relay J-directions between positions, concentrated in the first half of the workspace layers. Weights-based and fact-agnostic — identifies carriers of J-content in general, not promoters of a specific fact under a specific instruction. Ablating them degrades injected-thought report.
- **Clamp result (§3.1):** clamping J-lens coordinates to clean-pass values blocks re-entry; the non-J-space component's effect on report goes to ~0. So "in the J-space" is a causally meaningful state, not just a readout artifact.
- **§9.1, verbatim gap:** "We have not characterized what causes a representation to enter it."

### 1.2 What replications tell us about the 27B open model
- Neel Nanda's replication on Qwen3.6-27B: verbal-report swap effect "weak but positive"; directed modulation "moderate success"; workspace layer bands "notably less clean" than the paper's, plausibly 2–3 overlapping bands. **Consequence:** effect sizes here will be smaller than the paper's. All thresholds below are set relative to a within-model baseline, not the paper's numbers.
- R-lens post (Aug 2026): J-lens readouts are noisy at early layers because errors accumulate through the backprop from the final layer. **Consequence:** do not define the target node at the earliest layer where anything appears. Define it at the *first layer where the two conditions reliably diverge*, and require that layer to be inside the workspace band (E1).

### 1.3 Why not MOLTs / attribution graphs
Attribution graphs need a feature dictionary (transcoders or MOLTs) for the model. None is public for Qwen3.6-27B; training one is a multi-day job. Public transcoders exist only for small models where the existence of a workspace is unknown (§9.1 lists this as open). Node-level attribution over heads and MLP blocks answers the same question ("attention vs MLP, which layer, which positions") without a dictionary. Attribution graphs are a follow-up, not a prerequisite.

---

## 2. Hypotheses and pre-registered predictions

Define the **promotion set** P_prop for a property (e.g. *Spanish*) as the smallest set of nodes whose ablation in condition (B) drives the property's J-lens coordinate at stimulus positions down to the condition-(A) level.

| ID | Hypothesis | Prediction if true | Prediction if false |
|----|-----------|-------------------|--------------------|
| H1 | Promotion is done by attention: heads at stimulus positions attend to the question tokens and write an instruction-conditioned direction that lands in the J-space directly. | Top attribution nodes are attention heads; their query positions are stimulus tokens; their key positions are question tokens. Ablating them removes the fact from the J-space. | Attention nodes have attribution but ablating them alone doesn't clear the J-space; an MLP block is required. |
| H2 | Promotion involves an MLP-side transform: attention carries an instruction signal, but an MLP at layer L rotates the existing (non-J-space) fact representation into J-space. | An MLP block at a layer at or just below workspace onset carries large attribution *and* its ablation clears the J-space coordinate even when the instruction-reading heads are left intact. | MLP ablations at workspace onset have effect comparable to random layer-matched MLP ablations. |
| H3 (headline) | There is a shared promotion circuit across properties (attentional-selection reading of GWT). | Jaccard overlap of P_lang, P_tense, P_pos is ≥3× the overlap expected from random node sets of the same sizes, and cross-ablation (ablate P_lang while measuring tense) transfers ≥50% of the within-property effect. | Overlap at chance; cross-ablation transfers <20%. |
| H4 (dissociation) | Promotion is separable from use: removing the fact from the J-space leaves the model's task behavior intact. | After ablating P_prop, language-continuation perplexity and language-ID accuracy each stay within 10% of baseline while the J-space coordinate drops ≥70% of the (B)−(A) gap. | Behavior degrades in lockstep with the coordinate → the "promoters" are just the fact circuit itself. |

**Pre-registered deflationary explanations (must be addressed in write-up regardless of result):**
1. **It's just in-context conditioning.** A head copies an instruction-derived direction; nothing special about "workspace entry". If H1 holds and H2 fails, this is the honest reading, and it's still a result (it means workspace entry has no dedicated gate — a negative for the GWT framing). Report it as such.
2. **Lens artifact.** The (B)−(A) difference could reflect the lens direction for *Spanish* partly encoding "about to output the token Spanish" (motor content) rather than workspace content. Control: measure at a stimulus position ≥8 tokens before any position where the model would emit the label, and check the logit lens on the same position — if the logit lens shows the same difference, the J-lens isn't adding anything and the finding may be about output preparation.
3. **Ablation collateral.** Ablating any k nodes in a 27B model perturbs the residual stream. Every ablation is compared against ≥5 seeds of layer-matched random node sets of equal size.
4. **Attribution patching is a first-order approximation.** Top nodes are verified by activation patching before any claim.

---

## 3. Setup

### 3.1 Model and lens
- Qwen/Qwen3.6-27B, bf16, HF transformers with forward hooks on every token-mixing-block output, every MLP-block output, and the residual stream at every layer.
- **Lens (verified source):** `neuronpedia/jacobian-lens`, file `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` (3.3 GB, fp16, keys `J` / `n_prompts` / `d_model`; one [d_model, d_model] matrix per source layer). Fit by Neuronpedia + mntss (Anthropic) on 1000 wikitext prompts of 128 tokens — the paper's own fitting scale. Pin revision `qwen-n1000`. Load with `jlens.JacobianLens.from_pretrained(...)` from the official `anthropics/jacobian-lens` repo; `jlens.from_hf(hf_model, tok)` wraps the model and handles the hybrid architecture.
- **Convention notes that affect the design:**
  - Target layer is the **final** layer (Neuronpedia convention), not penultimate (the paper's default for Claude; Neel's replication used penultimate). Fine for our purposes, but state it in the write-up.
  - Lens is unfitted at positions <16 (jlens convention). All measurement positions p\* must be ≥16 in the full prompt — the question template guarantees this.
  - Wikitext-fit; chat-template prompts are a stated limitation. Run with `enable_thinking=False` and a minimal chat template, and include an E0 check that lens readouts on the question tokens are sensible.
  - The qwen3-32b entry in the same repo was uploaded as an unfinalised checkpoint (see HF discussion #3); on load, assert the 27B file has key `J` and `n_prompts == 1000`, not `jacobian_sum`.
- **Architecture consequence (important):** Qwen3.6-27B is a hybrid — 64 layers, 48 Gated DeltaNet linear-attention layers and 16 full softmax-attention layers (every 4th). "Attention head reading the question tokens" (H1) is only cleanly testable at the **16 full-attention layers**, where per-head key/query positions exist. GDN layers are recurrent: they carry cross-position information but have no attention pattern to inspect. So node types for attribution are: {full-attn head at layer L, GDN block at layer L, MLP block at layer L} × position. H1 is refined: *does the promotion signal route through full-attention heads (inspectable, question→stimulus) or through GDN state (a recurrent "carry")?* This is itself a non-obvious question the paper couldn't ask about Claude and is worth a sentence in the write-up.
- J-lens vector for token t at layer ℓ: Ĵ_ℓ[t] = normalise(J_ℓᵀ W_U[t]) — the direction in layer-ℓ residual space whose linear transport lands on t's unembedding. Compute once per (t, ℓ) and cache.
- **J-lens coordinate** for token t at layer ℓ, position p: c_t(ℓ,p) = ⟨h_ℓ(p), Ĵ_ℓ[t]⟩ where Ĵ is the unit-normalised lens vector. Report both the raw coordinate and the rank of t among all vocab tokens' coordinates (the paper's "top-25" criterion). Raw coordinate is the primary metric (continuous, differentiable — needed for attribution patching); rank is secondary.
- Also compute the logit-lens coordinate (W_U row) at the same (ℓ,p) as the deflationary control.

### 3.2 Properties and stimuli
Three properties, taken from the paper's own §3.5 list so results are comparable:
1. **Language** — target token *Spanish* (backup: *French*). Stimuli: 30 Spanish passages, 20–40 tokens.
2. **Tense** — target token *past*. Stimuli: 30 English past-tense passages.
3. **Part of speech** — target token *adjective*. Stimuli: 30 passages whose next word is syntactically forced to be an adjective.

Two conditions per stimulus, prompt templates fixed before running:
- **(A) continue:** "Read the following passage and predict the next word.\n\n{passage}"
- **(B) name:** "Read the following passage and identify the {language / tense / part of speech required next}.\n\n{passage}"

Question tokens are the *only* thing that differs. Stimulus token positions are aligned by construction. Keep question length within ±2 tokens across A/B so position indices at the stimulus are comparable (pad with a neutral phrase if needed; log the exact templates in DECISIONS.md).

### 3.3 Measurement position and layer
- **Position p\*:** the stimulus token at index 10 (counting from stimulus start), or the last stimulus token if shorter. Fixed in advance. Also report a curve over all stimulus positions for the figure.
- **Layer ℓ\*:** chosen in E1 as the *first* layer where the (B)−(A) coordinate gap is >2 SE across passages **and** stays >2 SE for at least 3 consecutive layers. All attribution is computed against the coordinate at (ℓ\*, p\*). ℓ\* must lie in Neel's reported workspace band for this model; if it doesn't, see K2.

---

## 4. Experiments

### E0 — Positive controls (must all pass before E1)
Behavioral checks first (cheap), internals checks second (one cached forward pass each).

- **E0.1 Lens sanity (GPU, 20 min):** download the Neuronpedia file, assert keys `J`/`n_prompts`/`d_model` with `n_prompts == 1000` and 63–64 layer matrices of shape [5120, 5120]. Run the official walkthrough's boot-currency prompt through `lens.apply` and confirm *Euro/lira* appear mid-stack before the logit lens shows them. On 10 wikitext snippets check late-layer J-lens top-5 overlaps ≥60% with logit-lens top-5. Confirms the file, the `jlens` hooks on the hybrid architecture, and the readout convention all agree.
- **E0.2 Behavioral (API or local, 15 min):** under (B), model names the correct property ≥90% of stimuli for each of the three properties. Under (A), it produces a plausible continuation. Drop stimuli that fail either.
- **E0.3 Workspace-difference exists (GPU, ~30 min):** one forward pass per stimulus per condition (180 passes). For each property, the property token's J-lens rank at stimulus positions is in the top-25 at ≥2× as many (layer, position) cells in (B) as in (A), per the paper's protocol. Report per-passage bootstrap CIs. **This is the gate: if the difference isn't there, there is nothing to localise.**
- **E0.4 Lens vs logit lens:** at the candidate ℓ\*, the (B)−(A) gap measured by J-lens must exceed the gap measured by logit lens by ≥1.5×. Otherwise the phenomenon is output preparation, not workspace content (see deflationary #2).

### E1 — Localise the divergence (~0.5 GPU-h)
Using E0.3 activations, plot the property coordinate at p\* across all layers for (A) and (B) with per-passage SE. Select ℓ\* by the rule in §3.3. Also record the first layer where metacognitive tokens (*identify, name, thinking, language*) enter the top-25 under (B) — if this is reliably below ℓ\*, that's corroborating evidence for the "instruction representation precedes content" ordering (§3.2) and a candidate for where the promotion signal originates.

### E2 — Attribution over nodes (~2 GPU-h)
Metric m = c_prop(ℓ\*, p\*). Clean run = condition (B); corrupted run = condition (A) for the same passage.
- **Attribution patching:** one clean forward, one corrupted forward, one backward of m in the clean run. For every attention-head output and every MLP-block output at every position and every layer ≤ ℓ\*, estimate Δm ≈ ⟨∇m, a_corrupt − a_clean⟩. Head outputs are obtained by hooking the per-head slice before the output projection (o_proj is linear, so per-head contributions are well-defined). Memory: sequences are ≤60 tokens, so backprop through 27B in bf16 with activation checkpointing fits on 80 GB; if not, chunk layers.
- Average attribution over the 30 passages per property. Rank nodes. Keep the top 30 heads and top 8 MLP blocks per property.
- **Verification by activation patching:** for the top 30 heads + top 8 MLPs, patch each node's (A)-run activation into the (B) run individually and measure true Δm. Nodes whose true effect has the wrong sign or is <25% of the attribution estimate are dropped. Report the correlation between attribution and true patching effect (a diagnostic the paper's reviewers asked for and a reusable sanity figure).
- Output per property: a sorted list of verified promoter candidates with (layer, head or MLP, position, true Δm), and a breakdown of attribution mass into {attention at stimulus positions attending to question tokens, attention elsewhere, MLP at stimulus positions, MLP at question positions}. This breakdown is the H1-vs-H2 answer.

### E3 — Causal validation and the behavioral dissociation (~1.5 GPU-h)
For each property, greedily build P_prop: add verified nodes in order of true effect until ablating the set (replacing with (A)-run activations — counterfactual ablation, not zero) drives c_prop(ℓ\*, p\*) to within 30% of the (A) level. Record |P_prop|. Cap at 20 nodes; if 20 doesn't reach the threshold, report the achieved fraction and stop.

Controls (mandatory, 5 seeds each):
- Random node sets, layer-matched and size-matched.
- Random *heads only* from the paper-style broadcast-head band, to check whether promoters are just broadcast heads.

Behavioral readouts under ablation of P_prop (H4):
- Language / tense / POS identification accuracy under (B) (model still names the property?).
- Continuation quality under (A): perplexity of the ground-truth next 10 tokens, and — for language — whether the continuation is still in Spanish (langdetect on 20 sampled tokens).
- Pre-registered success for the dissociation: coordinate drops ≥70% of the (B)−(A) gap while both behavioral metrics stay within 10% of baseline.

Note: it is *expected* that under (B) the model may still name the language even with the fact removed from the J-space at p\* — it can re-derive it later in the sequence. The interesting behavioral readout is whether the final answer changes, and where in the sequence the fact re-enters (rerun the lens over all positions under ablation and report the re-entry position). Re-entry downstream would itself be evidence of a repair mechanism and is worth a figure.

### E4 — Generality across properties (~1 GPU-h) — the headline
- Jaccard overlap between P_lang, P_tense, P_pos, against the distribution of overlaps for random size-matched sets (1000 draws). Report overlap at the head level and at the layer level separately.
- Cross-ablation matrix: ablate P_i, measure coordinate drop for property j, normalised by the within-property drop. A 3×3 matrix with off-diagonals near 1 = shared gate; near 0 = property-specific.
- Also check whether the shared component (if any) is dominated by the *instruction-reading* nodes (heads attending to the question) versus the *transform* nodes (MLPs at stimulus positions). A plausible and interesting middle outcome: instruction-reading heads are shared, transform nodes are property-specific.

### E5 (optional, only if ≥1.5 h remain) — Path patching from question tokens
For the top 5 shared heads: path-patch question-token residual streams into their key/value inputs only, to confirm the promotion signal originates at the instruction rather than at the stimulus. Confirms H1's "reads the instruction" claim directly.

---

## 5. Kill gates

| Gate | Trigger | Action |
|------|---------|--------|
| K0 | Lens file fails the key/`n_prompts` assertion, `jlens.from_hf` doesn't support the loaded revision, or E0.1 overlap <40% | Stop. Fit own lens with `jlens.fit` on ≥100 wikitext prompts only if ≤2 GPU-h (cost model: ~1 fwd + d_model/dim_batch bwd per prompt); else fallback. |
| K1 | E0.3 fails (ratio <2× for all three properties) or E0.4 fails for all three | Stop. The phenomenon isn't measurable in this model with this lens. Write up the null with the E0 figures (it's a real replication datapoint on Neel's own model). |
| K2 | ℓ\* falls outside the workspace band, or divergence never reaches 2 SE for 3 consecutive layers | Restrict to the property(ies) that pass; if none, K1. |
| K3 | Attribution-vs-patching correlation <0.4 on verified nodes | Attribution is unreliable here; switch to exhaustive activation patching on heads at layers ℓ\*−4…ℓ\* only (budget permitting) or stop. |
| K4 | P_prop needs >20 nodes to reach 70% of the gap, and random size-matched sets achieve ≥50% of that | Promotion is diffuse. Report as such; skip E4 cross-ablation, keep overlap analysis. |
| K5 | Halfway through budget (5 GPU-h) with E2 not complete | Halt-don't-heal. Write up E0–E1 as a characterisation result. |

Every gate decision goes in `DECISIONS.md` with the numbers that triggered it.

---

## 6. Bad-null checklist (what makes a null uninterpretable)

- [ ] Lens convention wrong (E0.1)
- [ ] Prompt templates change stimulus positions (fixed lengths, §3.2)
- [ ] p\* too close to output position → motor contamination (index-10 rule, E0.4)
- [ ] ℓ\* in the noisy early-layer regime (R-lens caveat, E1 rule)
- [ ] Attribution patching linearity failure (E2 verification, K3)
- [ ] Zero-ablation artifacts (counterfactual ablation only)
- [ ] No random-node baseline (5 seeds, layer-matched, mandatory)
- [ ] Single property → generalisation claims impossible (three properties from the start)
- [ ] No per-passage error bars (bootstrap CIs on everything)

---

## 7. Hour budget (10 h, human + GPU interleaved)

| Hours | Work |
|-------|------|
| 0–1 | Repo, `CLAUDE.md`, load model + lens, E0.1, E0.2, stimulus set finalised |
| 1–2 | E0.3, E0.4, E1 — **decision point K1/K2** |
| 2–4.5 | E2 attribution + patching verification (all three properties) |
| 4.5–5 | **Decision point K3/K5.** Write interim notes. |
| 5–6.5 | E3 causal validation + random baselines + behavioral readouts |
| 6.5–7.5 | E4 overlap + cross-ablation |
| 7.5–8 | E5 if time; otherwise start figures |
| 8–10 | Write-up (Olivia): figures, pre-registration table filled in with outcomes, deflationary explanations addressed, limitations |

---

## 8. Fallback project (if K0/K1 fire in hour ≤2)

**"How well does the public J-lens replicate the task-dependence result on Qwen3.6-27B?"** Run the full paired-question protocol from §3.5 on all four of the paper's properties with error bars, random-token baselines, and a logit-lens comparison at every layer. Neel's own replication was qualitative; a quantitative version with controls is a modest but honest contribution and directly serves his stated complaint about missing baselines. ~3 GPU-h.

---

## 9. Division of labor and `CLAUDE.md` requirements

- Claude Code: all empirics (E0–E5), figures, `EXPERIMENT_INDEX`. Olivia: all prose.
- Paste §2 (hypotheses table + deflationary explanations) and §5 (kill gates) **verbatim** into `CLAUDE.md`. Every stop-point report must state which E-number and which hypothesis row it serves.
- Named requirements that are not optional: counterfactual (not zero) ablation; 5-seed random baselines; attribution verified by patching before any node is called a "promoter"; per-passage bootstrap CIs; logit-lens comparison at ℓ\*.
- Any deviation → `DECISIONS.md` entry before proceeding.
- Do not load `default_600k.md` into agent context. Tier-1 reading for the agent: J-space paper §3.1, §3.2, §3.5, §4.3.2, §9.1 only; Neel's replication post; the attribution-patching section of whichever reference is used for the per-head hook implementation.

---

## 10. Write-up skeleton (for MATS / SPAR)

1. The open question (§9.1 quote) and why "one gate or many" is the informative version.
2. Setup + positive controls, with the E0.3/E0.4 figure.
3. Where the fact enters (E1 layer curve, metacognitive-first check).
4. What promotes it: attention vs MLP breakdown, verified node list, attribution-vs-patching diagnostic (E2).
5. Causal validation and the behavioral dissociation, with random baselines and re-entry positions (E3).
6. Generality: overlap and cross-ablation matrix (E4).
7. Deflationary explanations, each addressed with the corresponding control.
8. Limitations: single 27B model, three properties, single-token labels, lens fidelity at ℓ\*, no feature-level decomposition (attribution graphs as the natural next step).
