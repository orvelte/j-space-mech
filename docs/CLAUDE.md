# CLAUDE.md — J-space promotion mechanism

Working spec: [claude_jspace-promotion-mechanism-spec.md](claude_jspace-promotion-mechanism-spec.md).
Every deviation from the spec gets a `DECISIONS.md` entry **before** proceeding.
Every stop-point report must state which E-number and which hypothesis row it serves.

## Division of labor
- Claude Code: all empirics (E0–E5), figures, `EXPERIMENT_INDEX.md`.
- Olivia: all prose.

## Non-optional requirements
- Counterfactual (not zero) ablation.
- 5-seed random baselines, layer-matched and size-matched.
- Attribution verified by activation patching before any node is called a "promoter".
- Per-passage bootstrap CIs on everything.
- Logit-lens comparison at ℓ*.

## Reading list (tier 1 only)
J-space paper §3.1, §3.2, §3.5, §4.3.2, §9.1; Neel's replication post; the
attribution-patching section of the reference used for per-head hooks.
Do **not** load `default_600k.md` into agent context.

## Environment
- `export HF_HOME=/workspace/.cache/huggingface` in every script. `/` has only
  ~99 GB free; the 27B is 55.6 GB. Everything lands on `/workspace`.
- Lens: `neuronpedia/jacobian-lens`, revision `qwen-n1000` (commit `16a01f30`),
  file `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`
  (3,303,032,772 bytes). The same branch also holds a file *without* `_n1000` —
  the `n_prompts == 1000` assertion is what catches picking the wrong one.

## Model facts (verified against transformers 5.16.1 `modeling_qwen3_5.py`)
- 64 layers, d_model 5120, vocab 248320, 24 attn heads, head_dim 256, 4 KV heads (GQA).
- **Full-attention layers: indices 3, 7, 11, …, 63** (`i % 4 == 3`), 16 of 64.
  H1 is only testable there. The other 48 are Gated DeltaNet (recurrent, no
  attention pattern to inspect).
- `AutoModelForCausalLM` → `Qwen3_5ForCausalLM` (drops `model.visual.*`,
  `mtp.*`); `jlens` auto-detects it as `Layout("model")`.
- Attention applies an output gate `attn_output * sigmoid(gate)` **before**
  `o_proj`. Per-head decomposition is still exact (gate is elementwise over head
  dims): hook the gated pre-`o_proj` tensor, slice
  `o_proj.weight[:, h*256:(h+1)*256]`. `o_proj` in-features = 24 × 256 = 6144.
- `DecoderLayer.forward` returns a bare tensor → `ActivationRecorder` gets the
  residual stream cleanly. `Qwen3_5Attention.forward` returns a tuple.
- GDN kernels resolve via `use_kernel_func_from_hub_with_fallback` with pure-torch
  fallbacks. E2 needs a **backward** through the 48 GDN layers — E0.0 asserts
  autograd reaches the layer-0 residual.
- mRoPE: `position_ids` are 4-way expanded (text/temporal/h/w). Only matters for a
  custom forward using `inputs_embeds`.
- `jlens.fitting.SKIP_FIRST_N_POSITIONS == 16` → all measurement positions p* must
  be ≥ 16 in the full prompt.
- Memory is not a constraint at ≤60 tokens: the retained graph is a few hundred MB
  against ~25 GB headroom. No activation checkpointing or layer chunking needed.

---

## Section 2 of the spec, verbatim

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


---

## Section 5 of the spec, verbatim

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

