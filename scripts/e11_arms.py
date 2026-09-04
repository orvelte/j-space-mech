"""E11/E12/E13 — baseline arms, the graded ladder, and the question x label matrix.

All three share one set of forward passes: for each (question arm, passage) we run
the model once and read the J-lens coordinate of *every* label at p*. No patching.

Arms, in increasing "how much linguistic analysis does this demand":

    none        the passage alone, no prefix at all
    neutral     "Read the following passage." (task-free), padded to 13 tokens
    A           "...predict the next word that it requires."
    linewidth   "...identify the width of its longest line."          contentless
    wordcount   "...identify how many words are in it."               contentless, textual
    register    "...identify the register it is written in."          linguistic, not language-dependent
    tense       "...identify the tense it is written in."             language-adjacent
    pos         "...identify the part of speech required next."
    language    "...identify the language it is written in."

E11 asks where `none` and `neutral` sit relative to (A) and (B_language):
  both near (B) -> the (A) instruction *suppresses*, and layer 9 carries a
                   narrowing signal;
  both near (A) -> genuine promotion, the paper's framing stands;
  in between    -> both, and the L9 effect can be decomposed against `neutral`.

**Position caveat.** The `none` arm has no prefix, so a stimulus-relative offset
of 10 lands at absolute position ~11, below the lens's fitted floor of 16
(`jlens.fitting.SKIP_FIRST_N_POSITIONS`). Every measurement is therefore reported
at two stimulus offsets: 10 (the project's p*, valid for the prefixed arms only)
and 16 (valid for every arm, `none` included). The `none` arm's absolute position
still differs from the prefixed arms' at the same offset, which is a real
confound and is stated wherever `none` is quoted.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jspace import hooks, lens_ops, model as jmodel, stimuli

NEUTRAL = "Read the following passage. It is reproduced in full just below."
LADDER = ["none", "neutral", "A", "linewidth", "wordcount", "register",
          "tense", "pos", "language"]
LABELS = {"Spanish": " Spanish", "past": " past", "adjective": " adjective",
          "noun": " noun", "formal": " formal", "informal": " informal",
          # foils / controls: none of these is the answer to any question asked.
          "English": " English", "table": " table", "purple": " purple",
          "Thursday": " Thursday"}
#: 24 random vocab tokens, used as a null distribution: if the "rise" under a
#: question is a global rescaling of the residual rather than content entering the
#: workspace, random tokens rise with it.
N_RANDOM = 24
OFFSETS = [10, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--out", default="results/e11_arms.json")
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    l_star = jmodel.L_STAR["language"]
    layers = list(b.lens.source_layers)
    pre = dict(stimuli.build_prefixes(b.tok))
    pre["Q:neutral"] = NEUTRAL
    pre["Q:linewidth"] = "Read the following passage and identify the width of its longest line."
    pre["Q:wordcount"] = "Read the following passage and identify how many words are in it."
    pre["Q:register"] = "Read the following passage and identify the register it is written in."
    n_pre = {k: len(b.tok(v).input_ids) for k, v in pre.items()}
    assert len(set(n_pre.values())) == 1, n_pre
    print("all prefixes are", set(n_pre.values()), "tokens", flush=True)

    key = {"neutral": "Q:neutral", "A": "A", "linewidth": "Q:linewidth",
           "wordcount": "Q:wordcount", "register": "Q:register",
           "tense": "B:tense", "pos": "B:pos", "language": "B:language"}
    label_ids = {k: b.tok.encode(v)[0] for k, v in LABELS.items()}
    rng = np.random.default_rng(0)
    rand_ids = [int(x) for x in rng.choice(b.hf.lm_head.weight.shape[0], N_RANDOM,
                                           replace=False)]
    for i, t in enumerate(rand_ids):
        label_ids[f"rand{i:02d}"] = t
    all_labels = list(label_ids)

    # cache lens directions once: [label][layer]
    jh = {k: ({l: lens_ops.jhat(b, i, l) for l in layers} if k in LABELS
              else {l_star: lens_ops.jhat(b, i, l_star)})
          for k, i in label_ids.items()}

    passages = json.load(open(stimuli.DATA))["language"]
    # coords[arm][label][offset] -> list over passages; also full layer profile at offset 10
    coords = {a: {k: {o: [] for o in OFFSETS} for k in all_labels} for a in LADDER}
    profile = {a: {k: [] for k in LABELS} for a in LADDER}
    norms = {a: {o: [] for o in OFFSETS} for a in LADDER}   # ||h|| at p*, the scale control

    for pi, text in enumerate(passages):
        for arm in LADDER:
            prompt = text if arm == "none" else f"{pre[key[arm]]}\n\n{text}"
            ids = b.model.encode(prompt)
            # Locate the stimulus the same way stimuli.load does -- by where two
            # differently-prefixed prompts stop differing. A length-based rule is
            # wrong: the passage retokenises at the prefix boundary, which put an
            # earlier version of this script at position 24 instead of 22 and made
            # its coordinates incomparable with every other experiment.
            if arm == "none":
                start = 1
            else:
                ref = b.model.encode(f"{pre['A']}\n\n{text}")
                other = b.model.encode(f"{pre['B:language']}\n\n{text}")
                start = int((ref[0] != other[0]).nonzero().flatten()[-1]) + 1
            with torch.no_grad():
                with hooks.NodeCache(b, layers=layers) as c:
                    b.model.forward(ids)
                    for o in OFFSETS:
                        p = min(start + o, ids.shape[1] - 1)
                        norms[arm][o].append(float(c.resid[l_star][0, p].float().norm()))
                    for k in all_labels:
                        for o in OFFSETS:
                            p = min(start + o, ids.shape[1] - 1)
                            coords[arm][k][o].append(
                                float(c.resid[l_star][0, p].float() @ jh[k][l_star]))
                        # full-depth profile only for the named labels; the random
                        # control tokens have a direction cached at l* alone.
                        if k in LABELS:
                            p10 = min(start + 10, ids.shape[1] - 1)
                            profile[arm][k].append(
                                [float(c.resid[l][0, p10].float() @ jh[k][l])
                                 for l in layers])
        if (pi + 1) % 10 == 0:
            print(f"  {pi+1}/{len(passages)} passages", flush=True)

    npass = len(passages)
    mean = {a: {k: {o: float(np.mean(coords[a][k][o])) for o in OFFSETS}
                for k in all_labels} for a in LADDER}

    def paired(a, k, o):
        """Mean and SE of the *per-passage* difference from (A).

        All arms run on the same 30 passages, and between-passage variance in the
        baseline coordinate is large and common to every arm, so it cancels in the
        paired difference. The unpaired SE of each arm's mean is ~90% of the whole
        (A)->(B_language) gap and says nothing.
        """
        d = np.array(coords[a][k][o]) - np.array(coords["A"][k][o])
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(npass))

    ref = {k: {o: paired("language", k, o)[0] for o in OFFSETS} for k in all_labels}

    for o in OFFSETS:
        print(f"\n=== Spanish at (L{l_star}, offset {o}), paired against (A) ===")
        print(f"  {'arm':<11}{'Δ vs (A)':>11}{'±SE':>8}{'t':>7}"
              f"{'share of lang':>15}{'  ||h|| Δ%':>11}")
        for a in LADDER:
            d, e = paired(a, "Spanish", o)
            share = d / ref["Spanish"][o] if ref["Spanish"][o] else float("nan")
            dn = (np.mean(norms[a][o]) / np.mean(norms["A"][o]) - 1) * 100
            note = "  [abs. pos differs]" if a == "none" else ""
            print(f"  {a:<11}{d:>11.4f}{e:>8.4f}{d/e if e else 0:>7.1f}"
                  f"{share:>14.0%}{dn:>10.1f}%{note}")

    print(f"\n=== question x label, paired Δ vs (A) at offset 10, in units of that"
          f" label's own language-arm Δ ===")
    cols = [k for k in LABELS]
    print(f"  {'arm':<11}" + "".join(f"{c[:9]:>10}" for c in cols) + f"{'rand mean':>11}")
    matrix = {}
    for a in LADDER:
        row = {}
        for k in all_labels:
            d, e = paired(a, k, 10)
            row[k] = {"delta": d, "se": e,
                      "share_of_language_arm": d / ref[k][10] if ref[k][10] else None}
        rmean = float(np.mean([row[f"rand{i:02d}"]["delta"] for i in range(N_RANDOM)]))
        rsd = float(np.std([row[f"rand{i:02d}"]["delta"] for i in range(N_RANDOM)], ddof=1))
        row["_random_null"] = {"mean": rmean, "sd": rsd}
        matrix[a] = row
        print(f"  {a:<11}" + "".join(f"{row[c]['delta']:>10.3f}" for c in cols)
              + f"{rmean:>11.3f}")

    print("\n=== scale control: is the rise content, or just a bigger residual? ===")
    print(f"  {'arm':<11}{'Spanish Δ':>11}{'random Δ (mean±sd)':>22}{'||h|| Δ%':>11}"
          f"{'Spanish − random':>18}")
    for a in LADDER:
        sp = matrix[a]["Spanish"]["delta"]
        rn = matrix[a]["_random_null"]
        dn = (np.mean(norms[a][10]) / np.mean(norms["A"][10]) - 1) * 100
        print(f"  {a:<11}{sp:>11.3f}{rn['mean']:>13.3f}±{rn['sd']:<8.3f}{dn:>10.1f}%"
              f"{sp - rn['mean']:>18.3f}")

    json.dump({"l_star": l_star, "offsets": OFFSETS, "arms": LADDER,
               "labels": LABELS, "mean": mean, "matrix": matrix,
               "per_passage": {a: {k: {str(o): coords[a][k][o] for o in OFFSETS}
                                   for k in all_labels} for a in LADDER},
               "resid_norm": {a: {str(o): norms[a][o] for o in OFFSETS} for a in LADDER},
               "layer_profile_mean": {a: {k: np.mean(profile[a][k], 0).tolist()
                                          for k in LABELS} for a in LADDER},
               "layers": layers,
               "n_passages": len(passages)}, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
