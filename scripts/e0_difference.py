"""E0.2 / E0.3 / E0.4 — does the workspace difference exist? (spec §4; gate K1)

E0.2  behavioral: under (B) the model names the property; under (A) it continues.
E0.3  **the gate**: the property token's J-lens rank is in the top-25 at >=2x as
      many (layer, position) cells under (B) as under (A), with per-passage
      bootstrap CIs.
E0.4  the (B)-(A) gap measured by the J-lens must exceed the logit-lens gap by
      >=1.5x at the candidate layer, else the effect is output preparation.

Saves the full layer x position arrays so E1 can pick l* from the same data.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jspace import hooks, lens_ops, model as jmodel, stimuli


def bootstrap_ci(values, n=10000, seed=0):
    """Per-passage bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return [float("nan")] * 2
    draws = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


@torch.no_grad()
def measure(b, pair, condition, target_id, layers):
    """One forward pass; returns ranks over stimulus positions and the p* curves."""
    ids = pair.ids_a if condition == "A" else pair.ids_b
    with hooks.NodeCache(b, layers=layers) as cache:
        b.model.forward(ids)
        resid = {l: cache.resid[l][0] for l in layers}  # [T, d]

    pos = pair.stimulus_positions
    ranks = np.empty((len(layers), len(pos)), dtype=np.int32)
    coord_j = np.empty(len(layers), dtype=np.float32)
    coord_ll = np.empty(len(layers), dtype=np.float32)
    ll_dir = lens_ops.logit_direction(b, target_id)

    for i, l in enumerate(layers):
        h = resid[l][pos]  # [P, d]
        scores = lens_ops.coordinates(b, h, l)  # [P, vocab], unit-normalised
        ranks[i] = lens_ops.rank_of(scores, target_id).cpu().numpy()
        h_star = resid[l][pair.p_star].float()
        coord_j[i] = float(h_star @ lens_ops.jhat(b, target_id, l))
        coord_ll[i] = float(h_star @ ll_dir)
    return ranks, coord_j, coord_ll


#: Qwen3.x opens a <think> block even from raw text. Prefilling an empty, closed
#: block is the model's own non-thinking convention and keeps the probe to a few
#: tokens instead of a few hundred.
NO_THINK_CUE = "\n\nAnswer:<think>\n\n</think>\n\n"


@torch.no_grad()
def generate(b, prompt, n=40, strip_think=True):
    """Greedy continuation. Qwen3.x opens a ``<think>`` block even from raw text,
    so give it room and drop everything up to ``</think>`` before scoring."""
    ids = b.model.encode(prompt)
    out = b.hf.generate(ids, max_new_tokens=n, do_sample=False,
                        pad_token_id=b.tok.pad_token_id or b.tok.eos_token_id)
    text = b.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    if strip_think and "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--props", default="language,tense,pos")
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--max-stimuli", type=int, default=None)
    ap.add_argument("--drop-failing-e02", action="store_true")
    ap.add_argument("--out", default="results/e0_difference.json")
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    layers = list(b.lens.source_layers)
    summary = {"variant": args.variant, "layers": layers, "top_k": args.top_k,
               "band": list(jmodel.WORKSPACE_BAND), "properties": {}}

    for prop in args.props.split(","):
        pairs, rep = stimuli.load(b, prop, max_stimuli=args.max_stimuli)
        tgt = stimuli.target_ids(b, prop)
        target_id = tgt["primary_id"]
        print(f"\n=== {prop} === {rep['n_kept']} stimuli, target {tgt['token']!r} "
              f"(id {target_id}, single={tgt['primary_is_single']})", flush=True)

        # ---- E0.2 behavioral -------------------------------------------------
        named, continued = [], []
        for p in pairs:
            ans = generate(b, p.prompt_b + NO_THINK_CUE, n=16)
            hit = any(w.lower().strip() in ans.lower() for w in
                      [k.strip() for k in tgt["expected"]])
            named.append({"i": p.index, "answer": ans.strip()[:60], "hit": bool(hit)})
            continued.append({"i": p.index, "continuation": generate(b, p.prompt_a, n=24).strip()[:60]})
        acc = float(np.mean([x["hit"] for x in named]))
        print(f"E0.2 naming accuracy {acc:.2%}  (>=90% required)", flush=True)
        if args.drop_failing_e02:
            keep = {x["i"] for x in named if x["hit"]}
            dropped_e02 = [p.index for p in pairs if p.index not in keep]
            pairs = [p for p in pairs if p.index in keep]
            print(f"E0.2 dropped {len(dropped_e02)} stimuli, {len(pairs)} remain", flush=True)
        else:
            dropped_e02 = []

        # ---- E0.3 / E0.4 internals ------------------------------------------
        cells_a, cells_b, curves = [], [], {"A": [], "B": [], "A_ll": [], "B_ll": []}
        by_layer_a, by_layer_b = [], []
        for j, p in enumerate(pairs):
            ra, ja, la = measure(b, p, "A", target_id, layers)
            rb, jb, lb = measure(b, p, "B", target_id, layers)
            cells_a.append(int((ra < args.top_k).sum()))
            cells_b.append(int((rb < args.top_k).sum()))
            by_layer_a.append((ra < args.top_k).sum(axis=1))
            by_layer_b.append((rb < args.top_k).sum(axis=1))
            curves["A"].append(ja); curves["B"].append(jb)
            curves["A_ll"].append(la); curves["B_ll"].append(lb)
            if j == 0:
                np.savez(f"results/e0_ranks_{prop}_example.npz", rank_a=ra, rank_b=rb,
                         layers=np.array(layers), positions=np.array(p.stimulus_positions))
            print(f"  [{j+1}/{len(pairs)}] top-{args.top_k} cells  A={cells_a[-1]:>4}  B={cells_b[-1]:>4}",
                  flush=True)

        curves = {k: np.stack(v) for k, v in curves.items()}  # [n_passages, n_layers]
        np.savez(f"results/e0_curves_{prop}.npz", layers=np.array(layers),
                 cells_a=np.array(cells_a), cells_b=np.array(cells_b),
                 cells_by_layer_a=np.stack(by_layer_a),
                 cells_by_layer_b=np.stack(by_layer_b), **curves)

        ratio = (sum(cells_b) + 1e-9) / (sum(cells_a) + 1e-9)
        per_passage_ratio = [(bb + 1) / (aa + 1) for aa, bb in zip(cells_a, cells_b)]
        gap_j = curves["B"] - curves["A"]      # [n_passages, n_layers]
        gap_ll = curves["B_ll"] - curves["A_ll"]
        se_j = gap_j.std(axis=0, ddof=1) / np.sqrt(gap_j.shape[0])
        mean_j = gap_j.mean(axis=0)

        # spec §3.3: first layer where the gap exceeds 2 SE for >=3 consecutive layers
        sig = mean_j > 2 * se_j

        def first_run(lo=None, hi=None):
            """First layer starting a >=3-layer run of >2 SE, optionally within
            [lo, hi]. The unrestricted value is the spec's rule; the in-band one
            is the pre-registered fallback K2 needs (DECISIONS.md D3)."""
            for i in range(len(layers) - 2):
                if lo is not None and not (lo <= layers[i] <= hi):
                    continue
                if sig[i] and sig[i + 1] and sig[i + 2]:
                    return layers[i]
            return None

        l_star = first_run()
        l_star_in_band = first_run(*jmodel.WORKSPACE_BAND)

        entry = {
            "n_stimuli": rep["n_kept"], "alignment_report": rep,
            "target": {k: v for k, v in tgt.items() if k != "expected" or True},
            "E0.2": {"naming_accuracy": acc, "pass_90pct": acc >= 0.90,
                     "dropped_stimuli": dropped_e02, "n_after_drop": len(pairs),
                     "examples_named": named[:5], "examples_continued": continued[:5]},
            "E0.3": {"cells_A_total": sum(cells_a), "cells_B_total": sum(cells_b),
                     "ratio": ratio, "pass_2x": ratio >= 2.0,
                     "per_passage_ratio_mean": float(np.mean(per_passage_ratio)),
                     "per_passage_ratio_ci95": bootstrap_ci(per_passage_ratio),
                     "cells_A_ci95": bootstrap_ci(cells_a),
                     "cells_B_ci95": bootstrap_ci(cells_b)},
            "E1": {"l_star": l_star, "l_star_in_band": l_star_in_band,
                   "in_band": (l_star is not None
                               and jmodel.WORKSPACE_BAND[0] <= l_star <= jmodel.WORKSPACE_BAND[1]),
                   "mean_gap_by_layer": mean_j.tolist(), "se_gap_by_layer": se_j.tolist()},
        }
        if l_star is not None:
            i = layers.index(l_star)
            gj, gl = float(mean_j[i]), float(gap_ll.mean(axis=0)[i])
            # The J-lens gap must beat the logit-lens gap by >=1.5x. A logit-lens
            # gap at or below zero means the logit lens shows no effect at all, so
            # any positive J-lens gap passes; a ratio is undefined there.
            ratio_e04 = gj / gl if gl > 0 else float("inf")
            entry["E0.4"] = {"l_star": l_star, "jlens_gap": gj, "logit_lens_gap": gl,
                             "ratio": ratio_e04, "pass_1.5x": bool(gj > 0 and ratio_e04 >= 1.5)}
        summary["properties"][prop] = entry
        print(f"E0.3 ratio B/A = {ratio:.2f} (>=2 required) | l* = {l_star} "
              f"| in band {jmodel.WORKSPACE_BAND}: {entry['E1']['in_band']}", flush=True)
        if l_star is not None:
            print(f"E0.4 J gap {entry['E0.4']['jlens_gap']:.3f} vs LL gap "
                  f"{entry['E0.4']['logit_lens_gap']:.3f} -> ratio "
                  f"{entry['E0.4']['ratio']:.2f} (>=1.5 required)", flush=True)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
