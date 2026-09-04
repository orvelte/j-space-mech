"""E9 — is the signal crossing the boundary generic, or property-specific?

Same Spanish passage, four question prefixes (all exactly 13 tokens, so every
run is token-aligned):

    A       "predict the next word"                 -- no property instruction
    B_lang  "identify the language"                 -- the clean run we measure
    B_tense "identify the tense"                    -- a *different* property
    B_pos   "identify the part of speech ..."       -- another different property

We patch the layer-l boundary state of the B_lang run from each source and read
the Spanish coordinate at (l*=24, p*=22), using the same intervention as E7
(replace the mixer input at question positions, hold its output there).

Reading:
  patch-from-B_tense kills as much as patch-from-A  -> the carried signal is
      **property-specific**: "report the language" is not interchangeable with
      "report the tense".
  patch-from-B_tense kills ~nothing                 -> the carried signal is
      **generic**: any "identify a property" instruction carries the same thing
      and the specificity lives downstream.

Control that has to come first: the Spanish coordinate *unpatched* under each
question. If B_tense already elevates it as much as B_lang does, the two runs
carry the same thing anyway and the patch cannot separate the hypotheses.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jspace import hooks, lens_ops, model as jmodel, stimuli
from scripts.e7_boundary import MixerIO, BoundaryPatch

SOURCES = ["A", "tense", "pos", "language"]   # "language" = identity control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--max-layer", type=int, default=24)
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    l_star = jmodel.L_STAR["language"]
    layers = list(range(args.max_layer + 1))
    tgt = stimuli.target_ids(b, "language")["primary_id"]
    passages = json.load(open(stimuli.DATA))["language"]
    pre = stimuli.build_prefixes(b.tok)

    # Build the four token-aligned prompts per passage.
    items = []
    for text in passages:
        ids = {}
        for src, key in [("A", "A"), ("tense", "B:tense"),
                         ("pos", "B:pos"), ("language", "B:language")]:
            ids[src] = b.model.encode(f"{pre[key]}\n\n{text}")
        n = {k: v.shape[1] for k, v in ids.items()}
        if len(set(n.values())) != 1:
            continue                      # alignment is mandatory; drop otherwise
        start = int((ids["A"][0] != ids["language"][0]).nonzero().flatten()[-1]) + 1
        items.append({"ids": ids, "stim_start": start,
                      "p_star": min(start + stimuli.P_STAR_OFFSET, ids["A"].shape[1] - 1)})
    print(f"{len(items)}/{len(passages)} passages aligned across all four questions",
          flush=True)

    @torch.no_grad()
    def coord(ids, p_star, patch=None):
        ctx = patch if patch is not None else _n()
        with ctx:
            with hooks.NodeCache(b, layers=[l_star]) as c:
                b.model.forward(ids)
                return float(lens_ops.coordinate(b, c.resid[l_star][0, p_star], l_star, tgt))

    class _n:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    # ---- control: unpatched Spanish coordinate under each question -------
    base = {s: [] for s in SOURCES}
    for it in items:
        for s in SOURCES:
            base[s].append(coord(it["ids"][s], it["p_star"]))
    means = {s: float(np.mean(v)) for s, v in base.items()}
    gap = means["language"] - means["A"]
    print("\nunpatched Spanish coordinate at (l*=24, p*):")
    for s in SOURCES:
        share = (means[s] - means["A"]) / gap if gap else float("nan")
        print(f"  {s:<9} {means[s]:.4f}   ({share:>6.0%} of the language-vs-A gap)")
    print(f"  (B_language)-(A) gap = {gap:.4f}", flush=True)

    # ---- capture boundary inputs per source, outputs for the clean run ---
    print("\ncapturing boundary states...", flush=True)
    for it in items:
        with torch.no_grad():
            it["src_in"] = {}
            for s in SOURCES:
                with MixerIO(b, layers) as io:
                    b.model.forward(it["ids"][s])
                    it["src_in"][s] = {l: io.inp[l][:, :it["stim_start"]].clone()
                                       for l in layers}
            with MixerIO(b, layers) as io:
                b.model.forward(it["ids"]["language"])
                it["keep_out"] = {l: io.out[l][:, :it["stim_start"]].clone() for l in layers}

    # ---- patch, layer by layer, from each source -------------------------
    rows = []
    for l in layers:
        row = {"layer": l, "kind": "attn" if b.is_full_attn(l) else "gdn"}
        for s in SOURCES:
            ms = [coord(it["ids"]["language"], it["p_star"],
                        BoundaryPatch(b, l, list(range(it["stim_start"])),
                                      it["src_in"][s][l], it["keep_out"][l]))
                  for it in items]
            row[s] = float((means["language"] - np.mean(ms)) / gap) if gap else float("nan")
        rows.append(row)
        print(f"  L{l:<3} {row['kind']:<5} kills:  from A {row['A']:>7.1%} | "
              f"from tense {row['tense']:>7.1%} | from pos {row['pos']:>7.1%} | "
              f"identity {row['language']:>7.1%}", flush=True)

    top = sorted(rows, key=lambda r: -r["A"])[:6]
    print("\nat the layers that carry the most (ranked by the A patch):")
    print(f"  {'layer':<7}{'from A':>9}{'from tense':>12}{'from pos':>10}"
          f"{'identity':>10}{'tense/A':>9}")
    for r in top:
        ratio = r["tense"] / r["A"] if r["A"] else float("nan")
        print(f"  L{r['layer']:<6}{r['A']:>8.1%}{r['tense']:>12.1%}{r['pos']:>10.1%}"
              f"{r['language']:>10.1%}{ratio:>9.2f}")
    l9 = next(r for r in rows if r["layer"] == 9)
    verdict = {"l9_from_A": l9["A"], "l9_from_tense": l9["tense"],
               "l9_from_pos": l9["pos"], "l9_identity_control": l9["language"],
               "l9_tense_over_A": l9["tense"] / l9["A"] if l9["A"] else None}
    json.dump({"unpatched_coord": means, "gap": gap, "per_layer": rows,
               "verdict": verdict},
              open("results/e9_carrier_specificity.json", "w"), indent=1)
    print(f"\nL9: from-tense / from-A = {verdict['l9_tense_over_A']:.2f}"
          if verdict["l9_tense_over_A"] else "")
    print("wrote results/e9_carrier_specificity.json")


if __name__ == "__main__":
    main()
