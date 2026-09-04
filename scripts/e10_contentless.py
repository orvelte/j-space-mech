"""E10 — does a question with *no linguistic content* also admit Spanish?

The test that separates the two readings E9 could not (DECISIONS.md D15).

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

SOURCES = ["A", "tense", "pos", "wordcount", "firstletter", "linecount", "linewidth", "language"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--layers", default="8,9,10,12,15,24")
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    l_star = jmodel.L_STAR["language"]
    layers = [int(x) for x in args.layers.split(",")]
    tgt = stimuli.target_ids(b, "language")["primary_id"]
    passages = json.load(open(stimuli.DATA))["language"]
    pre = stimuli.build_prefixes(b.tok)
    # Four questions asking for a property that needs no linguistic analysis,
    # each exactly 13 tokens so they align with the existing prefixes.
    pre["Q:wordcount"] = "Read the following passage and identify how many words are in it."
    pre["Q:firstletter"] = "Read the following passage and identify the letter that it begins with."
    pre["Q:linecount"] = "Read the following passage and identify the number of lines it has."
    pre["Q:linewidth"] = "Read the following passage and identify the width of its longest line."
    n_pre = {k: len(b.tok(v).input_ids) for k, v in pre.items()}
    assert len(set(n_pre.values())) == 1, n_pre

    # Build the four token-aligned prompts per passage.
    items = []
    for text in passages:
        ids = {}
        for src, key in [("A", "A"), ("tense", "B:tense"), ("pos", "B:pos"),
                         ("wordcount", "Q:wordcount"), ("firstletter", "Q:firstletter"),
                         ("linecount", "Q:linecount"), ("linewidth", "Q:linewidth"),
                         ("language", "B:language")]:
            ids[src] = b.model.encode(f"{pre[key]}\n\n{text}")
        n = {k: v.shape[1] for k, v in ids.items()}
        if len(set(n.values())) != 1:
            continue                      # alignment is mandatory; drop otherwise
        start = int((ids["A"][0] != ids["language"][0]).nonzero().flatten()[-1]) + 1
        items.append({"ids": ids, "stim_start": start,
                      "p_star": min(start + stimuli.P_STAR_OFFSET, ids["A"].shape[1] - 1)})
    print(f"{len(items)}/{len(passages)} passages aligned across all {len(SOURCES)} questions",
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
        print(f"  L{l:<3} {row['kind']:<5} kills: " +
              "  ".join(f"{s_[:5]} {row[s_]:>6.1%}" for s_ in SOURCES), flush=True)

    top = sorted(rows, key=lambda r: -r["A"])[:6]
    print("\nboundary patch at the carrier layers (ranked by the A patch):")
    print("  layer  " + "".join(f"{s_[:9]:>11}" for s_ in SOURCES))
    for r in top:
        print(f"  L{r['layer']:<6}" + "".join(f"{r[s_]:>10.1%} " for s_ in SOURCES))

    l9 = next(r for r in rows if r["layer"] == 9)
    contentless = ["wordcount", "firstletter", "linecount", "linewidth"]
    shares = {s_: (means[s_] - means["A"]) / gap for s_ in contentless}
    mean_share = float(np.mean(list(shares.values())))
    verdict = {"contentless_shares": shares, "mean_share": mean_share,
               "generic_established_ge_0.50": bool(mean_share >= 0.50),
               "generic_retracted_lt_0.20": bool(mean_share < 0.20),
               "graded": bool(0.20 <= mean_share < 0.50),
               "l9_patch": {s_: l9[s_] for s_ in SOURCES}}
    print(f"\n  D15: mean contentless share = {mean_share:.0%} -> "
          f"{'GENERIC ESTABLISHED' if mean_share >= .5 else 'RETRACT generic' if mean_share < .2 else 'GRADED'}")
    json.dump({"unpatched_coord": means, "gap": gap, "per_layer": rows,
               "verdict": verdict},
              open("results/e10_contentless.json", "w"), indent=1)
    print("wrote results/e10_contentless.json")


if __name__ == "__main__":
    main()
