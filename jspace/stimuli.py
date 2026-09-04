"""Paired-question stimuli (spec §3.2/§3.3), raw text, exactly length-matched.

Prompts are **raw text**, not chat-templated (DECISIONS.md D1): the public lens
is fit on raw wikitext, and the paper's §3.5 protocol is a plain passage.

The A/B question prefixes are padded to an **identical** token count, not the
spec's ±2 (DECISIONS.md D4): E2 patches activations position-by-position between
the clean (B) and corrupted (A) runs, which requires exact alignment. Every
built pair asserts equal length *and* elementwise-equal stimulus token ids.

Question wordings follow spec §3.2. Label and foil sets are adapted from
``anthropics/jacobian-lens`` ``data/experiments/`` (Apache-2.0) so hit rates are
comparable to the paper's.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data", "passages.json")

#: All four wordings are hand-tuned to exactly 13 tokens with parallel structure
#: ("Read the following passage and {predict|identify} the ..."), so no filler
#: padding is needed and the A/B prompts differ only in the question itself.
QUESTION_A = "Read the following passage and predict the next word that it requires."
QUESTION_B = {
    "language": "Read the following passage and identify the language it is written in.",
    "tense": "Read the following passage and identify the tense it is written in.",
    "pos": "Read the following passage and identify the part of speech required next.",
}

#: Primary single-token target per property (spec §3.2), plus the paper's
#: synonym set (secondary "any-of" hit criterion) and a contrastive foil set.
TARGETS = {
    "language": {
        "token": " Spanish",
        "expected": ["Spanish", "Spain", "español"],
        "foil": ["English", "French", "Italian"],
    },
    "tense": {
        "token": " past",
        "expected": ["past"],
        "foil": ["present", "future"],
    },
    "pos": {
        "token": " adjective",
        "expected": ["adjective"],
        "foil": ["noun", "verb"],
    },
}

#: Neutral one-token adverbs used only to equalise prefix token counts.
_FILLERS = [" carefully", " closely", " attentively", " thoroughly", " again", " properly"]

#: Spec §3.3: p* is the stimulus token at index 10 from the stimulus start.
P_STAR_OFFSET = 10


@dataclass
class Pair:
    """One stimulus under both conditions, with aligned positions."""

    prop: str
    index: int
    passage: str
    prompt_a: str
    prompt_b: str
    ids_a: torch.Tensor  # [1, T]
    ids_b: torch.Tensor  # [1, T]
    stimulus_start: int  # first stimulus token index in the full prompt
    p_star: int

    @property
    def n_tokens(self) -> int:
        return self.ids_a.shape[1]

    @property
    def stimulus_positions(self) -> list[int]:
        return list(range(self.stimulus_start, self.n_tokens))


def _n_tokens(tok, text: str) -> int:
    return len(tok(text).input_ids)


def _pad_prefix(tok, prefix: str, target: int) -> str:
    """Insert neutral adverbs after "passage" until ``prefix`` is exactly
    ``target`` tokens long."""
    out = prefix
    used = 0
    while _n_tokens(tok, out) < target and used < len(_FILLERS):
        deficit = target - _n_tokens(tok, out)
        best = None
        for f in _FILLERS[used:]:
            cand = out.replace("passage", "passage" + f, 1)
            cost = _n_tokens(tok, cand) - _n_tokens(tok, out)
            if 0 < cost <= deficit and (best is None or cost > best[0]):
                best = (cost, cand)
        if best is None:
            break
        out = best[1]
        used += 1
    if _n_tokens(tok, out) != target:
        raise ValueError(
            f"could not pad prefix to {target} tokens (got {_n_tokens(tok, out)}): {out!r}"
        )
    return out


def build_prefixes(tok) -> dict[str, str]:
    """Return ``{"A": ..., "B:language": ..., ...}``, all the same token length."""
    raw = {"A": QUESTION_A} | {f"B:{k}": v for k, v in QUESTION_B.items()}
    target = max(_n_tokens(tok, v) for v in raw.values())
    return {k: _pad_prefix(tok, v, target) for k, v in raw.items()}


def e02_dropped(prop: str) -> set[int]:
    """Stimulus indices that failed the E0.2 behavioral control, as persisted by
    ``scripts/e0_difference.py --drop-failing-e02``.

    Spec E0.2 says to drop these. They must be excluded from *every* later
    experiment, not just the one that discovered them, or E2 measures a different
    stimulus set than E0.4 selected l* on.
    """
    path = os.path.join(os.path.dirname(__file__), os.pardir, "results", "e02_dropped.json")
    if not os.path.exists(path):
        return set()
    return set(json.load(open(path)).get(prop, []))


def load(bundle, prop: str, *, max_stimuli: int | None = None,
         apply_e02_filter: bool = True) -> tuple[list[Pair], dict]:
    """Build the aligned A/B pairs for one property.

    Returns ``(pairs, report)``; ``report`` counts stimuli dropped by the
    alignment assertions, which the E0 write-up must state.
    """
    tok = bundle.tok
    passages = json.load(open(DATA))[prop]
    excluded = e02_dropped(prop) if apply_e02_filter else set()
    if max_stimuli:
        passages = passages[:max_stimuli]
    prefixes = build_prefixes(tok)
    prefix_a, prefix_b = prefixes["A"], prefixes[f"B:{prop}"]
    n_prefix = _n_tokens(tok, prefix_a)

    pairs, dropped = [], []
    for i, passage in enumerate(passages):
        if i in excluded:
            dropped.append((i, "failed E0.2 behavioral control"))
            continue
        prompt_a = f"{prefix_a}\n\n{passage}"
        prompt_b = f"{prefix_b}\n\n{passage}"
        ids_a = bundle.model.encode(prompt_a)
        ids_b = bundle.model.encode(prompt_b)

        if ids_a.shape != ids_b.shape:
            dropped.append((i, f"length {ids_a.shape[1]} vs {ids_b.shape[1]}"))
            continue
        # Locate the stimulus by finding where the two prompts stop differing.
        diff = (ids_a[0] != ids_b[0]).nonzero().flatten()
        stimulus_start = int(diff[-1].item()) + 1 if len(diff) else n_prefix
        if not torch.equal(ids_a[0, stimulus_start:], ids_b[0, stimulus_start:]):
            dropped.append((i, "stimulus tokens differ across conditions"))
            continue
        p_star = min(stimulus_start + P_STAR_OFFSET, ids_a.shape[1] - 1)
        if p_star < 16:  # jlens.fitting.SKIP_FIRST_N_POSITIONS
            dropped.append((i, f"p*={p_star} < 16, lens unfitted there"))
            continue
        pairs.append(
            Pair(prop, i, passage, prompt_a, prompt_b, ids_a, ids_b, stimulus_start, p_star)
        )

    report = {
        "property": prop,
        "n_kept": len(pairs),
        "n_excluded_by_e02": len(excluded),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "prefix_tokens": n_prefix,
        "prefix_a": prefix_a,
        "prefix_b": prefix_b,
        "p_star_range": [min(p.p_star for p in pairs), max(p.p_star for p in pairs)] if pairs else None,
        "n_tokens_range": [min(p.n_tokens for p in pairs), max(p.n_tokens for p in pairs)] if pairs else None,
    }
    return pairs, report


def target_ids(bundle, prop: str) -> dict:
    """Resolve the property's label tokens to ids, keeping only single-token ones."""
    tok = bundle.tok
    spec = TARGETS[prop]
    primary = tok.encode(spec["token"])
    out = {"token": spec["token"], "primary_id": primary[0], "primary_is_single": len(primary) == 1}
    for key in ("expected", "foil"):
        ids = {}
        for w in spec[key]:
            for form in (w, " " + w, w.lower(), " " + w.lower(), w.capitalize(), " " + w.capitalize()):
                enc = tok.encode(form)
                if len(enc) == 1:
                    ids[form] = enc[0]
        out[key] = ids
    return out
