"""Model + lens loading, architecture introspection, and hook points.

Everything here is parameterised by ``variant`` so the pipeline can be debugged
on Qwen3.5-4B and run on Qwen3.6-27B by changing one string (DECISIONS.md D2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import transformers

import jlens

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_COMMIT = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"

VARIANTS = {
    "4b": (
        "Qwen/Qwen3.5-4B",
        "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    ),
    "27b": (
        "Qwen/Qwen3.6-27B",
        "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    ),
}

#: Workspace band for the 27B, from Olivia's prior study (DECISIONS.md D3).
WORKSPACE_BAND = (21, 46)

#: Measurement layer per property, selected in E1. language and pos come from the
#: pre-registered rule restricted to the band (D8); tense was moved 21 -> 28 after
#: the fact, for the reasons in DECISIONS.md D10 -- that change is post-hoc and
#: the write-up must say so.
L_STAR = {"language": 24, "tense": 28, "pos": 30}


@dataclass
class Bundle:
    """A loaded model, its lens, and the architecture facts the experiments need."""

    variant: str
    hf: torch.nn.Module
    tok: object
    model: jlens.HFLensModel  # the LensModel wrapper
    lens: jlens.JacobianLens
    n_layers: int
    d_model: int
    n_heads: int
    head_dim: int
    layer_types: list[str]
    full_attn_layers: list[int] = field(default_factory=list)
    gdn_layers: list[int] = field(default_factory=list)

    @property
    def blocks(self):
        return self.model.layers

    def block(self, layer: int):
        return self.model.layers[layer]

    def is_full_attn(self, layer: int) -> bool:
        return self.layer_types[layer] == "full_attention"


def load(variant: str = "4b", *, attn_implementation: str = "eager") -> Bundle:
    """Load the model and its pre-fitted lens, asserting everything the spec's
    E0.1 requires about the lens file before returning."""
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    name, lens_file = VARIANTS[variant]

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation=attn_implementation
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(name)
    model = jlens.from_hf(hf, tok)

    # jlens should auto-detect Layout("model") on Qwen3_5ForCausalLM; assert it
    # rather than trust it, since a wrong layout silently reads the wrong stack.
    assert model.layout.path in ("model", "model.language_model"), model.layout
    assert len(model.layers) == model.n_layers

    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=lens_file, revision=LENS_REVISION
    )
    # E0.1 / K0: the same branch ships a non-n1000 file, and the qwen3-32b entry
    # in this repo was uploaded as an unfinalised checkpoint (HF discussion #3).
    assert lens.n_prompts == 1000, f"n_prompts={lens.n_prompts}, expected 1000"
    assert lens.d_model == model.d_model, (lens.d_model, model.d_model)
    for layer, J in lens.jacobians.items():
        assert J.shape == (model.d_model, model.d_model), (layer, J.shape)

    text_config = hf.config.get_text_config()
    layer_types = list(
        getattr(text_config, "layer_types", ["full_attention"] * model.n_layers)
    )
    return Bundle(
        variant=variant,
        hf=hf,
        tok=tok,
        model=model,
        lens=lens,
        n_layers=model.n_layers,
        d_model=model.d_model,
        n_heads=text_config.num_attention_heads,
        head_dim=getattr(
            text_config,
            "head_dim",
            model.d_model // text_config.num_attention_heads,
        ),
        layer_types=layer_types,
        full_attn_layers=[i for i, t in enumerate(layer_types) if t == "full_attention"],
        gdn_layers=[i for i, t in enumerate(layer_types) if t == "linear_attention"],
    )
