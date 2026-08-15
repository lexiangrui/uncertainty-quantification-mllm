"""Semantic Anchor Intervention (SAI) — object-grounded latent intervention UQ.

Pipeline per sample (all on the model's own greedy response):

1. **Object mention resolution** — visual objects mentioned in the response
   (extracted upstream by an independent text model) are mapped to (a) their
   generated-token mention positions and (b) a vocabulary anchor token id.
2. **Vision logit lens** — visual token hidden states at a chosen decoder
   layer are pushed through (final norm → lm_head); the anchor token's
   log-probability per visual token *locates* the object's visual evidence
   (top-k visual positions) and yields a static *reading strength*.
3. **Semantic anchor intervention** — the located visual states are rotated
   along the anchor's semantic direction in the pre-norm residual space:

       h' = normalize(h + sign·σ·‖h‖·u),   u = normalize((W_t − W̄) ⊙ g)

   where W_t is the anchor's unembedding row, W̄ the vocabulary mean and g
   the final RMSNorm gain (so u is the pre-norm direction that raises the
   anchor's logit).  ``sign=+1`` pushes visual evidence *toward* the object
   semantics, ``sign=−1`` away.
4. **Response measurement** — teacher-forced log-probabilities at every
   object's mention positions (plus per-section NLL) are recorded for the
   baseline and each intervention; the full response matrix is persisted so
   scores are frozen post-processing of the extraction (LSB protocol).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from src.improvement.backend import SaiBackend, TeacherForcingBatch

SECTIONS = ("vision", "reasoning", "answer")


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from arbitrary parts (sample id, tag...)."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def unit_direction(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Random unit vector in hidden space, sampled deterministically on CPU."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    u = torch.randn(dim, generator=gen, dtype=torch.float32)
    return (u / u.norm().clamp_min(1e-12)).to(device)


def renorm_rotation(h: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Add a displacement and project back to the original norms (fp32 math)."""
    norms = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    out = h + displacement
    out = out * (norms / out.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    return out


# ----------------------------------------------------------------------
# Object → vocabulary anchor
# ----------------------------------------------------------------------

def anchor_token_id(backend: SaiBackend, surface: str) -> int | None:
    """First meaningful subword token id of ``surface`` as it appears mid-text.

    Mid-text mentions are usually preceded by a space, so the leading-space
    encoding is preferred.  Some tokenizers (LLaMA SentencePiece) split the
    leading space into a standalone whitespace token (e.g. 29871 '▁') —
    such pure-whitespace tokens are skipped so the anchor is the object's
    actual first subword.
    """
    tok = backend.tokenizer
    for form in (f" {surface}", surface):
        ids = tok.encode(form, add_special_tokens=False)
        ids = [i for i in ids if i not in getattr(tok, "all_special_ids", [])]
        ids = [
            i
            for i in ids
            if tok.decode([i], clean_up_tokenization_spaces=False).strip()
        ]
        if ids:
            return int(ids[0])
    return None


# ----------------------------------------------------------------------
# Vision logit lens
# ----------------------------------------------------------------------

class LensKit:
    """Caches the (norm, lm_head, vocabulary mean) needed for lens reads."""

    def __init__(self, backend: SaiBackend):
        self.backend = backend
        self._mean_row: torch.Tensor | None = None
        self._norm_gain: torch.Tensor | None = None

    @property
    def W(self) -> torch.Tensor:
        return self.backend.lm_head_weight()

    @property
    def mean_row(self) -> torch.Tensor:
        if self._mean_row is None:
            self._mean_row = self.W.float().mean(dim=0)
        return self._mean_row

    @property
    def norm_gain(self) -> torch.Tensor:
        """Final RMSNorm gain g; ones if the norm has no weight."""
        if self._norm_gain is None:
            norm = self.backend.final_norm()
            w = getattr(norm, "weight", None)
            if w is None:
                self._norm_gain = torch.ones(
                    self.W.shape[1], dtype=torch.float32, device=self.W.device
                )
            else:
                self._norm_gain = w.float()
        return self._norm_gain

    def visual_logits(
        self, hidden_L: torch.Tensor, vis_pos: torch.Tensor
    ) -> torch.Tensor:
        """Vocab logits of the visual states at one layer (post final norm)."""
        norm = self.backend.final_norm()
        h = hidden_L.index_select(1, vis_pos)
        with torch.autocast(device_type="cuda", enabled=False):
            h = norm(h.float())
            return h @ self.W.float().t()

    def object_logprobs(self, logits: torch.Tensor, token_id: int) -> torch.Tensor:
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp[:, :, token_id]

    def anchor_direction(
        self, token_id: int, *, center: bool = True, device=None
    ) -> torch.Tensor:
        """Unit direction in the *pre-norm* residual space that raises the
        anchor token's logit: u = normalize((W_t − W̄) ⊙ g)."""
        row = self.W[token_id].float()
        if center:
            row = row - self.mean_row
        u = row * self.norm_gain.to(row.device)
        u = u / u.norm().clamp_min(1e-12)
        return u.to(device or row.device)


# ----------------------------------------------------------------------
# Layer-entry intervention hook
# ----------------------------------------------------------------------

class InterventionHook:
    """Forward pre-hook rewriting the residual stream entering one layer.

    Registered on ``decoder_layers[probe_layer]``; arming rotates the states
    at ``positions`` along ``direction`` by radius σ (norm-preserving).
    ``direction`` may be a shared (D,) vector or a per-position (P, D)
    matrix (e.g. each token pointing toward an anchor point).
    transformers ≥ 5 passes ``hidden_states`` as a bare tensor, older
    versions as a tuple — both are handled.  Inactive while disarmed so the
    baseline forward passes through untouched.
    """

    def __init__(self):
        self.direction: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.sigma: float = 0.0
        self.weights: torch.Tensor | None = None

    def arm(self, positions: torch.Tensor, direction: torch.Tensor, sigma: float,
            weights: torch.Tensor | None = None):
        self.positions = positions
        self.direction = direction
        self.sigma = float(sigma)
        self.weights = weights

    def disarm(self):
        self.direction = None
        self.positions = None
        self.sigma = 0.0
        self.weights = None

    def __call__(self, module, args, kwargs):
        if self.direction is None or self.positions is None or self.sigma <= 0.0:
            return None
        hidden_states = kwargs.get("hidden_states") if kwargs else None
        in_kwargs = hidden_states is not None
        if hidden_states is None and args:
            hidden_states = args[0]
        if isinstance(hidden_states, tuple):
            h = hidden_states[0] if hidden_states else None
        elif isinstance(hidden_states, torch.Tensor):
            h = hidden_states
        else:
            return None
        pos = self.positions.to(h.device)
        hp = h.index_select(1, pos).float()
        direction = self.direction.to(hp.device).float()
        scale = self.sigma * hp.norm(dim=-1, keepdim=True)
        if self.weights is not None:
            scale = scale * self.weights.to(hp.device).float().view(1, -1, 1)
        if direction.dim() == 1:
            disp = scale * direction.view(1, 1, -1)
        else:
            u = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            disp = scale * u
        out_h = renorm_rotation(hp, disp).to(h.dtype)
        out = h.clone()
        out.index_copy_(1, pos, out_h)
        # transformers ≥ 5 requires (new_args, new_kwargs) from pre-hooks.
        if in_kwargs:
            kwargs["hidden_states"] = out
            return args, kwargs
        return (out,) + tuple(args[1:]), kwargs


# ----------------------------------------------------------------------
# Teacher-forced response measurement
# ----------------------------------------------------------------------

@dataclass
class SectionSpans:
    """Character spans of the three XML sections inside raw_response."""
    vision: tuple[int, int]
    reasoning: tuple[int, int]
    answer: tuple[int, int]


def section_spans(raw_response: str) -> SectionSpans | None:
    import re

    from src.generation.parser import XML_SECTION_PATTERN

    match = XML_SECTION_PATTERN.fullmatch(raw_response)
    if match is None:
        return None
    spans = {}
    for name in SECTIONS:
        s, e = match.span(name)
        while s < e and raw_response[s].isspace():
            s += 1
        while e > s and raw_response[e - 1].isspace():
            e -= 1
        spans[name] = (s, e)
    return SectionSpans(**spans)


@dataclass
class ForwardMeasurement:
    """Teacher-forced measurements of one (baseline or intervened) forward."""
    token_logprobs: dict[int, float]     # gen index → log p(token)
    token_logits: dict[int, float]       # gen index → raw logit of the token
    section_nll: dict[str, float]        # section → mean NLL per token
    section_flip: dict[str, int]         # section → #positions w/ argmax change (vs baseline argmax)
    argmax_ids: list[int] | None = None  # argmax token id per predict position (gen-aligned)


def run_forward(
    backend: SaiBackend,
    batch: TeacherForcingBatch,
    *,
    hook: InterventionHook | None = None,
    layer_module=None,
    positions: torch.Tensor | None = None,
    direction: torch.Tensor | None = None,
    sigma: float = 0.0,
    weights: torch.Tensor | None = None,
    want_argmax: bool = False,
    want_hidden: bool = False,
) -> tuple[ForwardMeasurement, torch.Tensor | None]:
    """Run one teacher-forced forward and measure the response.

    The hook is armed on ``layer_module`` only when ``positions`` is given
    (intervened forward); otherwise the forward passes through untouched
    (baseline).  Returns (measurement, hidden_states) — hidden states only
    when ``want_hidden`` (baseline forward used for lens reads).
    """
    kwargs = dict(batch.inputs)
    kwargs["use_cache"] = False
    kwargs["output_hidden_states"] = want_hidden
    with torch.inference_mode(), _HookContext(
        hook, layer_module, positions, direction, sigma, weights
    ):
        outputs = backend.model(**kwargs)
    logits = outputs.logits  # (1, seq, vocab)
    hidden = outputs.hidden_states if want_hidden else None
    del outputs

    gen_len = len(batch.gen_ids)
    # logits[p-1] predicts the token at absolute position p
    pred_logits = logits[
        0, batch.prompt_length - 1 : batch.prompt_length + gen_len - 1, :
    ].float()
    del logits
    targets = torch.tensor(batch.gen_ids, device=pred_logits.device)
    logprobs = torch.log_softmax(pred_logits, dim=-1)
    token_logprobs = {
        i: float(logprobs[i, t]) for i, t in enumerate(targets.tolist())
    }
    token_logits = {
        i: float(pred_logits[i, t]) for i, t in enumerate(targets.tolist())
    }
    argmax_ids = None
    if want_argmax:
        argmax_ids = pred_logits.argmax(dim=-1).tolist()
    measurement = ForwardMeasurement(
        token_logprobs=token_logprobs,
        token_logits=token_logits,
        section_nll={},
        section_flip={},
        argmax_ids=argmax_ids,
    )
    return measurement, hidden


class _HookContext:
    """Context manager arming a hook on one layer for the enclosed forward."""

    def __init__(self, hook, layer_module, positions=None, direction=None, sigma=0.0,
                 weights=None):
        self.hook = hook
        self.layer_module = layer_module
        self._positions = positions
        self._direction = direction
        self._sigma = sigma
        self._weights = weights
        self._handle = None

    def __enter__(self):
        if self.hook is None or self.layer_module is None:
            return self
        if self._positions is not None:
            self.hook.arm(self._positions, self._direction, self._sigma, self._weights)
        self._handle = self.layer_module.register_forward_pre_hook(
            self.hook, with_kwargs=True
        )
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self.hook is not None:
            self.hook.disarm()
        return False


def section_stats(
    measurement: ForwardMeasurement,
    batch: TeacherForcingBatch,
    spans: SectionSpans,
    gen_positions_by_section: dict[str, list[int]],
    baseline_argmax: list[int] | None,
) -> None:
    """Fill section_nll / section_flip on ``measurement`` in place."""
    for name in SECTIONS:
        positions = gen_positions_by_section.get(name, [])
        if not positions:
            measurement.section_nll[name] = float("nan")
            measurement.section_flip[name] = 0
            continue
        nll = -sum(measurement.token_logprobs[p] for p in positions) / len(positions)
        measurement.section_nll[name] = float(nll)
        if baseline_argmax is not None and measurement.argmax_ids is not None:
            flips = sum(
                1
                for p in positions
                if measurement.argmax_ids[p] != baseline_argmax[p]
            )
            measurement.section_flip[name] = int(flips)
        else:
            measurement.section_flip[name] = 0
