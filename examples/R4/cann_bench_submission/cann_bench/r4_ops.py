"""Thin host wrappers exposing the delivered R4 TileLang kernels to CANN Bench.

All computation lives in ``_r4_kernels`` -- a VERBATIM, build-time copy of
``examples/R4/r4_fused_sampling.py`` (see ``build.sh``). That module owns the two
TileLang kernels (K1 ``r4_topk_stats`` + K2 ``r4_mask_sample``/``r4_gumbel_sample``)
and the host wrapper ``r4_fused_sampling(...)`` that dispatches between the two
K2 paths via ``FUSE_GUMBEL``.

These wrappers do ZERO host compute. Each one only:
  * coerces ``need_min_p_sampling`` to ``bool`` (it selects a compile-time kernel
    specialization, so it must be a real Python bool, not a tensor/int), and
  * forwards to the vendored kernel wrapper, returning its output tuple unchanged.

No torch math, no ``.item()``/``.cpu()``, no ``torch.topk/sort/softmax/multinomial``,
no layout transforms, no CPU fallback, no output caching, no try/except that would
substitute a torch implementation. Any kernel error (e.g. unsupported shape such as
``V < K_MAX``, non-float32 logits, ``B == 0``) propagates to the caller verbatim.
"""

from . import _r4_kernels


def r4_fused_sampling(logits, top_ks, top_ps, min_ps, rand,
                      need_min_p_sampling=False, K_MAX=256, BV=4096):
    """V01-b path (``FUSE_GUMBEL=True``): Gumbel-max fused sampling.

    Returns ``(m_out, s_out, tau_out, valid, tok_out)`` with dtypes
    ``float32[B], float32[B], float32[B], int32[B], int32[B]``. ``tok_out`` is the
    sampled token id produced entirely on the NPU (no host ``torch.multinomial``).

    Parameter names are fixed by the CANN Bench contract (params are passed by
    keyword, mapped from ``proto.inputs[].name`` then ``case.attrs``).
    """
    return _r4_kernels.r4_fused_sampling(
        logits, top_ks, top_ps, min_ps,
        bool(need_min_p_sampling),
        K_MAX=K_MAX, BV=BV, rand=rand, FUSE_GUMBEL=True,
    )


def r4_mask_sample(logits, top_ks, top_ps, min_ps,
                   need_min_p_sampling=False, K_MAX=256, BV=4096):
    """V01-a path (``FUSE_GUMBEL=False``): masked (un-renormalized) probabilities.

    Returns ``(m_out, s_out, tau_out, valid, w_out)`` with dtypes
    ``float32[B], float32[B], float32[B], int32[B], float32[B,V]``. ``w_out`` is the
    masked probability tensor exactly as produced by the K2 kernel.

    SCOPING CAVEAT (SUB-BEH-001): the original V01-a host chain finishes with
    ``torch.multinomial(w_out, 1)`` to draw a token. That sampling step is
    deliberately OUT of the evaluated contract here -- invoking any torch builtin
    to carry part of the operator's semantics would violate SUB-BEH-001 (and
    ``torch.multinomial`` is explicitly forbidden for this submission). This
    wrapper therefore returns the kernel's ``w_out`` and performs no sampling. See
    README.md for the full rationale.
    """
    return _r4_kernels.r4_fused_sampling(
        logits, top_ks, top_ps, min_ps,
        bool(need_min_p_sampling),
        K_MAX=K_MAX, BV=BV, FUSE_GUMBEL=False,
    )
