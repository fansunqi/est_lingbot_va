# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from typing import Callable, ClassVar
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
    or_masks
)
from functools import partial

__all__ = ['WanTransformer3DModel']


def _load_flash_attn_func():
    """Resolve flash_attn_func lazily. Training/inference paths using
    attn_mode='flex' or 'torch' don't need this import, and FA4-only envs
    (where `flash_attn` is just the namespace package containing
    `flash_attn.cute`) can't satisfy it. Deferring the import to the call
    site keeps such envs importable.
    """
    try:
        from flash_attn_interface import flash_attn_func
        return flash_attn_func
    except ImportError:
        pass
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func
    except ImportError as e:
        raise ImportError(
            "attn_mode='flashattn' requires flash_attn (FA2/FA3) or "
            "flash_attn_interface. Neither was found (FA4 / flash_attn.cute "
            "alone is not sufficient — use attn_mode='flex' instead)."
        ) from e


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2))
    return out.transpose(1, 2)

_TRITON_KERNEL_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_M1": 32,
    "BLOCK_N1": 64,
    "BLOCK_M2": 64,
    "BLOCK_N2": 32,
}


class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = None
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None
    backend: ClassVar[str] = 'triton'

    @classmethod
    def set_backend(cls, backend: str, compile_mode: str = 'default') -> str:
        """Pick the FlexAttention backend used during this run.

        backend='flash' compiles flex_attention with kernel_options
        BACKEND='FLASH' (FlashAttention-4 CuTeDSL kernels for Hopper/Blackwell).
        Falls back to 'triton' with a warning if cutlass-dsl is missing or the
        kernel fails its smoke test (torch 2.10 stable's flex codegen lacks
        the seqlen_info contract that FA4 expects, so a torch nightly is
        currently required for the FA4 path to actually run).

        compile_mode is forwarded to ``torch.compile(...)`` on the triton path
        only — FA4 has its own cute-dsl autotuner, ``max-autotune`` doesn't
        apply. Use 'max-autotune-no-cudagraphs' to autotune the inner attention
        triton kernels (cudagraphs disabled because flex_attention's per-shape
        recompile breaks them).
        """
        backend = backend.lower()
        if backend == 'flash':
            backend = cls._probe_flash_backend()
        if backend == 'flash':
            # FA4 NYI: score_mod/mask_mod cannot capture SymInt under
            # dynamic=True. We compile the inner flex with dynamic=False and
            # graph-break at FlexAttnFunc.forward (via @torch.compiler.disable)
            # so the outer torch.compile(model, dynamic=True) never sees the
            # flex_attention op symbolically. Inner cute-dsl compiles per
            # shape but caches — re-using a shape is ~1ms, first compile per
            # shape ~200-400ms.
            cls.flex_attn = torch.compile(
                partial(flex_attention,
                        kernel_options={"BACKEND": "FLASH"}),
                dynamic=False,
            )
        elif backend == 'triton':
            cls.flex_attn = torch.compile(
                flex_attention, dynamic=True, mode=compile_mode,
            )
        else:
            raise ValueError(f"unknown flex_attn_backend: {backend!r}")
        cls.backend = backend
        return backend

    @classmethod
    def _probe_flash_backend(cls) -> str:
        """Return 'flash' if FA4 actually runs end-to-end, else 'triton'."""
        import importlib.util
        import warnings
        if importlib.util.find_spec('cutlass') is None:
            warnings.warn(
                "flex_attn_backend='flash' requested but the 'cutlass' "
                "(nvidia-cutlass-dsl) package is missing — falling back to "
                "the Triton backend. Install with "
                "`uv pip install --prerelease=allow flash-attn-4`.",
                stacklevel=3,
            )
            return 'triton'
        try:
            probe = torch.compile(
                partial(flex_attention,
                        kernel_options={"BACKEND": "FLASH"}),
                dynamic=False,
            )
            B, H, S, D = 1, 2, 256, 128
            q = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            probe(q, k, v).sum().backward()
        except Exception as e:
            warnings.warn(
                f"FlashAttention-4 probe failed ({type(e).__name__}: "
                f"{str(e).splitlines()[0][:200]}); falling back to the Triton "
                "backend. FA4 typically requires a torch nightly with the "
                "seqlen_info-aware flex codegen.",
                stacklevel=3,
            )
            return 'triton'
        return 'flash'

    def __init__(
        self,
        is_cross=False,
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
        if FlexAttnFunc.flex_attn is None:
            FlexAttnFunc.set_backend('triton')

    @torch.compiler.disable
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        # @torch.compiler.disable: outer torch.compile(model, dynamic=True)
        # would otherwise inline through the nested flex_attn torch.compile
        # and pull the flex_attention op into the outer Inductor lowering
        # with SymInt-shaped mask_mod inputs — which the FA4 (BACKEND='FLASH')
        # backend rejects (NYI: dynamic scalar capture). Breaking the graph
        # here keeps the inner flex_attn compile self-contained.
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        if FlexAttnFunc.backend == 'flash':
            # FA4 ignores Triton tile sizes; BACKEND is baked into the partial.
            x_out = FlexAttnFunc.flex_attn(q_varlen, k_varlen, v_varlen, block_mask=block_mask)
        else:
            x_out = FlexAttnFunc.flex_attn(q_varlen, k_varlen, v_varlen, block_mask=block_mask,
                                           kernel_options=_TRITON_KERNEL_OPTIONS)

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.compiler.disable
    @torch.no_grad()
    def init_mask(
        latent_shape,
        action_shape,
        padded_length,
        chunk_size,
        window_size,
        patch_size,
        device,
        text_seq_len,
        seq_ids_per_frame,
        frame_ids_per_frame,
        n_episodes,
    ):
        """Build self- and cross-attention block masks for packed training.

        @torch.compiler.disable keeps the outer torch.compile(transformer)
        from tracing through this — the body mutates inductor config and
        nests its own torch.compile(create_block_mask), which the outer
        Dynamo cannot reliably handle.
        """
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape
        assert patch_size[0] == 1, \
            f"mask build assumes no temporal patchification, got patch_size[0]={patch_size[0]}"

        seq_pf = seq_ids_per_frame.to(device=device, dtype=torch.long)
        frame_pf = frame_ids_per_frame.to(device=device, dtype=torch.long)

        lat_spatial = (L_H // patch_size[1]) * (L_W // patch_size[2])
        act_spatial = A_H * A_W

        def _broadcast(per_frame, spatial):
            return per_frame[:, :, None].expand(-1, -1, spatial).flatten()

        latent_seq_id = _broadcast(seq_pf, lat_spatial)
        action_seq_id = _broadcast(seq_pf, act_spatial)
        latent_frame_id = _broadcast(frame_pf, lat_spatial)
        action_frame_id = _broadcast(frame_pf, act_spatial)

        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)
        frame_ids = torch.cat(
            [latent_frame_id // chunk_size * 2] * 2
            + [action_frame_id // chunk_size * 2 + 1] * 2
        )
        noise_ids = torch.cat([
            torch.zeros_like(latent_frame_id),
            torch.ones_like(latent_frame_id),
            torch.zeros_like(action_frame_id),
            torch.ones_like(action_frame_id),
        ])

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        # window_size has to be a device tensor (not a Python int): under
        # whole-model torch.compile(dynamic=True) the int input becomes a
        # SymInt, which the FA4 (BACKEND='FLASH') CuteDSL template cannot
        # inline into mask_mod. Capturing a 0-dim tensor instead works for
        # both the Triton and FLASH backends.
        if not isinstance(window_size, torch.Tensor):
            window_size_t = torch.tensor(int(window_size), dtype=torch.long, device=device)
        else:
            window_size_t = window_size.to(device=device, dtype=torch.long)
        mask_mod = FlexAttnFunc._get_mask_mod(
            seq_ids.long().to(device), frame_ids.long().to(device),
            noise_ids.long().to(device), window_size_t,
        )
        FlexAttnFunc.attention_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device,
        )

        text_seq_ids = torch.arange(n_episodes)[:, None].expand(-1, text_seq_len).flatten().long().to(device)
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(seq_ids.long().to(device), text_seq_ids)
        FlexAttnFunc.cross_attention_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device,
        )
    
    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (text_seq_ids[kv_idx] >= 0)
        return seq_mask
    
    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (seq_ids[kv_idx] >= 0)
        
        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] <= frame_ids[q_idx])
        
        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] < frame_ids[q_idx])
        
        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] == frame_ids[q_idx])
        
        def clean2clean_mask(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)
        
        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)
        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)
        
        def block_window_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, window_size: torch.Tensor
        ):
            # window_size is a 0-dim long tensor; broadcast compare works for
            # both Triton and FA4 backends and avoids SymInt closure capture.
            return ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask
       
class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim,
                                                       dim,
                                                       act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(torch.nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode='torch',
    ):
        super().__init__()
        if attn_mode == 'torch':
            self.attn_op = custom_sdpa
        elif attn_mode == 'flashattn':
            self.attn_op = _load_flash_attn_func()
        elif attn_mode == 'flex':
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        ])
        self.norm_q = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'id':
            torch.full((total_tolen, ), -1, device=device),
            "mask":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            "is_pred":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]['mask'][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ):
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                                x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        if kv_cache is not None and kv_cache['k'] is not None:
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            qb = query.shape[0]
            if update_cache == 0:
                # GRADED / rollout path: do NOT write the current step's k/v into
                # the persistent pool. A pool write is an index_put that entangles
                # the whole ~120MB-per-block pool buffer into the autograd graph;
                # its IndexPutBackward then clones the entire buffer during
                # backward (30 blocks x {k,v} x ~120MB = ~6.7GB transient -> OOM,
                # confirmed by a CUDA memory snapshot). Instead attend over
                # [detached historical cache ++ live current k/v]. custom_sdpa is
                # maskless, so attention is order-invariant over keys => identical
                # output (guarded by the logprob-consistency check), while keeping
                # current k/v differentiable (to_k/to_v LoRA) and history detached
                # (history is built under no_grad, i.e. a constant either way).
                # The oldest-first eviction of allocate_slots is replicated so the
                # attended key SET matches production exactly (incl. sliding window).
                cache = self.attn_caches[cache_name]
                mask = cache['mask']
                ids = cache['id']
                key_size = key.shape[1]
                used = mask.nonzero(as_tuple=False).squeeze(-1)
                free_n = int(mask.numel() - used.numel())
                need_evict = key_size - free_n
                if need_evict > 0 and used.numel() > 0:
                    order = torch.argsort(ids[used])
                    hist_valid = used[order[need_evict:]]
                else:
                    hist_valid = used
                hist_k = key_pool[:qb, hist_valid]
                hist_v = value_pool[:qb, hist_valid]
                key = torch.cat([hist_k, key], dim=1)
                value = torch.cat([hist_v, value], dim=1)
            else:
                # update_cache == 1: persist path (video prefix / deferred
                # last_step), always under no_grad, so the index_put write carries
                # no grad_fn and there is no backward clone. Keep original behavior.
                slots = self.update_cache(cache_name,
                                          key,
                                          value,
                                          is_pred=(update_cache == 1))
                mask = self.attn_caches[cache_name]['mask']
                valid = mask.nonzero(as_tuple=False).squeeze(-1)
                key = key_pool[:qb, valid]
                value = value_pool[:qb, valid]

        hidden_states = self.attn_op(query, key, value)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim,
                               inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ) -> torch.Tensor:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1. + scale_msa) +
                              shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states,
                                 norm_hidden_states,
                                 norm_hidden_states,
                                 rotary_emb,
                                 update_cache=update_cache,
                                 cache_name=cache_name)
        hidden_states = (hidden_states.float() +
                         attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(
            hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states,
                                 encoder_hidden_states,
                                 None,
                                 update_cache=0,
                                 cache_name=cache_name)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) *
                              (1. + c_scale_msa) +
                              c_shift_msa).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (hidden_states.float() +
                         ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    TODO
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
                                        # "patch_embedding", 
                                        "patch_embedding_mlp",
                                        "condition_embedder", 
                                        'condition_embedder_action',
                                        "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", 
                             "scale_shift_table", 
                             "scale_shift_table_action",
                             "norm1", 
                             'action_norm1',
                             'text_norm1',
                             "norm2", 
                             'action_norm2',
                             'text_norm2',
                             "norm3",
                             'action_norm3',
                             'text_norm3'
                             ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(self,
                 patch_size=[1, 2, 2],
                 num_attention_heads=24,
                 attention_head_dim=128,
                 in_channels=48,
                 out_channels=48,
                 action_dim=30,
                 text_dim=4096,
                 freq_dim=256,
                 ffn_dim=14336,
                 num_layers=30,
                 cross_attn_norm=True,
                 eps=1e-06,
                 rope_max_seq_len=1024,
                 pos_embed_seq_len=None,
                 attn_mode="torch"):
        r"""
        TODO
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size,
                                      rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim)
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim,
                                ffn_dim,
                                num_attention_heads,
                                cross_attn_norm,
                                eps,
                                attn_mode=attn_mode) for _ in range(num_layers)
        ])

        # Full-block gradient checkpointing, enabled by the GRPO server for the
        # grad recompute (_backward_episode_logprob) only. Recomputing the block
        # (incl. the cache-reading attn1) in backward is safe there because the
        # server DEFERS the no_grad last_step cache write until AFTER .backward(),
        # so self.attn_caches is identical during the original forward and the
        # checkpoint recompute. Gated in forward() on torch.is_grad_enabled() and
        # update_cache == 0 so no_grad rollout/eval and the update_cache==1 write
        # are never checkpointed.
        self.gradient_checkpointing = False

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim,
                                  out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window,
                           latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size):
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(cache_name, total_tolen,
                                      self.num_attention_heads,
                                      self.attention_head_dim, device, dtype, batch_size)
    
    def _input_embed(self, latents, input_type='latent'):
        if input_type == 'latent':
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == 'action':
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == 'text':
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) *
            (W // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    def forward_train(self, input_dict):
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]

        latent_hidden_states = self._input_embed(latent_dict['noisy_latents'], input_type='latent').flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text')

        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        condition_latent_hidden_states = self._input_embed(latent_dict['latent'], input_type='latent').flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(action_dict['latent'], input_type='action').flatten(0, 1)[None]

        hidden_states = torch.cat([latent_hidden_states, 
                                   condition_latent_hidden_states,
                                   action_hidden_states, 
                                   condition_action_hidden_states], dim=1)


        latent_grid_id = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)

        rotary_emb = self.rope(full_grid_id)[:, :, None] 

        latent_time_steps = torch.cat(
            [latent_dict['timesteps'].flatten(0, 1), latent_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        action_time_steps = torch.cat(
            [action_dict['timesteps'].flatten(0, 1), action_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        latent_temb, latent_timestep_proj =self._time_embed(latent_time_steps, 
                        latent_dict['noisy_latents'].shape[-2], 
                        latent_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=False)
        action_temb, action_timestep_proj = self._time_embed(action_time_steps,
                        action_dict['noisy_latents'].shape[-2], 
                        action_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=True)
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        total_length = hidden_states.shape[1]
        # If the trainer pre-aligned a global self-attn shape (FA4 path needs
        # this so all ranks compile the same cute-dsl kernel in lockstep), use
        # it; otherwise fall back to the original "pad to multiple of 128".
        attn_target = input_dict.get('attn_target_length')
        if attn_target is not None:
            padded_length = max(0, int(attn_target) - total_length)
        else:
            padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [latent_hidden_states.shape[1], 
                      condition_latent_hidden_states.shape[1], 
                      action_hidden_states.shape[1], 
                      condition_action_hidden_states.shape[1],
                      padded_length]

        FlexAttnFunc.init_mask(latent_dict['noisy_latents'].shape,
                               action_dict['noisy_latents'].shape,
                               padded_length,
                               input_dict["chunk_size"],
                               window_size=input_dict['window_size'],
                               patch_size=self.patch_size,
                               device=hidden_states.device,
                               text_seq_len=latent_dict['text_emb'].shape[-2],
                               seq_ids_per_frame=input_dict['seq_ids_per_frame'],
                               frame_ids_per_frame=input_dict['frame_ids_per_frame'],
                               n_episodes=input_dict['n_episodes'],
                               )

        for block in self.blocks:
            hidden_states = block(hidden_states,
                                         text_hidden_states,
                                         timestep_proj,
                                         rotary_emb,
                                         update_cache=False)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(latent_hidden_states,
                                             '1 (b l) (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size), b=batch_size)  #
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states,
                                             '1 (b l) c -> b l c',
                                             b=batch_size)  #

        return latent_hidden_states, action_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if train_mode:
            return self.forward_train(input_dict)
        if action_mode:  # action input emb
            latent_hidden_states = rearrange(input_dict['noisy_latents'],
                                             'b c f h w -> b (f h w) c')
            latent_hidden_states = self.action_embedder(
                latent_hidden_states)  # B L1 C
        else:  # latent input emb
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            latent_hidden_states = self.patch_embedding_mlp(
                latent_hidden_states)
        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"])  # B L2 C

        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        use_ckpt = (
            self.gradient_checkpointing
            and torch.is_grad_enabled()
            and update_cache == 0
        )
        for block in self.blocks:
            if use_ckpt:
                # _b=block binds the current block (avoid closure late-binding to
                # the last loop iteration when checkpoint recomputes in backward).
                def _blk(hs, ths, tp, re, _b=block):
                    return _b(hs, ths, tp, re,
                              update_cache=update_cache, cache_name=cache_name)
                latent_hidden_states = torch.utils.checkpoint.checkpoint(
                    _blk,
                    latent_hidden_states,
                    text_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    use_reentrant=False,
                )
            else:
                latent_hidden_states = block(latent_hidden_states,
                                             text_hidden_states,
                                             timestep_proj,
                                             rotary_emb,
                                             update_cache=update_cache,
                                             cache_name=cache_name)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(latent_hidden_states)

        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))  #

        return latent_hidden_states


if __name__ == '__main__':
    model = WanTransformer3DModel(patch_size=[1, 2, 2],
                                  num_attention_heads=24,
                                  attention_head_dim=128,
                                  in_channels=48,
                                  out_channels=48,
                                  action_dim=30,
                                  text_dim=4096,
                                  freq_dim=256,
                                  ffn_dim=14336,
                                  num_layers=30,
                                  cross_attn_norm=True,
                                  eps=1e-6,
                                  rope_max_seq_len=1024,
                                  pos_embed_seq_len=None,
                                  attn_mode="torch")
    print(model)
