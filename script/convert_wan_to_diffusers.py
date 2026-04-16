"""Convert a local Wan2.2-TI2V-5B checkpoint to the Lingbot-VA layout.

Usage::

    python script/convert_wan_to_diffusers.py \
        --wan_path /path/to/Wan2.2-TI2V-5B \
        --text_encoder_dir /path/to/text_encoder \
        --tokenizer_dir /path/to/tokenizer \
        --output_path /path/to/output

The text encoder and tokenizer directories must already be in HuggingFace
format (e.g. saved via ``save_pretrained``).  They are copied as-is into
the output.

The Lingbot-specific ``condition_embedder_action`` is warm-started from
the converted ``condition_embedder``; ``action_embedder`` and
``action_proj_out`` are freshly initialised (default ``nn.Linear`` init).
"""

import argparse
import pathlib
import shutil
from copy import deepcopy
from typing import Any, Dict

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

from diffusers import AutoencoderKLWan
from wan_va.modules.model import WanTransformer3DModel


# ---------------------------------------------------------------------------
# Transformer key renaming (unchanged from upstream diffusers)
# ---------------------------------------------------------------------------

TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Norm swap: original order is norm1, norm3, norm2 → we normalise to 1, 2, 3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # I2V / FLF2V image conditioning
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # Self-attention
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    # Cross-attention
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}


# ---------------------------------------------------------------------------
# VAE config for Wan 2.2 (unchanged from upstream diffusers)
# ---------------------------------------------------------------------------

vae22_diffusers_config = {
    "base_dim": 160,
    "z_dim": 48,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "decoder_base_dim": 256,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
    "patch_size": 2,
    "latents_mean": [
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
        -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
        -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
        -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
         0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
    ],
    "latents_std": [
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    ],
    "clip_output": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> None:
    state_dict[new_key] = state_dict.pop(old_key)


def load_sharded_safetensors(dir: pathlib.Path):
    file_paths = list(dir.glob("diffusion_pytorch_model*.safetensors"))
    state_dict = {}
    for path in sorted(file_paths):
        if "index" not in path.name:
            state_dict.update(load_file(path))
    return state_dict


# ---------------------------------------------------------------------------
# Transformer conversion
# ---------------------------------------------------------------------------

def convert_transformer(wan_dir: pathlib.Path):
    diffusers_config = {
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "in_channels": 48,
        "num_attention_heads": 24,
        "num_layers": 30,
        "out_channels": 48,
        "action_dim": 30,
        "patch_size": [1, 2, 2],
        "text_dim": 4096,
        "attn_mode": "flex",
    }

    original_state_dict = load_sharded_safetensors(wan_dir)
    print(f"  Loaded {len(original_state_dict)} transformer keys from {wan_dir}")

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    # Wan uses a Conv3d patch_embedding; Lingbot uses an equivalent nn.Linear
    # named patch_embedding_mlp.  Rename and reshape the weight accordingly.
    w = original_state_dict.pop("patch_embedding.weight")
    original_state_dict["patch_embedding_mlp.weight"] = w.reshape(w.shape[0], -1)
    original_state_dict["patch_embedding_mlp.bias"] = original_state_dict.pop("patch_embedding.bias")

    with init_empty_weights():
        transformer = WanTransformer3DModel(**diffusers_config)

    missing, unexpected = transformer.load_state_dict(original_state_dict, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in Wan checkpoint after renaming — likely a "
            f"mapping bug:\n  " + "\n  ".join(sorted(unexpected)))

    EXPECTED_MISSING_PREFIXES = ("action_embedder.", "action_proj_out.", "condition_embedder_action.")
    truly_unexpected_missing = [k for k in missing if not k.startswith(EXPECTED_MISSING_PREFIXES)]
    if truly_unexpected_missing:
        raise RuntimeError(
            f"Keys missing that should have come from the Wan checkpoint — "
            f"likely a mapping bug:\n  " + "\n  ".join(sorted(truly_unexpected_missing)))

    print(f"  Missing keys (Lingbot-specific, will be initialised): {len(missing)}")
    for k in sorted(missing):
        print(f"    - {k}")

    # Warm-start condition_embedder_action from condition_embedder (mirrors model.py).
    transformer.condition_embedder_action = deepcopy(transformer.condition_embedder)

    # New action layers: replace meta-tensor modules with default nn.Linear init.
    inner_dim = diffusers_config["num_attention_heads"] * diffusers_config["attention_head_dim"]
    transformer.action_embedder = torch.nn.Linear(diffusers_config["action_dim"], inner_dim)
    transformer.action_proj_out = torch.nn.Linear(inner_dim, diffusers_config["action_dim"])

    return transformer.to("cpu")


# ---------------------------------------------------------------------------
# VAE conversion (unchanged from upstream diffusers convert_vae_22)
# ---------------------------------------------------------------------------

def convert_vae_22(wan_dir: pathlib.Path):
    vae_ckpt_path = wan_dir / "Wan2.2_VAE.pth"
    old_state_dict = torch.load(vae_ckpt_path, weights_only=True)
    new_state_dict = {}
    print(f"  Loaded {len(old_state_dict)} VAE keys from {vae_ckpt_path}")

    # Create mappings for specific components
    middle_key_mapping = {
        # Encoder middle block
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
        # Decoder middle block
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }

    # Create a mapping for attention blocks
    attention_mapping = {
        # Encoder middle attention
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
        # Decoder middle attention
        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }

    # Create a mapping for the head components
    head_mapping = {
        # Encoder head
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        # Decoder head
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    # Create a mapping for the quant components
    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    # Process each key in the state dict
    for key, value in old_state_dict.items():
        # Handle middle block keys using the mapping
        if key in middle_key_mapping:
            new_key = middle_key_mapping[key]
            new_state_dict[new_key] = value
        # Handle attention blocks using the mapping
        elif key in attention_mapping:
            new_key = attention_mapping[key]
            new_state_dict[new_key] = value
        # Handle head keys using the mapping
        elif key in head_mapping:
            new_key = head_mapping[key]
            new_state_dict[new_key] = value
        # Handle quant keys using the mapping
        elif key in quant_mapping:
            new_key = quant_mapping[key]
            new_state_dict[new_key] = value
        # Handle encoder conv1
        elif key == "encoder.conv1.weight":
            new_state_dict["encoder.conv_in.weight"] = value
        elif key == "encoder.conv1.bias":
            new_state_dict["encoder.conv_in.bias"] = value
        # Handle decoder conv1
        elif key == "decoder.conv1.weight":
            new_state_dict["decoder.conv_in.weight"] = value
        elif key == "decoder.conv1.bias":
            new_state_dict["decoder.conv_in.bias"] = value
        # Handle encoder downsamples
        elif key.startswith("encoder.downsamples."):
            # Change encoder.downsamples to encoder.down_blocks
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")

            # Handle residual blocks - change downsamples to resnets and rename components
            if "residual" in new_key or "shortcut" in new_key:
                # Change the second downsamples to resnets
                new_key = new_key.replace(".downsamples.", ".resnets.")

                # Rename residual components
                if ".residual.0.gamma" in new_key:
                    new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
                elif ".residual.2.weight" in new_key:
                    new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
                elif ".residual.2.bias" in new_key:
                    new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
                elif ".residual.3.gamma" in new_key:
                    new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
                elif ".residual.6.weight" in new_key:
                    new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
                elif ".residual.6.bias" in new_key:
                    new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
                elif ".shortcut.weight" in new_key:
                    new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")
                elif ".shortcut.bias" in new_key:
                    new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")

            # Handle resample blocks - change downsamples to downsampler and remove index
            elif "resample" in new_key or "time_conv" in new_key:
                # Change the second downsamples to downsampler and remove the index
                parts = new_key.split(".")
                # Find the pattern: encoder.down_blocks.X.downsamples.Y.resample...
                # We want to change it to: encoder.down_blocks.X.downsampler.resample...
                if len(parts) >= 4 and parts[3] == "downsamples":
                    # Remove the index (parts[4]) and change downsamples to downsampler
                    new_parts = parts[:3] + ["downsampler"] + parts[5:]
                    new_key = ".".join(new_parts)

            new_state_dict[new_key] = value

        # Handle decoder upsamples
        elif key.startswith("decoder.upsamples."):
            # Change decoder.upsamples to decoder.up_blocks
            new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")

            # Handle residual blocks - change upsamples to resnets and rename components
            if "residual" in new_key or "shortcut" in new_key:
                # Change the second upsamples to resnets
                new_key = new_key.replace(".upsamples.", ".resnets.")

                # Rename residual components
                if ".residual.0.gamma" in new_key:
                    new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
                elif ".residual.2.weight" in new_key:
                    new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
                elif ".residual.2.bias" in new_key:
                    new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
                elif ".residual.3.gamma" in new_key:
                    new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
                elif ".residual.6.weight" in new_key:
                    new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
                elif ".residual.6.bias" in new_key:
                    new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
                elif ".shortcut.weight" in new_key:
                    new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")
                elif ".shortcut.bias" in new_key:
                    new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")

            # Handle resample blocks - change upsamples to upsampler and remove index
            elif "resample" in new_key or "time_conv" in new_key:
                # Change the second upsamples to upsampler and remove the index
                parts = new_key.split(".")
                if len(parts) >= 4 and parts[3] == "upsamples":
                    # Remove the index (parts[4]) and change upsamples to upsampler
                    new_parts = parts[:3] + ["upsampler"] + parts[5:]
                    new_key = ".".join(new_parts)

            new_state_dict[new_key] = value
        else:
            # Keep other keys unchanged
            new_state_dict[key] = value

    with init_empty_weights():
        vae = AutoencoderKLWan(**vae22_diffusers_config)
    vae.load_state_dict(new_state_dict, strict=True, assign=True)
    return vae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert a Wan2.2-TI2V-5B checkpoint to the Lingbot-VA layout"
    )
    parser.add_argument("--wan_path", type=str, required=True,
                        help="Local path to the Wan2.2-TI2V-5B directory")
    parser.add_argument("--text_encoder_dir", type=str, required=True,
                        help="Local directory with the text encoder in HuggingFace format")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Local directory with the tokenizer in HuggingFace format")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", default="none", choices=["fp32", "fp16", "bf16", "none"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    wan_dir = pathlib.Path(args.wan_path)
    output = pathlib.Path(args.output_path)

    # 1. Transformer
    print("[1/4] Converting transformer …")
    transformer = convert_transformer(wan_dir)

    # If user has specified "none", we keep the original dtypes of the state dict without any conversion
    if args.dtype != "none":
        dtype = DTYPE_MAPPING[args.dtype]
        transformer.to(dtype)

    transformer.save_pretrained(output / "transformer", safe_serialization=True, max_shard_size="5GB")
    del transformer
    print(f"  → {output / 'transformer'}")

    # 2. VAE
    print("[2/4] Converting VAE …")
    vae = convert_vae_22(wan_dir)
    vae.save_pretrained(output / "vae", safe_serialization=True)
    del vae
    print(f"  → {output / 'vae'}")

    # 3. Text encoder (copy existing HF directory)
    print(f"[3/4] Copying text encoder from {args.text_encoder_dir} …")
    te_dst = output / "text_encoder"
    shutil.copytree(args.text_encoder_dir, te_dst)
    print(f"  → {te_dst}")

    # 4. Tokenizer (copy existing HF directory)
    print(f"[4/4] Copying tokenizer from {args.tokenizer_dir} …")
    tok_dst = output / "tokenizer"
    shutil.copytree(args.tokenizer_dir, tok_dst)
    print(f"  → {tok_dst}")

    print(f"\nDone. Output at {output}")
