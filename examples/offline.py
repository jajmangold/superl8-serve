# SPDX-License-Identifier: MIT
"""Minimal offline demo (scaffold). Shows the intended entry points; the generate()
loop lands with the engine port. Run inside the superl8 container (needs superl8 + CUDA)."""
from superl8serve import ServeConfig, load_superl8_checkpoint
from superl8serve.loader import checkpoint_info


def main(ckpt: str = "model.superl8") -> None:
    cfg = ServeConfig(model=ckpt, weight_bits=8, kv_cache_dtype="int8")
    print("config:", cfg)
    print("checkpoint:", checkpoint_info(ckpt))
    # Zero-transform weight load (mmap + copy), optionally a single PP stage:
    weights = load_superl8_checkpoint(ckpt, device="cuda")
    print(f"loaded {len(weights)} tensors resident (dp4a layout, no dequant)")
    # TODO: engine.generate(prompts, sampling_params) once the engine port lands.


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "model.superl8")
