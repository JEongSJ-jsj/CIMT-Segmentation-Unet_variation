import argparse
import torch
from importlib import import_module

_DEFAULTS = {
    "unet": dict(base=33),                          # ≈ 8.25M params
    "unetpp": dict(base=32),                        # ≈ 9.16M params
    "resunet": dict(base=36),                       # ≈ 10.27M params
    "attention_unet": dict(base_c=64),              # ≈ 31.56M params (논문형)
    "attention_resunet": dict(base=35),             # ≈ 9.86M params (fixed attention gate)
    "seunet": dict(base=33),                        # ≈ 8.32M params
    "denseunet": dict(base=48, growth=24, layers=6),# ≈ 8.9M params (안정형, 256² output)
    "inceptionunet": dict(base=60),                 # ≈ 6.9M params (fixed 256² output)
    "transunet": dict(embed_dim=256, depth=4, heads=4, mlp_ratio=4),  # ≈ 6.04M params, safe ViT
    "unext": dict(base=40, expansion=4),            # ≈ 0.42M params (논문형)
}

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
_CLASSNAME_MAP = {
    "unet": "UNet",
    "unetpp": "UNetPP",
    "resunet": "ResUNet",
    "attention_unet": "AttentionUNet",
    "attention_resunet": "AttentionResUNet",
    "seunet": "SEUNet",
    "denseunet": "DenseUNet",
    "inceptionunet": "InceptionUNet",
    "transunet": "TransUNet",
    "unext": "UNeXt",
}

def test_model(name, **kwargs):
    try:
        mod = import_module(f"models.{name}")
        cls_name = _CLASSNAME_MAP.get(name, "".join(part.capitalize() for part in name.split("_")))

        model_cls = getattr(mod, cls_name)
        model = model_cls(**kwargs)
        total, train = count_params(model)
        print(f"[{name:<16}] params (total/train): {total:,} / {train:,}")
    except Exception as e:
        print(f"[{name:<16}] ❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--kw", nargs="*", help="key=value style args")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for name, kw in _DEFAULTS.items():
            test_model(name, **kw)
    elif args.model:
        kw = dict(kv.split("=") for kv in (args.kw or []))
        for k in kw:
            try: kw[k] = int(kw[k])
            except: pass
        test_model(args.model, **kw)
    else:
        print("Usage: python parametercheck.py --model unet")
