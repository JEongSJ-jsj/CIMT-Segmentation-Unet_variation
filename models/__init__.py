from .unet import UNet
from .unetpp import UNetPP
from .resunet import ResUNet
from .attention_unet import AttentionUNet
from .attention_resunet import AttentionResUNet
from .seunet import SEUNet
from .denseunet import DenseUNet
from .inceptionunet import InceptionUNet
from .transunet import TransUNet
from .unext import UNeXt

def get_model(model_name: str, **kwargs):
    model_name = model_name.lower()
    models = {
        "unet": UNet,
        "unetpp": UNetPP,
        "resunet": ResUNet,
        "attention_unet": AttentionUNet,
        "attention_resunet": AttentionResUNet,
        "seunet": SEUNet,
        "denseunet": DenseUNet,
        "inceptionunet": InceptionUNet,
        "transunet": TransUNet,
        "unext": UNeXt,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")
    return models[model_name](in_ch=1, out_ch=1, **kwargs)
