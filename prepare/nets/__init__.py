from .mobilenetv2 import mobilenetv2
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg11, vgg13, vgg16
from .vision_transformer import vit_b_16
from .swin_transformer import swin_transformer_base, swin_transformer_small, swin_transformer_tiny
from .clip_classifier1 import ImageClassifier

get_model_from_name = {
    "mobilenetv2": mobilenetv2,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vit_b_16": vit_b_16,
    "swin_transformer_tiny": swin_transformer_tiny,
    "swin_transformer_small": swin_transformer_small,
    "swin_transformer_base": swin_transformer_base
}
