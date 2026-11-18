import torch
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

NUM_CLASSES = 9
UNFREEZE_FROM = 11


def create_model(pretrained=True):
    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_small(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for block in model.features[UNFREEZE_FROM:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    input_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(
        input_features,
        NUM_CLASSES,
    )

    return model