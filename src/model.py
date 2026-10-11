
import torch
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


NUM_CLASSES = 9


def create_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # Congela as camadas de extração de características.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Substitui a camada final para classificar as nove categorias.
    input_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(
        input_features,
        NUM_CLASSES,
    )

    return model
