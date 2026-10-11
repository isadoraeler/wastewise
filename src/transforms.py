
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights


def get_transforms():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            preprocess,
        ]
    )

    eval_transform = preprocess

    return train_transform, eval_transform
