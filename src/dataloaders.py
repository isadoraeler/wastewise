
from torch.utils.data import DataLoader

from src.dataset import WasteDataset
from src.transforms import get_transforms


BATCH_SIZE = 32


def create_dataloaders():
    train_transform, eval_transform = get_transforms()

    train_dataset = WasteDataset(
        split="train",
        transform=train_transform,
    )

    validation_dataset = WasteDataset(
        split="validation",
        transform=eval_transform,
    )

    test_dataset = WasteDataset(
        split="test",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, validation_loader, test_loader
