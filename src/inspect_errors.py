
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import WasteDataset
from src.model import create_model
from src.transforms import get_transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_v3_small.pth"

BATCH_SIZE = 32
MAX_IMAGES = 20


def main():
    device = torch.device("cpu")
    print(f"Dispositivo: {device}")

    _, eval_transform = get_transforms()

    dataset = WasteDataset(
        split="validation",
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    categories = dataset.categories
    errors = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            wrong_indices = (
                predictions != labels.to(device)
            ).nonzero(as_tuple=True)[0]

            for index in wrong_indices.tolist():
                dataset_index = batch_index * BATCH_SIZE + index

                errors.append(
                    {
                        "dataset_index": dataset_index,
                        "real": categories[labels[index].item()],
                        "predicted": categories[
                            predictions[index].item()
                        ],
                    }
                )

    print(f"\nTotal de erros encontrados: {len(errors)}")

    if not errors:
        print("Nenhum erro encontrado na validação.")
        return

    print("\nPrimeiros erros:")
    for error in errors[:MAX_IMAGES]:
        print(
            f"Imagem {error['dataset_index']}: "
            f"real = {error['real']} | "
            f"previsto = {error['predicted']}"
        )

    images_to_show = errors[:MAX_IMAGES]
    columns = 4
    rows = (len(images_to_show) + columns - 1) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, rows * 4),
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, error in zip(axes, images_to_show):
        sample = dataset.samples[error["dataset_index"]]
        image_path = PROJECT_ROOT / sample["image_path"]

        with __import__("PIL.Image", fromlist=["Image"]).open(image_path) as image:
            axis.imshow(image.convert("RGB"))

        axis.set_title(
            f"Real: {error['real']}\n"
            f"Previsto: {error['predicted']}",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes[len(images_to_show):]:
        axis.axis("off")

    figure.suptitle("Erros do WasteWise — conjunto de validação")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
