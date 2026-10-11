
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from src.dataset import WasteDataset
from src.model import create_model
from src.transforms import get_transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_v3_small_augmented.pth"
BATCH_SIZE = 32


def main():
    device = torch.device("cpu")
    print(f"Dispositivo: {device}")

    _, eval_transform = get_transforms()

    validation_dataset = WasteDataset(
        split="validation",
        transform=eval_transform,
    )

    validation_loader = DataLoader(
        validation_dataset,
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

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    categories = validation_dataset.categories

    accuracy = accuracy_score(all_labels, all_predictions)

    print("\n=== RESULTADOS DA VALIDAÇÃO ===")
    print(f"Imagens avaliadas: {len(all_labels)}")
    print(f"Acurácia: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\n=== RELATÓRIO POR CATEGORIA ===")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(categories))),
            target_names=categories,
            digits=4,
            zero_division=0,
        )
    )

    print("\n=== MATRIZ DE CONFUSÃO ===")
    print(f"Ordem das categorias: {categories}")
    print(
        confusion_matrix(
            all_labels,
            all_predictions,
            labels=list(range(len(categories))),
        )
    )


if __name__ == "__main__":
    main()
