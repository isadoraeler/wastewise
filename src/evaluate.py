import argparse
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.dataset import WasteDataset
from src.model import create_model
from src.transforms import get_transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATHS = {
    "baseline": PROJECT_ROOT / "models" / "mobilenet_v3_small_baseline.pth",
    "augmented": PROJECT_ROOT / "models" / "mobilenet_v3_small_augmented.pth",
    "finetuned": PROJECT_ROOT / "models" / "mobilenet_v3_small_finetuned.pth",
}

BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser(
        description="Avalia um modelo WasteWise."
    )

    parser.add_argument(
        "--model",
        choices=["baseline", "augmented", "finetuned"],
        default="augmented",
        help="Modelo que será avaliado.",
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Conjunto de dados usado na avaliação.",
    )

    args = parser.parse_args()

    device = torch.device("cpu")
    model_path = MODEL_PATHS[args.model]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}"
        )

    print(f"Dispositivo: {device}")
    print(f"Modelo avaliado: {args.model}")

    _, eval_transform = get_transforms()

    dataset = WasteDataset(
        split=args.split,
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
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    categories = dataset.categories
    category_ids = list(range(len(categories)))

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=category_ids,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=category_ids,
        average="weighted",
        zero_division=0,
    )

    print(f"\n=== RESULTADOS: {args.split.upper()} ===")
    print(f"Imagens avaliadas: {len(all_labels)}")
    print(f"Acurácia: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"F1 macro: {macro_f1:.4f}")
    print(f"F1 ponderado: {weighted_f1:.4f}")

    print("\n=== RELATÓRIO POR CATEGORIA ===")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=category_ids,
            target_names=categories,
            digits=4,
            zero_division=0,
        )
    )

    print("\n=== MATRIZ DE CONFUSÃO ===")
    print("Linhas = classe real; colunas = classe prevista.")
    print(f"Ordem das categorias: {categories}")

    print(
        confusion_matrix(
            all_labels,
            all_predictions,
            labels=category_ids,
        )
    )


if __name__ == "__main__":
    main()