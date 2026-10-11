
import csv
import random
from pathlib import Path

# Localiza a raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define os caminhos de entrada e saída.
DATASET_DIR = PROJECT_ROOT / "data" / "realwaste-main" / "RealWaste"
OUTPUT_FILE = PROJECT_ROOT / "data" / "splits.csv"

# Configurações da divisão.
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset não encontrado: {DATASET_DIR}")
        return

    random.seed(RANDOM_SEED)
    rows = []

    categories = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    for category in categories:
        images = sorted(
            file for file in category.iterdir()
            if file.is_file() and file.suffix.lower() == ".jpg"
        )

        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        validation_end = train_end + int(total * VALIDATION_RATIO)

        train_images = images[:train_end]
        validation_images = images[train_end:validation_end]
        test_images = images[validation_end:]

        for image in train_images:
            rows.append((image, category.name, "train"))

        for image in validation_images:
            rows.append((image, category.name, "validation"))

        for image in test_images:
            rows.append((image, category.name, "test"))

        print(
            f"{category.name}: "
            f"treino={len(train_images)}, "
            f"validação={len(validation_images)}, "
            f"teste={len(test_images)}"
        )

    random.shuffle(rows)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "category", "split"])

        for image, category, split in rows:
            relative_path = image.relative_to(PROJECT_ROOT)
            writer.writerow([relative_path.as_posix(), category, split])

    print(f"\nTotal de imagens: {len(rows)}")
    print(f"Arquivo criado: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
