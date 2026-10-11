
import hashlib
from collections import defaultdict
from pathlib import Path

# Localiza a pasta das imagens.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "realwaste-main" / "RealWaste"


def calculate_hash(image_path):
    """Calcula uma impressão digital do conteúdo de um arquivo."""
    hasher = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset não encontrado: {DATASET_DIR}")
        return

    files_by_hash = defaultdict(list)
    total_images = 0

    print("Verificando arquivos do dataset...\n")

    for category in sorted(DATASET_DIR.iterdir()):
        if not category.is_dir():
            continue

        for image_path in category.iterdir():
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() != ".jpg":
                continue

            total_images += 1
            image_hash = calculate_hash(image_path)
            files_by_hash[image_hash].append(image_path)

    duplicate_groups = [
        paths for paths in files_by_hash.values()
        if len(paths) > 1
    ]

    print(f"Total de imagens analisadas: {total_images}")
    print(f"Grupos de arquivos idênticos: {len(duplicate_groups)}")

    for group in duplicate_groups:
        print("\nPossível grupo duplicado:")

        for image_path in group:
            print(f"- {image_path.relative_to(DATASET_DIR)}")


if __name__ == "__main__":
    main()
