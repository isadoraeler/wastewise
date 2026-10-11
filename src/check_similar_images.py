
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "realwaste-main" / "RealWaste"
SPLITS_FILE = PROJECT_ROOT / "data" / "splits.csv"

# Quanto menor a distância, mais parecidas são as imagens.
MAX_DISTANCE = 5


def main():
    if not DATASET_DIR.exists() or not SPLITS_FILE.exists():
        print("Erro: dataset ou splits.csv não encontrado.")
        return

    image_info = {}

    print("Calculando hashes visuais das imagens...")

    for category in sorted(DATASET_DIR.iterdir()):
        if not category.is_dir():
            continue

        for path in sorted(category.glob("*.jpg")):
            try:
                with Image.open(path) as image:
                    visual_hash = imagehash.phash(image.convert("RGB"))

                relative_path = path.relative_to(PROJECT_ROOT).as_posix()
                image_info[relative_path] = {
                    "hash": visual_hash,
                    "category": category.name,
                }

            except Exception as error:
                print(f"Não foi possível analisar {path.name}: {error}")

    print(f"Imagens analisadas: {len(image_info)}")

    # Associa cada imagem ao conjunto definido em splits.csv.
    split_by_path = {}

    import csv

    with SPLITS_FILE.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            split_by_path[row["image_path"]] = row["split"]

    # Agrupa as imagens por categoria para comparar imagens da mesma classe.
    by_category = defaultdict(list)

    for path, info in image_info.items():
        by_category[info["category"]].append((path, info["hash"]))

    suspicious_pairs = []

    print("\nComparando imagens da mesma categoria...")

    for category, images in sorted(by_category.items()):
        for (path_a, hash_a), (path_b, hash_b) in combinations(images, 2):
            distance = hash_a - hash_b

            if distance <= MAX_DISTANCE:
                split_a = split_by_path.get(path_a, "desconhecido")
                split_b = split_by_path.get(path_b, "desconhecido")

                if split_a != split_b:
                    suspicious_pairs.append(
                        (distance, category, path_a, split_a, path_b, split_b)
                    )

    print(f"Pares semelhantes entre conjuntos diferentes: {len(suspicious_pairs)}")

    for distance, category, path_a, split_a, path_b, split_b in suspicious_pairs:
        print(f"\nCategoria: {category} | Distância: {distance}")
        print(f"  [{split_a}] {path_a}")
        print(f"  [{split_b}] {path_b}")

    print("\nVerificação concluída.")
    print("Os pares encontrados são suspeitas para inspeção visual, não duplicatas confirmadas.")


if __name__ == "__main__":
    main()
