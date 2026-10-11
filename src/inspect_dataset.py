
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Localiza a raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define os caminhos do dataset e do arquivo de divisões.
DATASET_DIR = PROJECT_ROOT / "data" / "realwaste-main" / "RealWaste"
SPLITS_FILE = PROJECT_ROOT / "data" / "splits.csv"


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset não encontrado: {DATASET_DIR}")
        return

    categories = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    print(f"Categorias encontradas: {len(categories)}\n")

    total_images = 0
    valid_images = 0
    invalid_images = []

    for category in categories:
        files = [
            file for file in category.iterdir()
            if file.is_file() and file.suffix.lower() == ".jpg"
        ]

        total_images += len(files)

        for file in files:
            try:
                with Image.open(file) as image:
                    image.verify()

                valid_images += 1

            except Exception:
                invalid_images.append(file)

        print(f"{category.name}: {len(files)} imagens")

    print("\nResultado da validação:")
    print(f"Total de imagens: {total_images}")
    print(f"Imagens válidas: {valid_images}")
    print(f"Imagens inválidas: {len(invalid_images)}")

    if invalid_images:
        print("\nArquivos que precisam ser verificados:")
        for file in invalid_images:
            print(file.relative_to(DATASET_DIR))

    # Lê as categorias registradas no arquivo de divisões.
    with SPLITS_FILE.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    counts = Counter(row["category"] for row in rows)

    # Organiza os dados para o gráfico.
    names = list(counts.keys())
    quantities = list(counts.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, quantities)

    plt.title("WasteWise — Quantidade de imagens por categoria")
    plt.xlabel("Categoria")
    plt.ylabel("Quantidade de imagens")
    plt.xticks(rotation=35, ha="right")

    # Mostra o número de imagens acima de cada barra.
    for bar, quantity in zip(bars, quantities):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(quantity),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
