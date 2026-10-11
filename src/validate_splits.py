
import csv
from collections import Counter, defaultdict
from pathlib import Path

# Localiza a raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define os caminhos.
DATASET_DIR = PROJECT_ROOT / "data" / "realwaste-main" / "RealWaste"
SPLITS_FILE = PROJECT_ROOT / "data" / "splits.csv"

EXPECTED_TOTAL = 4752
EXPECTED_CATEGORIES = 9
EXPECTED_SPLITS = {"train", "validation", "test"}


def main():
    if not SPLITS_FILE.exists():
        print(f"Arquivo não encontrado: {SPLITS_FILE}")
        return

    with SPLITS_FILE.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    required_columns = {"image_path", "category", "split"}

    if not required_columns.issubset(reader.fieldnames or []):
        print("Erro: o CSV não contém todas as colunas esperadas.")
        return

    print(f"Total de registros: {len(rows)}")

    paths = [row["image_path"] for row in rows]
    duplicates = [
        path for path, count in Counter(paths).items()
        if count > 1
    ]

    missing_files = [
        path for path in paths
        if not (PROJECT_ROOT / path).is_file()
    ]

    categories = {row["category"] for row in rows}
    splits = {row["split"] for row in rows}

    counts_by_split = Counter(row["split"] for row in rows)
    counts_by_category = defaultdict(Counter)

    for row in rows:
        counts_by_category[row["category"]][row["split"]] += 1

    print(f"Categorias encontradas: {len(categories)}")
    print(f"Conjuntos encontrados: {sorted(splits)}")
    print(f"Caminhos duplicados: {len(duplicates)}")
    print(f"Arquivos ausentes: {len(missing_files)}")

    print("\nQuantidade por conjunto:")
    for split in ("train", "validation", "test"):
        print(f"{split}: {counts_by_split[split]}")

    print("\nCategorias por conjunto:")
    for category in sorted(categories):
        counts = counts_by_category[category]
        print(
            f"{category}: "
            f"treino={counts['train']}, "
            f"validação={counts['validation']}, "
            f"teste={counts['test']}"
        )

    errors = []

    if len(rows) != EXPECTED_TOTAL:
        errors.append("O total de registros não é 4.752.")

    if len(categories) != EXPECTED_CATEGORIES:
        errors.append("A quantidade de categorias não é 9.")

    if splits != EXPECTED_SPLITS:
        errors.append("Os conjuntos não correspondem aos esperados.")

    if duplicates:
        errors.append("Existem caminhos duplicados.")

    if missing_files:
        errors.append("Existem arquivos ausentes.")

    if any(not row["image_path"] for row in rows):
        errors.append("Existem caminhos vazios.")

    if any(not row["category"] for row in rows):
        errors.append("Existem categorias vazias.")

    if any(row["split"] not in EXPECTED_SPLITS for row in rows):
        errors.append("Existem registros com conjunto inválido.")

    if any(set(counts_by_category[category]) != EXPECTED_SPLITS
           for category in categories):
        errors.append("Alguma categoria não aparece nos três conjuntos.")

    print("\nResultado:")
    if errors:
        for error in errors:
            print(f"- ERRO: {error}")
    else:
        print("Todas as verificações passaram.")


if __name__ == "__main__":
    main()
