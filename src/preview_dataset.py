
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "realwaste-main" / "RealWaste"

RANDOM_SEED = 42
IMAGES_PER_CATEGORY = 2


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset não encontrado: {DATASET_DIR}")
        return

    random.seed(RANDOM_SEED)

    categories = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    for category in categories:
        images = sorted(
            file for file in category.iterdir()
            if file.is_file() and file.suffix.lower() == ".jpg"
        )

        selected_images = random.sample(
            images,
            min(IMAGES_PER_CATEGORY, len(images))
        )

        fig, axes = plt.subplots(
            1, len(selected_images),
            figsize=(12, 7)
        )

        if len(selected_images) == 1:
            axes = [axes]

        fig.suptitle(
            f"WasteWise — Categoria: {category.name}",
            fontsize=16
        )

        for ax, image_path in zip(axes, selected_images):
            with Image.open(image_path) as image:
                ax.imshow(image.copy())

            ax.set_title(image_path.name, fontsize=11)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
