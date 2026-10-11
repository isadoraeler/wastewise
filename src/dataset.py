
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_FILE = PROJECT_ROOT / "data" / "splits.csv"


class WasteDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform

        with SPLITS_FILE.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        self.samples = [
            row for row in rows
            if row["split"] == split
        ]

        self.categories = sorted(
            {row["category"] for row in rows}
        )

        self.category_to_index = {
            category: index
            for index, category in enumerate(self.categories)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = PROJECT_ROOT / sample["image_path"]
        image = Image.open(image_path).convert("RGB")

        label = self.category_to_index[sample["category"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
