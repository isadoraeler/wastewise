
import copy
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.dataloaders import create_dataloaders
from src.model import create_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 4


def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_images = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

    return total_loss / total_images


def main():
    device = torch.device("cpu")
    print(f"Dispositivo: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, _ = create_dataloaders()
    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.features.eval()

        total_train_loss = 0.0
        total_train_images = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_images += batch_size

        train_loss = total_train_loss / total_train_images

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"Época {epoch:02d}/{EPOCHS} | "
            f"Treino: {train_loss:.4f} | "
            f"Validação: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validação não melhorou.")
            break

    model.load_state_dict(best_state)

    output_path = MODEL_DIR / "mobilenet_v3_small_augmented.pth"
    torch.save(model.state_dict(), output_path)

    print(f"\nMelhor perda de validação: {best_validation_loss:.4f}")
    print(f"Modelo salvo em: {output_path}")


if __name__ == "__main__":
    main()
