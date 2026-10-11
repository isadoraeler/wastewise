
import copy
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.dataloaders import create_dataloaders
from src.model import create_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

INITIAL_MODEL_PATH = MODEL_DIR / "mobilenet_v3_small_augmented.pth"
OUTPUT_MODEL_PATH = MODEL_DIR / "mobilenet_v3_small_finetuned.pth"

EPOCHS = 15
BACKBONE_LEARNING_RATE = 0.00001
CLASSIFIER_LEARNING_RATE = 0.0001
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

    if not INITIAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo inicial não encontrado: {INITIAL_MODEL_PATH}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, _ = create_dataloaders()

    model = create_model().to(device)

    # Começa com os pesos do melhor modelo anterior.
    state_dict = torch.load(
        INITIAL_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    backbone_parameters = [
        parameter
        for parameter in model.features.parameters()
        if parameter.requires_grad
    ]

    classifier_parameters = list(model.classifier.parameters())

    optimizer = Adam(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LEARNING_RATE,
            },
            {
                "params": classifier_parameters,
                "lr": CLASSIFIER_LEARNING_RATE,
            },
        ]
    )

    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    print(f"Modelo inicial: {INITIAL_MODEL_PATH.name}")
    print(f"Taxa de aprendizado do extrator: {BACKBONE_LEARNING_RATE}")
    print(f"Taxa de aprendizado do classificador: {CLASSIFIER_LEARNING_RATE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Mantém as camadas BatchNorm do extrator em modo de avaliação.
        # Os pesos dos blocos 11 e 12 continuam recebendo gradientes.
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
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)

    print(f"\nMelhor perda de validação: {best_validation_loss:.4f}")
    print(f"Modelo salvo em: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
