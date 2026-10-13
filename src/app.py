from pathlib import Path

import gradio as gr
import torch

from src.database import (
    clear_analysis_history,
    get_analysis_history,
    save_analysis,
)
from src.model import create_model
from src.transforms import get_transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mobilenet_v3_small_augmented.pth"
)

CATEGORIES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

DEVICE = torch.device("cpu")

_, eval_transform = get_transforms()


def get_history_table():
    history = get_analysis_history()

    return [
        [
            item["id"],
            item["predicted_category"],
            round(item["confidence"], 2),
            item["second_category"],
            round(item["second_confidence"], 2),
            item["created_at"],
        ]
        for item in history
    ]


def clear_history():
    clear_analysis_history()
    return get_history_table()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pesos do modelo não encontrados: {MODEL_PATH}"
        )

    model = create_model()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


model = load_model()


def predict(image):
    if image is None:
        return {}, "Envie uma imagem para começar."

    image = image.convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    scores = {
        category: float(probabilities[index].item())
        for index, category in enumerate(CATEGORIES)
    }

    ranked_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    predicted_category, first_probability = ranked_scores[0]
    second_category, second_probability = ranked_scores[1]

    confidence = first_probability * 100
    second_confidence = second_probability * 100

    save_analysis(
        predicted_category=predicted_category,
        confidence=confidence,
        second_category=second_category,
        second_confidence=second_confidence,
    )

    confidence_difference = confidence - second_confidence

    if confidence < 60:
        status = (
            "### Resultado incerto\n\n"
            "O modelo não conseguiu identificar uma categoria "
            "com confiança estimada de pelo menos 60%.\n\n"
            "Verifique o material manualmente antes de decidir "
            "como descartá-lo ou reciclá-lo."
        )
    else:
        status = f"### Categoria prevista: {predicted_category}"

    if confidence_difference < 15:
        comparison_message = (
            f"**Atenção:** o modelo ficou dividido entre "
            f"{predicted_category} ({confidence:.2f}%) e "
            f"{second_category} ({second_confidence:.2f}%)."
        )
    else:
        comparison_message = (
            f"**Segunda opção:** {second_category} "
            f"({second_confidence:.2f}%)."
        )

    message = (
        f"{status}\n\n"
        f"**Categoria mais provável:** {predicted_category}\n\n"
        f"**Confiança estimada:** {confidence:.2f}%\n\n"
        f"{comparison_message}\n\n"
        "A confiança é uma estimativa do modelo, não uma garantia "
        "de que a classificação está correta.\n\n"
        "A previsão é apenas visual. Confirme as orientações locais "
        "antes de decidir como descartar ou reciclar o resíduo."
    )

    return scores, message


with gr.Blocks() as demo:
    gr.Markdown("# WasteWise — Classificador visual de resíduos")

    gr.Markdown(
        "Envie uma foto para obter uma estimativa da categoria "
        "do resíduo usando inteligência artificial."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Envie uma imagem do resíduo",
            )

            predict_button = gr.Button(
                "Analisar imagem",
                variant="primary",
            )

        with gr.Column():
            prediction_output = gr.Label(
                num_top_classes=3,
                label="Categorias previstas",
            )

            message_output = gr.Markdown()

    gr.Markdown("## Histórico de análises")

    with gr.Row():
        refresh_button = gr.Button("Atualizar histórico")

        clear_button = gr.Button(
            "Limpar histórico",
            variant="stop",
        )

    history_output = gr.Dataframe(
        headers=[
            "ID",
            "Categoria prevista",
            "Confiança (%)",
            "Segunda categoria",
            "Segunda confiança (%)",
            "Data e hora",
        ],
        datatype=[
            "number",
            "str",
            "number",
            "str",
            "number",
            "str",
        ],
        value=get_history_table,
        interactive=False,
        label="Análises salvas",
    )

    confirmation_message = gr.Markdown(
        "Tem certeza de que deseja excluir todo o histórico?",
        visible=False,
    )

    confirm_clear = gr.Button(
        "Confirmar exclusão",
        variant="stop",
        visible=False,
    )

    cancel_clear = gr.Button(
        "Cancelar",
        visible=False,
    )

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            prediction_output,
            message_output,
        ],
    ).then(
        fn=get_history_table,
        inputs=None,
        outputs=history_output,
    )

    refresh_button.click(
        fn=get_history_table,
        inputs=None,
        outputs=history_output,
    )

    clear_button.click(
        fn=lambda: (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        ),
        inputs=None,
        outputs=[
            confirmation_message,
            confirm_clear,
            cancel_clear,
        ],
    )

    confirm_clear.click(
        fn=clear_history,
        inputs=None,
        outputs=history_output,
    ).then(
        fn=lambda: (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        inputs=None,
        outputs=[
            confirmation_message,
            confirm_clear,
            cancel_clear,
        ],
    )

    cancel_clear.click(
        fn=lambda: (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        inputs=None,
        outputs=[
            confirmation_message,
            confirm_clear,
            cancel_clear,
        ],
    )


if __name__ == "__main__":
    demo.launch()