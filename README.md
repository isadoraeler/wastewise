# WasteWise — Classificação de Resíduos com IA

O **WasteWise** é um projeto de inteligência artificial voltado à classificação visual de resíduos. A aplicação utiliza um modelo de aprendizado profundo para analisar imagens e estimar a categoria do material apresentado.

O projeto está sendo desenvolvido com Python e PyTorch, utilizando transfer learning com a arquitetura MobileNetV3-Small e o conjunto de dados RealWaste.

## Objetivo

Desenvolver uma aplicação capaz de classificar resíduos a partir de imagens, explorando técnicas de visão computacional e aprendizado profundo para apoiar a identificação visual de materiais.

## Tecnologias utilizadas

- **Python** — linguagem de programação.
- **PyTorch** — construção e execução do modelo de aprendizado profundo.
- **MobileNetV3-Small** — arquitetura utilizada para classificação de imagens.
- **Transfer Learning** — técnica utilizada no desenvolvimento do classificador.
- **Gradio** — interface para interação com o modelo.
- **SQLite** — armazenamento do histórico de análises.
- **RealWaste** — conjunto de dados utilizado no projeto.

## Categorias de classificação

O modelo trabalha com nove categorias de resíduos:

- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

## Funcionalidades

- Classificação de resíduos a partir de imagens.
- Apresentação das categorias previstas e suas respectivas estimativas de confiança.
- Exibição de uma segunda categoria provável para comparação.
- Histórico local das análises realizadas.
- Atualização e exclusão do histórico.
- Interface interativa desenvolvida com Gradio.

## Conjunto de dados

O projeto utiliza o conjunto de dados **RealWaste**, disponibilizado pela University of California, Irvine (UCI), para o desenvolvimento e a avaliação do classificador.

## Status do projeto

**Em desenvolvimento.**

O projeto está sendo construído por etapas, desde a preparação dos dados e treinamento do modelo até a integração com a interface e o armazenamento do histórico. A publicação da aplicação online será realizada após a conclusão e validação dessas etapas.

## Observação

As classificações e estimativas de confiança são produzidas pelo modelo e podem apresentar erros. Os resultados devem ser interpretados como previsões visuais, não como garantia da composição do material ou de sua possibilidade de reciclagem.
