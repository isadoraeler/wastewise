# Experimentos — WasteWise

## Experimento 1 — Baseline

### Objetivo
Estabelecer uma referência inicial para a classificação de resíduos utilizando transfer learning com MobileNetV3-Small.

### Dataset
- Nome: RealWaste
- Total de imagens: 4.752
- Número de categorias: 9
- Treino: 3.323 imagens
- Validação: 710 imagens
- Teste: 719 imagens
- Divisão: 70% treino, 15% validação e 15% teste, aproximadamente.
- Semente aleatória: 42.

### Configuração
- Arquitetura: MobileNetV3-Small.
- Pesos iniciais: pré-treinados no ImageNet.
- Estratégia: congelamento do extrator de características e treinamento do classificador.
- Otimizador: Adam.
- Taxa de aprendizado: 0,001.
- Função de perda: CrossEntropyLoss.
- Tamanho do lote: 32.
- Máximo de épocas: 15.
- Early stopping: paciência de 4 épocas.
- Dispositivo: CPU.

### Resultados de validação
- Melhor época: 8.
- Melhor perda de validação: 0,4212.
- Acurácia: 83,94%.
- F1-score macro: 0,8468.
- F1-score ponderado: 0,8392.

### F1-score por categoria
| Categoria | F1-score |
|---|---:|
| Cardboard | 0,8507 |
| Food Organics | 0,9600 |
| Glass | 0,7207 |
| Metal | 0,8291 |
| Miscellaneous Trash | 0,7531 |
| Paper | 0,8333 |
| Plastic | 0,8114 |
| Textile Trash | 0,8864 |
| Vegetation | 0,9767 |

### Observações
- A perda de treino continuou diminuindo enquanto a perda de validação oscilou, sugerindo possível sobreajuste.
- Glass e Miscellaneous Trash apresentaram os menores F1-scores.
- Foram observadas confusões entre categorias visualmente semelhantes, incluindo Glass e Plastic, Plastic e Metal, e Cardboard e Paper.

### Próxima investigação
Testar aumento de dados no conjunto de treino, mantendo a mesma divisão do dataset e comparando os resultados com este baseline.

### Status
Baseline concluído. Conjunto de teste ainda não utilizado.

## Experimento 2 — Data Augmentation

### Objetivo
Avaliar se transformações aleatórias nas imagens de treino melhoram a generalização do MobileNetV3-Small.

### Alterações
- Espelhamento horizontal aleatório, com probabilidade de 50%.
- Rotação aleatória de até 10 graus.
- Pré-processamento de validação mantido sem transformações aleatórias.
- Arquitetura, divisão dos dados, otimizador e demais hiperparâmetros mantidos.

### Resultados de validação
- Melhor época: 7.
- Melhor perda de validação: 0,3828.
- Acurácia: 86,20%.
- F1-score macro: 0,8666.
- F1-score ponderado: 0,8618.

### Comparação com o baseline
- Acurácia: aumento de 83,94% para 86,20% (+2,26 pontos percentuais).
- F1-score macro: aumento de 0,8468 para 0,8666.
- Perda de validação: redução de 0,4212 para 0,3828.
- Melhorias relevantes observadas em Glass, Miscellaneous Trash, Plastic e Metal.
- Cardboard apresentou redução no F1-score.

### Conclusão provisória
O aumento de dados apresentou resultados melhores nas métricas gerais de validação nesta execução. O resultado é promissor, mas ainda não demonstra, por si só, que a melhoria se repetirá em outras execuções ou em dados externos.

### Próxima etapa
Investigar os erros restantes e definir o modelo candidato para avaliação final. O conjunto de teste permanece reservado.

### Status
Experimento concluído; resultados de validação registrados.