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

### Experimento 3 — Fine-tuning parcial

**Objetivo:** verificar se o ajuste dos dois últimos blocos da MobileNetV3-Small melhora o modelo com data augmentation.

**Configuração:**
- Modelo inicial: `mobilenet_v3_small_augmented.pth`
- Blocos treináveis do extrator: 11 e 12
- Taxa de aprendizado do extrator: 0,00001
- Taxa de aprendizado do classificador: 0,0001
- Otimizador: Adam
- Função de perda: CrossEntropyLoss
- Máximo de épocas: 15
- Early stopping: paciência de 4 épocas
- Conjunto utilizado para acompanhamento: validação

**Resultados:**

| Época | Perda de treino | Perda de validação |
|---|---:|---:|
| 1 | 0,0784 | 0,3983 |
| 2 | 0,0671 | 0,3881 |
| 3 | 0,0530 | 0,3916 |
| 4 | 0,0510 | 0,4013 |
| 5 | 0,0434 | 0,3999 |
| 6 | 0,0407 | 0,4011 |

Melhor perda de validação: **0,3881**, na época 2.

**Conclusão provisória:** o treinamento foi interrompido pelo early stopping após seis épocas. A perda de treino continuou diminuindo, enquanto a de validação deixou de melhorar. O melhor valor de validação ainda ficou acima do melhor resultado do experimento com augmentation (0,3828). Portanto, ainda não há evidência de que o fine-tuning seja superior. A decisão será tomada após avaliar as métricas de classificação.

**Checkpoint:** `models/mobilenet_v3_small_finetuned.pth`

**Status:** aguardando avaliação de acurácia, F1 e matriz de confusão na validação.

### Experimento 4 — Fine-tuning parcial

**Objetivo:** verificar se o ajuste parcial do extrator de características melhora o modelo com data augmentation.

**Configuração:**
- Modelo inicial: `mobilenet_v3_small_augmented.pth`
- Arquitetura: MobileNetV3-Small pré-treinada
- Blocos ajustados: 11 e 12
- Taxa de aprendizado do extrator: 0,00001
- Taxa de aprendizado do classificador: 0,0001
- Otimizador: Adam
- Função de perda: CrossEntropyLoss
- Máximo de épocas: 15
- Early stopping: paciência de 4 épocas

**Resultados do treinamento:**

| Época | Perda de treino | Perda de validação |
|---|---:|---:|
| 1 | 0,0784 | 0,3983 |
| 2 | 0,0671 | 0,3881 |
| 3 | 0,0530 | 0,3916 |
| 4 | 0,0510 | 0,4013 |
| 5 | 0,0434 | 0,3999 |
| 6 | 0,0407 | 0,4011 |

Melhor perda de validação: **0,3881**, na época 2.

**Resultados da avaliação:**

- Imagens de validação: 710
- Acurácia: 85,49%
- F1 macro: 0,8617
- F1 ponderado: 0,8542

**Comparação com os experimentos anteriores:**

O fine-tuning apresentou desempenho inferior ao modelo com data augmentation, que obteve acurácia de 86,20%, F1 macro de 0,8666 e F1 ponderado de 0,8618. O modelo com fine-tuning acertou 607 das 710 imagens, enquanto o modelo com augmentation acertou 612.

O fine-tuning apresentou resultados melhores em algumas categorias, como Food Organics, mas teve queda em outras, incluindo Glass, Metal, Miscellaneous Trash e Plastic.

**Conclusão:** com a configuração testada, o fine-tuning parcial não superou o modelo com data augmentation. A perda de treino continuou diminuindo, enquanto a perda de validação deixou de melhorar, indicando possível sobreajuste. O modelo com augmentation permanece como o melhor candidato até o momento.

**Checkpoint:** `models/mobilenet_v3_small_finetuned.pth`

**Status:** avaliação de validação concluída. O conjunto de teste permanece reservado para a avaliação final.

### Avaliação final — conjunto de teste

**Modelo selecionado:** MobileNetV3-Small com data augmentation.

**Conjunto de teste:** 719 imagens, reservado para a avaliação final.

**Resultados gerais:**

- Acurácia: 86,79%
- F1 macro: 0,8712
- F1 ponderado: 0,8677
- Acertos: 624 de 719
- Erros: 95 de 719

**Resultados por categoria — F1:**

| Categoria | F1 |
|---|---:|
| Cardboard | 0,8831 |
| Food Organics | 0,9120 |
| Glass | 0,8976 |
| Metal | 0,8621 |
| Miscellaneous Trash | 0,7871 |
| Paper | 0,8794 |
| Plastic | 0,8448 |
| Textile Trash | 0,8261 |
| Vegetation | 0,9481 |

**Conclusão:** o modelo apresentou 86,79% de acurácia no conjunto de teste, resultado próximo aos 86,20% observados na validação. Vegetation obteve o melhor F1, enquanto Miscellaneous Trash apresentou o menor. Os resultados indicam desempenho promissor, mas também mostram limitações em algumas categorias. O modelo deve ser entendido como um classificador visual de resíduos, e suas previsões não substituem orientações locais sobre descarte e reciclagem.
