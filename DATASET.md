# Dataset

## RealWaste

O WasteWise utiliza o dataset RealWaste, disponibilizado pelo UCI Machine Learning Repository.

- **Fonte:** [UCI Machine Learning Repository — RealWaste](https://archive-beta.ics.uci.edu/dataset/908/realwaste)
- **Artigo:** [RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning](https://www.mdpi.com/2078-2489/14/12/633)
- **Autores:** Sam Single, Saeid Iranmanesh e Raad Raad
- **Ano:** 2023
- **Licença informada na documentação original:** CC BY-NC-SA 4.0
- **Total de imagens:** 4.752
- **Número de categorias:** 9

## Categorias originais

| Categoria | Quantidade de imagens |
|---|---:|
| Cardboard | 461 |
| Food Organics | 411 |
| Glass | 420 |
| Metal | 790 |
| Miscellaneous Trash | 495 |
| Paper | 500 |
| Plastic | 921 |
| Textile Trash | 318 |
| Vegetation | 436 |
| **Total** | **4.752** |

## Preparação e validação dos dados

As quantidades apresentadas correspondem à contagem de arquivos realizada localmente nas pastas do dataset. Durante a preparação dos dados, as imagens foram verificadas quanto à integridade e os arquivos duplicados exatos foram investigados por meio de hashes SHA-256.

O conjunto foi dividido de forma estratificada, utilizando a semente aleatória 42, em três subconjuntos:

- **Treinamento:** 3.323 imagens (70%).
- **Validação:** 710 imagens (15%).
- **Teste:** 719 imagens (15%).

A divisão busca preservar aproximadamente a proporção das categorias em cada subconjunto. O conjunto de validação é utilizado para acompanhar o treinamento e comparar experimentos. O conjunto de teste permanece reservado para a avaliação final do modelo selecionado.

## Licença e atribuição

A documentação original do dataset indica a licença **CC BY-NC-SA 4.0 (Creative Commons Atribuição-NãoComercial-CompartilhaIgual 4.0)**.

A utilização e a redistribuição das imagens devem respeitar os termos dessa licença, incluindo:

- **Atribuição:** dar os créditos apropriados aos autores, fornecer a referência da licença e indicar modificações realizadas, quando aplicável.
- **Uso não comercial:** não utilizar o material licenciado para fins comerciais nos termos proibidos pela licença.
- **Compartilhamento pela mesma licença:** respeitar as condições aplicáveis à redistribuição de material adaptado.

A licença das imagens do dataset não determina automaticamente a licença do código-fonte desenvolvido para o WasteWise.

## Limitações

A classificação produzida pelo WasteWise representa uma previsão visual da categoria do resíduo. Ela não garante, por si só, que um objeto seja reciclável nem substitui as orientações locais de coleta seletiva e descarte adequado.

Os resultados do modelo também dependem das características do dataset utilizado e podem não representar todos os resíduos, condições de iluminação, ângulos ou ambientes encontrados no mundo real.