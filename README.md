# Classificação de Doenças em Plantas com Deep Learning

**Aluno:** Thomaz Ritter
**Disciplina:** Deep Learning — Desafio Grau B

## Estrutura

| Arquivo | Descrição |
|---------|-----------|
| [`relatorio_desafio.md`](relatorio_desafio.md) | Relatório completo com análise e discussão |
| [`desafio_doencas_plantas.ipynb`](desafio_doencas_plantas.ipynb) | Notebook com código e experimentos |
| [`imagens/`](imagens/) | Gráficos gerados pelos experimentos |

## Como executar

1. Abra o notebook no [Google Colab](https://colab.research.google.com/)
2. Ative a GPU: `Runtime > Change runtime type > GPU`
3. Execute todas as células: `Runtime > Run all`

O treinamento leva aproximadamente 40 minutos com GPU T4 no Colab.

## Resultados

| Experimento | Acc. Validação |
|-------------|---------------|
| ResNet18 + Aug. Leve | **99.89%** |
| ResNet50 + Aug. Leve | 99.83% |
| ResNet18 + Aug. Agressiva | 99.74% |
