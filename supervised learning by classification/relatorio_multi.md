# Relatório dos Resultados – Classificação Multiclasse

## 1. Contexto

- **Dataset:** 297 amostras, 20 atributos, 5 classes (target = 0 a 4).  
- **Problema:** Classificação multiclasse desbalanceada (classe 0 domina com 49/90 amostras no teste).  
- **Objetivo:** Comparar 5 modelos supervisionados (Regressão Logística, KNN, SVM-RBF, Random Forest, XGBoost) usando métricas adequadas para multiclasse.

## 2. Métricas Avaliadas

- **Acurácia:** Proporção de previsões corretas.  
- **Precisão (Precision):** Fração de previsões corretas dentro de cada classe.  
- **F1-score:** Média harmônica entre precisão e recall, equilibrando erros de falsos positivos e falsos negativos.  
- **Average used:** `macro`, para ponderar as classes de acordo com seu tamanho.

## 3. Resultados Agregados

| Modelo               | Acurácia | Precisão (macro) | F1-score (macro) |
|---------------------|----------|-------------------|-------------------|
| Regressão Logística  | 0.5667   | 0.2563            | 0.2456            |
| KNN                  | 0.5778   | 0.2906            | 0.2777            |
| SVM-RBF              | 0.6000   | 0.2548            | 0.2605            |
| Random Forest        | 0.5778   | 0.2714            | 0.2641            |
| XGBoost              | 0.5222   | 0.2483            | 0.2426            |

> **Observação:** A acurácia é relativamente alta para a classe majoritária, mas a precisão e F1-weighted mostram que os modelos não estão performando bem nas classes minoritárias.

## 4. Observações Detalhadas

### Desbalanceamento das classes

- Classe 0 domina, enquanto a classe 4 possui apenas 3 amostras no conjunto de teste.  
- Isso causa problemas na predição: muitas classes minoritárias não são previstas (precision e recall = 0).

### Comparação entre modelos

- **SVM-RBF** obteve a maior acurácia (0.60), mas ainda apresenta baixo F1 e precisão ponderada.  
- **KNN** e **Random Forest** têm F1-score e precisão ponderada ligeiramente melhores, sugerindo maior equilíbrio entre classes.  
- **XGBoost** teve o pior desempenho geral neste setup.

### Classes difíceis

- Classes 3 e 4 apresentam recall zero em quase todos os modelos.  
- Isto indica que os modelos não conseguem identificar corretamente as classes raras.