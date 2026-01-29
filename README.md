# Inteligência Artificial aplicada a Sistemas Físico-Cibernéticos

Projeto desenvolvido para a disciplina **Inteligência Artificial aplicada a Sistemas Físico-Cibernéticos** do curso de **Engenharia Elétrica** na **UFCG**.

## Sobre o Projeto

Este repositório contém o desenvolvimento de um projeto prático aplicado à disciplina mencionada, com foco na utilização de **Inteligência Artificial** em **Sistemas Físico-Cibernéticos (CPS)**. O objetivo do projeto é explorar técnicas de IA para análise, previsão e controle em sistemas que integram componentes físicos e computacionais.
Estou aplicando esta cadeira pois é uma área pela qual sou apaixonado e na qual busco constantemente expandir meu conhecimento.

## Dataset

https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease

## Fluxo do Projeto

1. **Exploração de Dados (EDA)**
   - Leitura e limpeza do dataset (`processed.cleveland.data`).
   - Tratamento de valores ausentes e duplicados.
   - Análise univariada, bivariada e multivariada.
   - Visualizações com **matplotlib** e **seaborn** (histogramas, boxplots, scatter plots, heatmaps).
   - Detecção de outliers usando Z-score e 1.5*IQR.

2. **Extração, Transformação e Limpeza (ETL)**
   - Criação de variáveis derivadas:
     - `oldpeak_log` (log-transformação da depressão ST).
     - Variáveis dummy para `cp` (tipo de dor no peito).
     - Codificação de variáveis binárias (`sex_male`, `exang_yes`).
   - Geração do dataset final `heart_disease_final.csv`.

3. **Modelagem e Treinamento**
   - Divisão em **treino e teste** (70/30).
   - Escalonamento de dados com `StandardScaler`.
   - Treinamento de modelos:
     - Regressão Logística
     - KNN
     - SVM-RBF
     - Random Forest
     - XGBoost
   - Avaliação com métricas:
     - **Acurácia**, **Precisão**, **F1-score**, **ROC AUC**
   - Comparação visual de desempenho com gráficos de barras e curvas ROC.

4. **Interface Interativa (Streamlit)**
   - Inputs clínicos do paciente: idade, sexo, pressão, colesterol, ECG, etc.
   - Transformações e dummy encoding replicando o pré-processamento do dataset.
   - Escalonamento com o `StandardScaler` treinado.
   - Predição usando todos os modelos treinados.
   - Exibição de:
     - Probabilidade de doença cardíaca
     - Classificação de risco (Alto/Baixo)
     - Recomendações médicas ou preventivas

## Tecnologias Utilizadas

- Python 3.10.11
- **pandas** – manipulação de dados
- **numpy** – cálculos numéricos
- **matplotlib** – gráficos e visualizações
- **seaborn** – gráficos estatísticos
- **scipy** – funções estatísticas e detecção de outliers
- **scikit-learn** – modelagem e machine learning
- **xgboost** – Gradient Boosting
- **joblib** – serialização de modelos
- **streamlit** – interface web interativa

## Streamlit Cloud: 
https://apresentacaomodelosia-2gxdaayuxhzlz9dvjseh8a.streamlit.app/

## Links úteis
https://streamlit.io/cloud
https://docs.streamlit.io/
https://joblib.readthedocs.io/en/stable/

## Autor

Kaio Vitor - [GitHub](https://github.com/Kaio-0708)
