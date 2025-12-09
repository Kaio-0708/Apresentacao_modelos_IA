import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score, 
    f1_score
)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("D:\\IA_UFCG\\heart_disease_final.csv")
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30, random_state=42)
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelos = {
    "Regressão Logística": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM-RBF": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

usar_scaled = ["Regressão Logística", "KNN", "SVM-RBF"]
resultados = {}
f1s = {}
precisoes = {}
for nome, modelo in modelos.items():

    print(f"Treinando modelo: {nome}")

    if nome in usar_scaled:
        modelo.fit(X_train_scaled, y_train)
        pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    prec = precision_score(y_test, pred, average='macro')
    #Usei o average macro devido o dataset está desbalanceado, assim é o mais indicado para avaliar o desempenho do modelo em todas as classes de forma mais próxima de ser igualitária. 
    resultados[nome] = acc
    precisoes[nome] = prec
    f1s[nome] = f1

    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão (macro): {prec:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print("Relatório de Classificação:")
    print(classification_report(y_test, pred, zero_division=0))

print("Gráfico de comparação de acurácia")
plt.figure(figsize=(10,5))
plt.bar(resultados.keys(), resultados.values())
plt.title("Comparação de Acurácia")
plt.ylabel("Acurácia")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

print("Gráfico de precisão")
plt.figure(figsize=(10,5))
plt.bar(precisoes.keys(), precisoes.values(), color='orange')
plt.title("Comparação de Precisão")
plt.ylabel("Precisão")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

print("Gráfico de F1-score")
plt.figure(figsize=(10,5))
plt.bar(f1s.keys(), f1s.values(), color='green')
plt.title("Comparação de F1-score")
plt.ylabel("F1-score")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

print("\nComparação final:")
for nome in resultados:
    print(f"{nome}: Acurácia={resultados[nome]:.4f}, Precisão={precisoes[nome]:.4f}, F1-score={f1s[nome]:.4f}")