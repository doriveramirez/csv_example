####################################
# Ejemplo Clasificador Naive Bayes #
####################################

import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

###########################
# 1. Preparación de datos #
###########################

# Cargar el dataset (asegúrate de tener el archivo en la misma carpeta)
vote_data = pd.read_csv("house-votes-84.data", header=None, na_values='?')

# Renombrar columnas
header = ["NAME"] + [f"V{i}" for i in range(1, 17)]
vote_data.columns = header

# Mostrar estructura del dataset
print(vote_data.head())
print(vote_data.tail())

# Sustituir NaN por vacío o valor nulo manejable
vote_data = vote_data.fillna("")

# Convertir NAME a categórico
vote_data["NAME"] = vote_data["NAME"].astype("category")

# Distribución de clases
print("Distribución de clases:")
print(vote_data["NAME"].value_counts(normalize=True))

##############################################
# 2. Creación de datos de entrenamiento/test #
##############################################

vote_raw_train = vote_data.iloc[:370, :]
vote_raw_test  = vote_data.iloc[370:, :]

print("Proporciones en entrenamiento:")
print(vote_raw_train["NAME"].value_counts(normalize=True))
print("Proporciones en test:")
print(vote_raw_test["NAME"].value_counts(normalize=True))

##########################################
# 3. Creación de features para el modelo #
##########################################

# Convertir categorías ("y", "n", "") en números
encoder = LabelEncoder()

X_train = vote_raw_train.drop(columns=["NAME"]).apply(encoder.fit_transform)
y_train = vote_raw_train["NAME"].cat.codes

X_test = vote_raw_test.drop(columns=["NAME"]).apply(encoder.fit_transform)
y_test = vote_raw_test["NAME"].cat.codes

# Entrenar Naive Bayes categórico con suavizado Laplace
nb = CategoricalNB(alpha=1)
nb.fit(X_train, y_train)

# Predicción de clases
y_pred = nb.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=vote_data["NAME"].cat.categories)
disp.plot(cmap="Blues")
plt.show()

# Tabla de proporciones
cm_prop = cm / cm.sum(axis=0)
print("\nProporciones por clase:")
print(cm_prop)

# Filas donde se equivoca el modelo
misclassified = vote_raw_test.iloc[np.where(y_test != y_pred)]
print("\nPredicciones erróneas:")
print(misclassified)

##############################################
# 4. Predicción con probabilidades y curvas  #
##############################################

# Probabilidades de predicción
y_proba = nb.predict_proba(X_test)
pred_df = pd.DataFrame(y_proba, columns=vote_data["NAME"].cat.categories)
print("\nProbabilidades de predicción (primeras filas):")
print(pred_df.head())

# Calcular curva ROC, AUC
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Curva Precisión-Recall
precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
plt.figure()
plt.plot(recall, precision, lw=2, color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.show()

# Curva Sensitivity-Specificity
specificity = 1 - fpr
plt.figure()
plt.plot(tpr, specificity, lw=2, color='purple')
plt.xlabel('Sensibilidad (TPR)')
plt.ylabel('Especificidad (1-FPR)')
plt.title('Curva Sensitivity-Specificity')
plt.show()