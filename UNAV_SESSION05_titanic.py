import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Cargar datos desde GitHub
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

# 2. Seleccionar columnas relevantes y eliminar nulos en 'Embarked'
data = df[['Survived', 'Pclass', 'Sex', 'Embarked']].copy()
data = data.dropna(subset=['Embarked'])

# 3. Convertir variables categóricas a dummies
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# 4. Definir variables independientes y dependiente
y = data['Survived']
X = data.drop('Survived', axis=1)

# 5. Dividir en train y test (reproducible)
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)

# 6. Entrenar modelo de regresión logística con scikit-learn
logit_model = LogisticRegression(max_iter=1000, solver='lbfgs')
logit_model.fit(X_train, y_train)

# 7. Imprimir "pseudo summary"
print("=== Logistic Regression Results (scikit-learn) ===")
print("Intercept:", logit_model.intercept_)
print("Coefficients:", logit_model.coef_)
print("Feature Names:", X.columns.tolist())

# Evaluación en test
y_pred = logit_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Guardar modelo con joblib
joblib.dump(logit_model, "titanic.joblib")

# 9. Guardar dataset procesado en Excel
data.to_excel("titanic.xlsx", index=False)

print("✅ Modelo guardado como titanic.joblib")
print("✅ Datos guardados como titanic.xlsx")
