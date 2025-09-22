import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Cargar datos desde GitHub
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

# 2. Seleccionar únicamente las variables numéricas relevantes
data = df[['Survived', 'Pclass']].copy()

# 3. Definir variables independiente (X) y dependiente (y)
y = data['Survived']
X = data[['Pclass']]   # solo la variable numérica Pclass

# 4. Dividir en train y test (reproducible)
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)

# 5. Entrenar modelo de regresión logística con scikit-learn
logit_model = LogisticRegression(max_iter=1000, solver='lbfgs')
logit_model.fit(X_train, y_train)

# 6. Imprimir resultados
print("=== Logistic Regression Results (only numeric features) ===")
print("Intercept:", logit_model.intercept_)
print("Coefficient for Pclass:", logit_model.coef_[0][0])

# Evaluación en test
y_pred = logit_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Guardar modelo con joblib
joblib.dump(logit_model, "titanic.joblib")

# 8. Guardar dataset procesado en Excel
data.to_excel("titanic.xlsx", index=False)

print("✅ Modelo guardado como titanic.joblib")
print("✅ Datos guardados como titanic.xlsx")
