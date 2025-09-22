import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import joblib

# 1. Load Titanic data from GitHub
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

# 2. Select relevant columns and handle missing values
data = df[['Survived', 'Pclass', 'Sex', 'Embarked']].copy()
data = data.dropna(subset=['Embarked'])

# 3. Convert categorical variables into dummy variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# 4. Define target and features
y = data['Survived']
X = data.drop('Survived', axis=1)
X = sm.add_constant(X)

# 5. Train-test split (reproducible)
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)

# 6. Fit logistic regression model
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit(disp=False)

# >>> Print the model summary
print(result.summary())

# 7. Save model with joblib
joblib.dump(result, "titanic.joblib")

# 8. Save processed dataset to Excel
data.to_excel("titanic.xlsx", index=False)

print("✅ Model saved as titanic.joblib")
print("✅ Data saved as titanic.xlsx")
