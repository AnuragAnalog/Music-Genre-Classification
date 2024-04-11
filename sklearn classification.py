import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load the 3 sec data features
features3 = pd.read_csv("./Data/features_3_sec.csv")
features3.head()

# Seperating features and labels

X = features3.iloc[:, 2:-1]
y = features3["label"]

# Split the data into features and target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

# Encode the target
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test the model
X_test = scaler.transform(X_test)
y_test = label_encoder.transform(y_test)

y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
cr = classification_report(y_test, y_pred)
print(cr)

# Optimize the model using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)