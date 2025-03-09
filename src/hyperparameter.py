import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
import mlflow 

# Load dataset
data = load_breast_cancer() 
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target') 

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

# Initialize model
rf = RandomForestClassifier()

# Define parameter grid (Fixed typo)
parameter_grid = {
    'n_estimators': [10, 50, 100],  # Corrected 'n_estimanors' to 'n_estimators'
    'max_depth': [None, 10, 20, 30]
}

# Apply GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=parameter_grid, cv=5, n_jobs=-1, verbose=2)

# Train the model
grid_search.fit(X_train, y_train)

# Get best parameters and score
best_parameter = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Parameters:", best_parameter)
print("Best Accuracy:", best_accuracy)
