from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
import numpy as np
import time


# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Define base classifiers for stacking
base_classifiers = [
    ('decision_tree', DecisionTreeClassifier(random_state=42)),
    ('random_forest', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ('adaboost', AdaBoostClassifier(n_estimators=30, random_state=42, algorithm='SAMME')),
    ('gradient_boosting', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('xgboost', XGBClassifier(n_estimators=50, eval_metric='mlogloss', n_jobs=-1))
]

# Define the meta-model for stacking
meta_model = LogisticRegression(random_state=42)

# Create the stacking classifier
stacking_clf = StackingClassifier(estimators=base_classifiers, final_estimator=meta_model)

# Define hybrid ensemble
class HybridEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.adaboost = AdaBoostClassifier(n_estimators=30, random_state=42, algorithm='SAMME')
        self.stacking = stacking_clf
        self.meta_model = LogisticRegression(random_state=42)

    def fit(self, X, y):
        # Fit individual models
        self.rf.fit(X, y)
        self.adaboost.fit(X, y)
        self.stacking.fit(X, y)

        # Save class labels for scikit-learn compatibility
        self.classes_ = np.unique(y)

        # Combine predictions to train the meta-model
        rf_preds = self.rf.predict_proba(X)
        adaboost_preds = self.adaboost.predict_proba(X)
        stacking_preds = self.stacking.predict_proba(X)
        combined_preds = np.hstack((rf_preds, adaboost_preds, stacking_preds))
        self.meta_model.fit(combined_preds, y)
        return self

    def predict(self, X):
        # Get predictions from individual models
        rf_preds = self.rf.predict_proba(X)
        adaboost_preds = self.adaboost.predict_proba(X)
        stacking_preds = self.stacking.predict_proba(X)
        combined_preds = np.hstack((rf_preds, adaboost_preds, stacking_preds))
        # Use meta-model to make final prediction
        return self.meta_model.predict(combined_preds)

    def predict_proba(self, X):
        # Get predictions from individual models
        rf_preds = self.rf.predict_proba(X)
        adaboost_preds = self.adaboost.predict_proba(X)
        stacking_preds = self.stacking.predict_proba(X)
        combined_preds = np.hstack((rf_preds, adaboost_preds, stacking_preds))
        # Use meta-model to compute probabilities
        return self.meta_model.predict_proba(combined_preds)


# Define all classifiers including the hybrid ensemble
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=30, random_state=42, algorithm='SAMME'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=50, eval_metric='mlogloss', n_jobs=-1),
    "Stacking": stacking_clf,
    "Hybrid Ensemble": HybridEnsemble()  # Add hybrid ensemble
}

# Evaluate each classifier using cross-validation and measure time
print("Performance of Ensemble Methods:")
for name, clf in classifiers.items():
    start_time = time.time()
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')  # Reduced to 3-fold CV
    elapsed_time = time.time() - start_time
    print(f"{name} Accuracy: {scores.mean():.2f} ± {scores.std():.2f}, Time: {elapsed_time:.2f} seconds")

