from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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
    ('xgboost', XGBClassifier(n_estimators=50, eval_metric='mlogloss', n_jobs=-1))  # Removed use_label_encoder
]

# Define the meta-model for stacking
meta_model = LogisticRegression(random_state=42)

# Create the stacking classifier
stacking_clf = StackingClassifier(estimators=base_classifiers, final_estimator=meta_model)

# Define all classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=30, random_state=42, algorithm='SAMME'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=50, eval_metric='mlogloss', n_jobs=-1),  # Removed use_label_encoder
    "Stacking": stacking_clf  # Add stacking classifier
}

# Evaluate each classifier using cross-validation and measure time
print("Performance of Ensemble Methods:")
for name, clf in classifiers.items():
    start_time = time.time()
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')  # Reduced to 3-fold CV
    elapsed_time = time.time() - start_time
    print(f"{name} Accuracy: {scores.mean():.2f} ± {scores.std():.2f}, Time: {elapsed_time:.2f} seconds")
