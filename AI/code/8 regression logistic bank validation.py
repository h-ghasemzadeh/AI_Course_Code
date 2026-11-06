import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm

# Set plotting aesthetics
plt.rc("font", size=14)
sns.set(style="whitegrid", color_codes=True)

# Load the dataset
data = pd.read_csv("bank.csv", delimiter=',')  # Adjust delimiter if necessary

# Check the column names to identify the correct name for 'education'
print(data.columns)

# Update education values to group them together
data['education'] = data['education'].replace(['basic.9y', 'basic.6y', 'basic.4y'], 'Basic')

# Subscription stats
count_no_sub = len(data[data['y'] == 0])
count_sub = len(data[data['y'] == 1])
# print(count_no_sub,count_sub)
pct_of_no_sub = count_no_sub / (count_no_sub + count_sub) * 100
pct_of_sub = count_sub / (count_no_sub + count_sub) * 100
print("Percentage of no subscription:", pct_of_no_sub)
print("Percentage of subscription:", pct_of_sub)

# Convert 'y' column to binary (0 = no subscription, 1 = subscription)
# data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Visualize purchase frequency by job title
pd.crosstab(data.job, data.y).plot(kind='bar', title='Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.show()

# Encoding categorical variables
cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
data = pd.get_dummies(data, columns=cat_vars, drop_first=True)

# Define features and target
X = data.drop('y', axis=1)
y = data['y']

# Use SMOTE for oversampling
os = SMOTE(random_state=0, k_neighbors=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_os, y_train_os = os.fit_resample(X_train, y_train)

# Feature selection using RFE
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
n_features = min(10, X_train_os.shape[1])  # Select up to 10 features or fewer based on availability
rfe = RFE(logreg, n_features_to_select=n_features)
rfe.fit(X_train_os, y_train_os)
selected_features = X.columns[rfe.support_]
print("Selected features:", selected_features)

# Model training with selected features
X_train_os = X_train_os[selected_features]
X_test = X_test[selected_features]
logit_model = sm.Logit(y_train_os, X_train_os)
result = logit_model.fit()
print(result.summary2())

# Logistic Regression and evaluation
logreg.fit(X_train_os, y_train_os)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print(metrics.classification_report(y_test, y_pred))

# ROC curve
logit_roc_auc = metrics.roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = metrics.roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
