from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

df = pd.read_csv("../data/raw/churned.csv")

# Splitting the data into training and test sets
X = df[['age', 'subscription_duration', 'last_purchase', 'average_monthly_usage', 'customer_support_calls']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save the trained model
model_filename = "../models/logistic_regression_model.pkl"
joblib.dump(model, model_filename)

print(accuracy, report)
