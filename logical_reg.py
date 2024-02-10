import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning

# Load the dataset
print("Loading dataset ...")
data = pd.read_csv("data2.csv")
X = data.drop("label", axis=1)
y = data["label"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)  # Stratified sampling
X_train_red, _, y_train_red, _ = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)  # Further reduce for speed

# Scale numerical features
print("Standard scaling numerical features (optional)...")
scaler = StandardScaler()
X_train_red = scaler.fit_transform(X_train_red)
X_val = scaler.transform(X_val)

# Define and train the model
print("Creating and training the model... (C=1 for faster training)")
model = LogisticRegression(C=1, random_state=42, max_iter=1000)  # Increase max_iter

# Suppress the specific warning for precision being ill-defined
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    model.fit(X_train_red, y_train_red)

# Make predictions on the validation set (optional)
print("Making predictions on the validation set (optional)...")
y_pred_val = model.predict(X_val)

# Evaluate model performance (optional)
print("Evaluating model performance on the validation set (optional)...")
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)

# Print the classification report (optional)
print("Classification Report:")
# Set zero_division=1 for precision, recall, and f1-score
print(classification_report(y_val, y_pred_val, zero_division=1))

print("Code execution completed (targeting 5 minutes)!")
