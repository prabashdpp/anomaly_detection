import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
print("Loading dataset (20% sample)...")
data = pd.read_csv("data.csv")
X = data.drop("label", axis=1)
y = data["label"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)  # Stratified sampling

# Scale numerical features
print("Standard scaling numerical features (optional)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define and train the Decision Tree model
print("Creating and training the Decision Tree model...")
decision_tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)  # You can adjust max_depth as needed
decision_tree_model.fit(X_train_scaled, y_train)

# Make predictions on the validation set
print("Making predictions on the validation set...")
y_pred_val = decision_tree_model.predict(X_val_scaled)

# Evaluate model performance on the validation set
print("Evaluating model performance on the validation set...")
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)

# Print the classification report
print("Classification Report on Validation Set:")
print(classification_report(y_val, y_pred_val, zero_division=1))

print("Code execution completed!")
