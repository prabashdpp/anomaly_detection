import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
print("Loading dataset (20% sample)...")
data = pd.read_csv("data.csv")
X = data.drop("label", axis=1)
y = data["label"]

# Convert string labels to integer labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)  # Stratified sampling

# Scale numerical features
print("Standard scaling numerical features (optional)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define and train the Gaussian Naive Bayes model
print("Creating and training the Gaussian Naive Bayes model...")
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_scaled, y_train)

# Make predictions on the validation set
print("Making predictions on the validation set...")
y_pred_val = naive_bayes_model.predict(X_val_scaled)

# Evaluate model performance on the validation set
print("Evaluating model performance on the validation set...")
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)

# Print the classification report
print("Classification Report on Validation Set:")
class_report = classification_report(y_val, y_pred_val, target_names=label_encoder.classes_, zero_division=1)
print(class_report)

print("Code execution completed!")
