import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the dataset
print("Loading dataset (20% sample)...")
data = pd.read_csv("0.01_2.csv")

# Separate features (X) and target variable (y)
X = data.drop("label", axis=1)
y = data["label"]

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)  # Include random_state for reproducibility

# Handling categorical variables
# Assuming 'protocol' is a categorical column, you can use one-hot encoding
X_train = pd.get_dummies(X_train, columns=['protocol'], drop_first=True)
X_val = pd.get_dummies(X_val, columns=['protocol'], drop_first=True)

# Create and fit the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save the trained scaler
scaler_filename = "scaler.joblib"
joblib.dump(scaler, scaler_filename)
print(f"Trained scaler saved as '{scaler_filename}'")

# Create and fit the LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Save the trained LabelEncoder
label_encoder_filename = "label_encoder.joblib"
joblib.dump(label_encoder, label_encoder_filename)
print(f"LabelEncoder saved as '{label_encoder_filename}'")

# Define and train the Random Forest model
print("Creating and training the Random Forest model...")
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y_train_encoded)

# Save the trained Random Forest model
model_filename = "random_forest_model.joblib"
joblib.dump(random_forest_model, model_filename)
print(f"Trained model saved as '{model_filename}'")

# Make predictions on the validation set
print("Making predictions on the validation set...")
y_pred_val = random_forest_model.predict(X_val_scaled)

# Decode the predictions using the LabelEncoder
y_pred_val_decoded = label_encoder.inverse_transform(y_pred_val)

# Evaluate model performance on the validation set
print("Evaluating model performance on the validation set...")
accuracy_val = accuracy_score(y_val, y_pred_val_decoded)
print("Validation Accuracy:", accuracy_val)

# Print the classification report
print("Classification Report on Validation Set:")
print(classification_report(y_val, y_pred_val_decoded, zero_division=1))

print("Code execution completed!")
