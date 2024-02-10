import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the test dataset
print("Loading test dataset...")
test_data = pd.read_csv("test_data2.csv")

# Check if 'label' column exists in the test data
if 'label' in test_data.columns:
    # Extract the 'label' column for later comparison
    true_labels = test_data['label']

    # Drop the 'label' column
    test_data = test_data.drop(['label'], axis=1)
else:
    # If 'label' column does not exist, create a placeholder for true labels
    true_labels = pd.Series(['Unknown'] * len(test_data))

# Load the scaler and label encoder used during training
print("Loading scaler and label encoder used during training...")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Scale numerical features
print("Scaling numerical features...")
X_test_scaled = scaler.transform(test_data)

# Load the trained Random Forest model from the saved file
print("Loading trained Random Forest model...")
model_filename = "random_forest_model.joblib"
trained_model = joblib.load(model_filename)

# Make predictions on the scaled test data
print("Making predictions on the test data...")
predictions = trained_model.predict(X_test_scaled)

# Decode predictions using the label encoder
decoded_predictions = label_encoder.inverse_transform(predictions)

# Create a DataFrame to compare true labels with predicted labels
results_df = pd.DataFrame({'True Labels': true_labels, 'Predicted Labels': decoded_predictions})

# Output the DataFrame to inspect the results
print("Test Data Predictions:")
print(results_df)
