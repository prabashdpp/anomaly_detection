import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import classification_report

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Split features and target variable
    X = df.drop('label', axis=1)  # Replace 'label' with your actual column name
    y = df['label']

    # Identify numerical and categorical features
    numerical_features = [i for i in range(X.shape[1]) if df.dtypes[i] not in ['object', 'category']]
    categorical_features = [i for i in range(X.shape[1]) if i not in numerical_features]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.iloc[:, numerical_features])
    X_val_scaled = scaler.transform(X_val.iloc[:, numerical_features])

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
    X_train_encoded = encoder.fit_transform(X_train.iloc[:, categorical_features])
    X_val_encoded = encoder.transform(X_val.iloc[:, categorical_features])

    # Combine features
    X_train_combined = np.concatenate([X_train_scaled, X_train_encoded.toarray()], axis=1)
    X_val_combined = np.concatenate([X_val_scaled, X_val_encoded.toarray()], axis=1)

    return X_train_combined, X_val_combined, y_train, y_val

# Function to create and train MLP model
def create_and_train_mlp_model(X_train, y_train, X_val, y_val):
    # Define your MLP model (hidden layers, activation, solver, max_iter)
    mlp_model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=100, random_state=42)

    # Train the model
    mlp_model.fit(X_train, y_train)

    # Evaluate on validation set
    accuracy, report = evaluate_model(mlp_model, X_val, y_val)
    print(f"MLP Validation Accuracy: {accuracy}")
    print(report)

    return mlp_model

# Function to evaluate model performance
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    
    # Convert y_pred to binary values based on a threshold
    threshold = 0.5
    y_pred_binary = (y_pred.flatten() > threshold).astype(int)
    
    # Convert y_val to binary values
    y_val_binary = y_val.astype(int)

    # Calculate accuracy and classification report
    accuracy = np.mean(y_val_binary == y_pred_binary)
    report = classification_report(y_val_binary, y_pred_binary, zero_division=1)

    return accuracy, report

# Load data and preprocess
X_train, X_val, y_train, y_val = load_and_preprocess_data('data.csv')

# Create and train MLP model
mlp_model = create_and_train_mlp_model(X_train, y_train, X_val, y_val)
