import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Load the dataset
print("Loading dataset ...")
data = pd.read_csv("data2.csv")
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

# Define and train the Isolation Forest model
print("Creating and training the Isolation Forest model...")
isolation_forest_model = IsolationForest(contamination=0.05, random_state=42)  # Adjust contamination as needed
isolation_forest_model.fit(X_train_scaled)

# Make predictions on the validation set
print("Making predictions on the validation set...")
y_pred_val = isolation_forest_model.predict(X_val_scaled)
y_pred_val = [1 if pred == -1 else 0 for pred in y_pred_val]  # Convert -1 (anomaly) to 1, 0 for normal

# Calculate and print accuracy
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)

# Evaluate model performance on the validation set
print("Classification Report on Validation Set:")
class_report = classification_report(y_val, y_pred_val, target_names=label_encoder.classes_, zero_division=1)
print(class_report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred_val)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix as a heatmap
sns.set(font_scale=1.2)  # Adjust font size if needed
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot ROC Curve and calculate AUC
fpr, tpr, thresholds = roc_curve(y_val, y_pred_val)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print("Code execution completed!")
