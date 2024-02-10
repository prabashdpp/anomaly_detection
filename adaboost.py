import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading dataset ...")
data = pd.read_csv("data.csv")
X = data.drop("label", axis=1)
y = data["label"]

# Convert string labels to numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)  # Stratified sampling

# Scale numerical features
print("Standard scaling numerical features (optional)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define and train the AdaBoost model
print("Creating and training the AdaBoost model...")
base_classifier = DecisionTreeClassifier(max_depth=1)  # Weak learner (stump)
adaboost_model = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)
adaboost_model.fit(X_train_scaled, y_train)

# Make predictions on the validation set
print("Making predictions on the validation set...")
y_pred_val = adaboost_model.predict(X_val_scaled)

# Evaluate model performance on the validation set
print("Evaluating model performance on the validation set...")
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)

# Print the classification report
print("Classification Report on Validation Set:")
print(classification_report(y_val, y_pred_val, zero_division=1))

# Calculate and display AUC-ROC
y_prob_val = adaboost_model.predict_proba(X_val_scaled)[:, 1]  # Probability of positive class
roc_auc = roc_auc_score(y_val, y_prob_val)
print("AUC-ROC Score:", roc_auc)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_prob_val)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Display confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("Code execution completed!")
