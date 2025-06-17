import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Load the dataset
df = pd.read_csv('Train.txt', delimiter=',', header=None)

# Assign column names based on KDD Cup 1999 dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'attack', 'last_flag'
]
df.columns = column_names
 
# Drop unnecessary columns
df = df.drop(['land', 'num_outbound_cmds', 'urgent', 'num_failed_logins'], axis=1)

# Encode categorical columns using separate LabelEncoders
categorical_columns = ['protocol_type', 'service', 'flag']
label_encoders = {}

for column in categorical_columns + ['attack']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Save the label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Prepare features and labels
X = df.drop(columns=['attack'])
y = df['attack']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Encode labels as categorical
num_classes = len(np.unique(y))
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# Define the neural network model
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),  # Input layer
    layers.Dense(128, activation='relu'),            # Hidden layer
    layers.Dropout(0.3),                             # Dropout layer
    layers.Dense(64, activation='relu'),             # Hidden layer
    layers.Dropout(0.2),                             # Dropout layer
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Define EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train_categorical,
                    epochs=50,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stopping])

# Plot the accuracy graph
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
epochs = range(1, len(training_accuracy) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, training_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('accuracy_graph.png')
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_categorical)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save('intrusion_detection_model.h5')
print("Model, scaler, and label encoders have been saved successfully.")

# Predict on test data
y_pred_proba = model.predict(X_test_scaled)  # Get probabilities
y_pred = np.argmax(y_pred_proba, axis=1)     # Get class predictions
y_true = y_test.values                       # True labels

# Classification report (includes F1 score)
print("Classification Report:")
# Retrieve the unique labels from the dataset
class_labels = np.unique(y_true)

# Get the corresponding class names for the labels
class_names = [label_encoders['attack'].inverse_transform([label])[0] for label in class_labels]

# Generate the classification report
print(classification_report(y_true, y_pred, labels=class_labels, target_names=class_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Display confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['attack'].classes_, yticklabels=label_encoders['attack'].classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.savefig('confusion_matrix_heatmap.png')
plt.show()

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_categorical[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curves
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()