import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# CONFIGURATION
CSV_PATH = "pose_landmarks.csv"
MODEL_SAVE_PATH = "yoga_pose_model.pkl"
ENCODER_SAVE_PATH = "pose_label_encoder.pkl"

# 1. Load the dataset
df = pd.read_csv(CSV_PATH)

# 2. Split features and labels
X = df.drop(columns=["label"])
y = df["label"]

# 3. Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 5. Train an SVM classifier
model = SVC(kernel="rbf", probability=True, class_weight="balanced")
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("\n✅ Model Training Complete!")
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 7. Save the model and encoder
with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(model, f)

with open(ENCODER_SAVE_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
print(f"✅ Label encoder saved to: {ENCODER_SAVE_PATH}")
