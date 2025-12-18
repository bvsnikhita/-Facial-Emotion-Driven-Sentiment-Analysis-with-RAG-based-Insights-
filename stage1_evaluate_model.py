import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

# --------------------- LOAD TEST DATA ---------------------
test_path = r"C:/Users/NIKHITA/OneDrive/Desktop/code scribed project/archive/test"

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    test_path, target_size=(48,48),
    color_mode='grayscale', class_mode='categorical',
    batch_size=64, shuffle=False
)

# --------------------- LOAD MODEL ---------------------
model = tf.keras.models.load_model("emotion_model_partial.h5")
print("✅ Model loaded successfully!")

# --------------------- EVALUATE MODEL ---------------------
loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# --------------------- PREDICTIONS ---------------------
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

# --------------------- CLASSIFICATION REPORT ---------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# --------------------- CONFUSION MATRIX ---------------------
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# --------------------- SAVE PREDICTIONS ---------------------
emotion_map = {v:k for k,v in test_data.class_indices.items()}
predicted_emotions = [emotion_map[i] for i in y_pred_classes]
filenames = test_data.filenames
pd.DataFrame({
    "image": filenames,
    "predicted_emotion": predicted_emotions
}).to_csv("predicted_emotions.csv", index=False)

print("\n✅ Predictions saved to predicted_emotions.csv")
