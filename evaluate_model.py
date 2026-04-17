import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

# CONFIG

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
THRESHOLD = 0.45
DATASET_DIR = "/Users/mac/Desktop/VISIBILITY_PROJECT/dataset"
MODEL_PATH = "fog_no_fog_model_FINAL.keras"


model = tf.keras.models.load_model(MODEL_PATH)


# LOAD TEST DATA 
test_gen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


# EVALUATION
test_loss, test_acc = model.evaluate(test_data, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

y_true = test_data.classes
y_prob = model.predict(test_data, verbose=0)
y_pred = (y_prob > THRESHOLD).astype(int).ravel()


# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix\n", cm)
print("\nClassification Report\n",
      classification_report(y_true, y_pred, target_names=["fog", "no_fog"]))

plt.figure()
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["fog", "no_fog"],
    yticklabels=["fog", "no_fog"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()


