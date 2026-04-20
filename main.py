import pandas as pd
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC

# ── 1. LOAD LABELS ──────────────────────────────────────────────
df = pd.read_csv("train.csv")
print(df.head())
print(df.shape)

# ── 2. EXTRACT HOG FEATURES FROM ALL TRAINING IMAGES ────────────
features_list = []
labels_list = []

for index, row in df.iterrows():
    img_id = row["Id"]
    label = row["Category"]
    path = f"train/train/{label}/{img_id}.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)
    features_list.append(features)
    labels_list.append(label)

print("Done! Total images processed:", len(features_list))

# ── 3. TRAIN THE SVM ON ALL TRAINING DATA ───────────────────────
# We validated 99.5% accuracy earlier with an 80/20 split
# Now we train on the full dataset for the best possible predictions
X = np.array(features_list)
y = np.array(labels_list)

clf = SVC(kernel="rbf")
clf.fit(X, y)
print("Training done!")

# ── 4. PREDICT ON COMPETITION TEST IMAGES ────────────────────────
test_df = pd.read_csv("test.csv")

test_features_list = []

for index, row in test_df.iterrows():
    img_id = row["Id"]
    path = f"test/test/{img_id}.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)
    test_features_list.append(features)

X_test_final = np.array(test_features_list)
predictions = clf.predict(X_test_final)

# save predictions to submission.csv
submission = pd.DataFrame({"Id": test_df["Id"], "Category": predictions})
submission.to_csv("submission.csv", index=False)
print("Submission saved!")