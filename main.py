import pandas as pd
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# ── 1. LOAD LABELS ──────────────────────────────────────────────
df = pd.read_csv("train.csv")
print(df.head())
print(df.shape)

# ── 2. EXTRACT HOG FEATURES FROM ALL TRAINING IMAGES ────────────
# 8x8 cells proved better than 4x4 for this dataset
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

# ── 3. SCALE FEATURES & FIND BEST SVM SETTINGS ──────────────────
X = np.array(features_list)
y = np.array(labels_list)

# normalize all HOG values to the same scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# focused grid search around our best known values
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=5000, random_state=42, stratify=y)
param_grid = {"C": [5, 10, 20, 50], "gamma": [0.0005, 0.001, 0.005, 0.01]}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=3, verbose=2)
grid.fit(X_sample, y_sample)
print("─" * 40)
print("Best C:", grid.best_params_["C"])
print("Best gamma:", grid.best_params_["gamma"])
print("Best score on sample:", round(grid.best_score_, 4))
print("─" * 40)

# ── 4. TRAIN FINAL SVM ON ALL DATA WITH BEST SETTINGS ───────────
clf = SVC(kernel="rbf", C=grid.best_params_["C"], gamma=grid.best_params_["gamma"])
clf.fit(X, y)
print("Training done!")

# ── 5. PREDICT ON COMPETITION TEST IMAGES ────────────────────────
test_df = pd.read_csv("test.csv")

test_features_list = []

for index, row in test_df.iterrows():
    img_id = row["Id"]
    path = f"test/test/{img_id}.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)
    test_features_list.append(features)

# scale test features using the same scaler as training
X_test_final = scaler.transform(np.array(test_features_list))
predictions = clf.predict(X_test_final)

# save predictions to submission.csv
submission = pd.DataFrame({"Id": test_df["Id"], "Category": predictions})
submission.to_csv("submission.csv", index=False)
print("Submission saved!")