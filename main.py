import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import beta
from scipy.spatial.distance import cdist
import time

# Load and preprocess MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = X / 255.0  # Normalize pixel values to [0, 1]

# Split into training (60,000) and testing (10,000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)

# 10-fold cross-validation setup
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# ---- SVM ----
# ---- SVM ----
print("\n=== SVM Classifiers ===")
for kernel, degree in [('linear', None), ('poly', 2), ('rbf', None)]:
    if kernel == 'poly':
        svm_model = SVC(kernel=kernel, degree=degree, random_state=42)
    else:
        svm_model = SVC(kernel=kernel, random_state=42)

    start = time.time()
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=cv, n_jobs=-1)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    end = time.time()

    print(f"SVM ({kernel} kernel):")
    print(f"  CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Training + Prediction Time: {end - start:.2f} seconds\n")


# ---- Random Forest ----
print("\n=== Random Forest Classifier ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
start = time.time()
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
end = time.time()
print(f"Random Forest:")
print(f"  CV Accuracy: {np.mean(rf_cv_scores):.4f} ± {np.std(rf_cv_scores):.4f}")
print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  Training + Prediction Time: {end - start:.2f} seconds\n")

# ---- Naïve Bayes (Beta Distribution) ----
class NaiveBayesBeta:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.class_priors = None

    def fit(self, X, y):
        n_classes = np.unique(y).size
        n_features = X.shape[1]

        self.alpha = np.zeros((n_classes, n_features))
        self.beta = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)

        for c in range(n_classes):
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / X.shape[0]
            mean = X_c.mean(axis=0)
            var = X_c.var(axis=0)
            self.alpha[c] = mean * ((mean * (1 - mean)) / var - 1)
            self.beta[c] = (1 - mean) * ((mean * (1 - mean)) / var - 1)

    def predict(self, X):
        log_probs = []
        for c in range(len(self.class_priors)):
            likelihood = np.sum(beta.logpdf(X, self.alpha[c], self.beta[c]), axis=1)
            log_prob = np.log(self.class_priors[c]) + likelihood
            log_probs.append(log_prob)
        return np.argmax(np.column_stack(log_probs), axis=1)

print("\n=== Naive Bayes (Beta Distribution) ===")
nb_model = NaiveBayesBeta()
start = time.time()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
end = time.time()
print(f"Naive Bayes (Beta Distribution):")
print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"  Training + Prediction Time: {end - start:.2f} seconds\n")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

# ---- k-NN ----
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = cdist(X, self.X_train, metric='euclidean')
        neighbors = np.argsort(distances, axis=1)[:, :self.k]
        neighbor_classes = self.y_train[neighbors]
        return np.array([np.bincount(row).argmax() for row in neighbor_classes])

print("\n=== k-NN Classifier ===")
knn_model = KNNClassifier(k=5)
start = time.time()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
end = time.time()
print(f"k-NN (k=5):")
print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"  Training + Prediction Time: {end - start:.2f} seconds\n")
