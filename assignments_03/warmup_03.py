import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

output_path = "./outputs"
os.makedirs(output_path, exist_ok=True)

# Preprocessing

# --- Preprocessing Question 1 ---
# Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42. Print the shapes of all four arrays.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4, stratify = y)
# stratify maintains class proportions

print(f"Shape of X : X.train shape ({X_train.shape}), X.test shape ({X_test.shape})")
print(f"Shape of y : y.train shape ({y_train.shape}), y.test shape ({y_test.shape})")

# --- Preprocessing Question 2 ---
# Fit a StandardScaler on X_train and use it to transform both X_train and X_test. Print the mean of each column in X_train_scaled -- they should all be very close to 0. Add a comment explaining in one sentence why you fit the scaler on X_train only.
scaler = StandardScaler()
# Only using fit_transform on training data to prevent data from test leaking into model during training
X_train_scaled = scaler.fit_transform(X_train)
# Using only transform applies same scaling as we did on train.
X_test_scaled = scaler.transform(X_test)

print(f"Mean of X_train_scaled : {X_train_scaled.mean(axis = 0)}")

# KNN

# --- KNN Question 1 ---
# Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train), and predict on the test set. Print the accuracy score and the full classification report.
knn = KNeighborsClassifier(n_neighbors = 5)

# fits on unscaled trainings data
knn.fit(X_train, y_train)

# predicts on test set
y_predict = knn.predict(X_test)

print(f"Accuracy Score : {accuracy_score(y_test, y_predict)}")
print(f"Classification report :\n {classification_report(y_test, y_predict)}")

# --- KNN Question 2 ---
# Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). Print the accuracy score. Add a comment: does scaling improve performance, hurt it, or make no difference? Why might that be for this particular dataset?

# Created new model so I could call previous knn if needed
knn_scaled = KNeighborsClassifier(n_neighbors = 5)

# fits to scaled model
knn_scaled.fit(X_train_scaled, y_train)

# predicts on scaled set
y_predict_scaled = knn_scaled.predict(X_test_scaled)

print(f"Accuracy Score : {accuracy_score(y_test, y_predict_scaled)}")
print(f"Classification report :\n {classification_report(y_test, y_predict_scaled)}")

# The accuracy score did not differ between the trained and untrained data.  This suggests that the features were already similar scales, where not one single feature dominated.


# --- KNN Question 3 ---
# Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data. Print each fold score, the mean, and the standard deviation. Add a comment: is this result more or less trustworthy than a single train/test split, and why?
knn = KNeighborsClassifier(n_neighbors = 5)

# Cross validation on training data
cv_scores = cross_val_score(knn, X_train, y_train, cv = 5)

# Print each fold score, mean and std
print(f"Fold scores : {cv_scores}")
print(f"Mean CV score: {cv_scores.mean()}")
print(f"Standard deviation: {cv_scores.std()}")

# The results are more trustworthy than a single train/test split because cross-validation uses multiple splits and then evaluating them.  This means it gives a more reliable estimate on how the model will generalize unseen data and reduces the dependence on a particular split.

# --- KNN Question 4 ---
# Loop over k values [1, 3, 5, 7, 9, 11, 13, 15]. For each, compute 5-fold cross-validation accuracy on the unscaled training data and print k and the mean CV score. Add a comment identifying which k you would choose and why.
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv = 5)
    print(f"n_neighbors : {k}")
    print(f"Mean CV score : {cv_scores.mean()}")
    
# I would choose k = 9 with a mean score of 0.9833.  Since k = 7 and k = 9 have the same mean score, I would want the larger k because it is less sensitive to noise and generalize better.

# Classifier Evaluation

# --- Classifier Evaluation Question 1 ---
# Using your predictions from KNN Question 1, create a confusion matrix and display it with ConfusionMatrixDisplay, passing display_labels=iris.target_names. Save the figure to outputs/knn_confusion_matrix.png. Add a comment: which pair of species does the model most often confuse (if any)?

c_matrix = confusion_matrix(y_test, y_predict)
c_matrix_display = ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = iris.target_names)

c_matrix_display.plot()
plt.title("KNN Confusion Matrix (k = 5)")

plt.savefig(f"{output_path}/knn_confusion_matrix.png")
plt.close()

# The model confuses Virginica and Veriscolor more often.

# The sklearn API: Decision Trees

# --- Decision Trees Question 1 ---
# Create a DecisionTreeClassifier(max_depth=3, random_state=42), fit it on the unscaled training data, and predict on the test set. Print the accuracy score and classification report. Add a comment comparing the Decision Tree accuracy to KNN. Then add a second comment: given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?
d_tree_classifier = DecisionTreeClassifier(max_depth = 3, random_state = 42)
d_tree_classifier.fit(X_train, y_train)

d_tree_predict = d_tree_classifier.predict(X_test)

print(f"Accuracy Score : {accuracy_score(y_test, d_tree_predict)}")
print(f"Classification report :\n {classification_report(y_test, d_tree_predict)}")

# The Decision Tree accuracy of 0.9667 is higher than the KNN accuracy of 0.9333.

# It the data was scaled or unscaled, the Decision Tree would make the same decisions because it doesn't rely on the measurement of distance between points.

# Logistic Regression and Regularization

# --- Logistic Regression Question 1 ---
# Train three logistic regression models on the scaled Iris data, identical in every way except for the C parameter: C=0.01, C=1.0, and C=100. Use max_iter=1000 and solver='liblinear' for all three. For each model, print the C value and the total size of all coefficients using np.abs(model.coef_).sum(). Add a comment: what happens to the total coefficient magnitude as C increases? What does this tell you about what regularization is doing?
c_values = [0.01, 1.0, 100]

for c in c_values:
    log_regression_model = LogisticRegression(C = c, max_iter = 1000, solver = "liblinear")
    log_regression_model.fit(X_train_scaled, y_train)
    print(f"C is {c}")
    print(f"Total Size of all coefficients : {np.abs(log_regression_model.coef_).sum()}")
    
# As C increases the total coefficient magnitude increases.  This shows that regularization is being reduced and is allowing the model to fit the training data with more flexibility by using larger weights.

# PCA

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# --- PCA Question 1 ---
# Print the shape of X_digits and images. Then create a 1-row subplot showing one example of each digit class (0-9), using cmap='gray_r' with each digit's label as the title. Save the figure to outputs/sample_digits.png. (gray_r is the reversed grayscale colormap -- it renders higher pixel values as darker, so digits appear as dark ink on a light background, which is more readable than the default.)

# Read not to use f strings so changing how I handle strings
print(f"Shape of X_digits : {X_digits.shape}")
print(f"Shape of images : {images.shape}")

fig, axs = plt.subplots(1, 10, figsize = (3, 1))

for digit in range(10):
    # find first occurrence of each digit
    idx = next(i for i, label in enumerate(y_digits) if label == digit)

    axs[digit].imshow(images[idx], cmap='gray_r')
    axs[digit].set_title(f"{digit}")
    axs[digit].axis("off")

plt.tight_layout()

plt.savefig(f"{output_path}/sample_digits.png")
plt.close()

# --- PCA Question 2 ---
# Fit PCA() on X_digits (with no n_components argument) then get the scores with scores = pca.transform(X_digits). As in the lesson, scores tell you how strongly each component is weighted for each sample -- scores[i, 0] is the weighting for PC1 in sample i, scores[i, 1] is the weighting for PC2, and so on.
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

# Use scores[:, 0] and scores[:, 1] to make a scatter plot, coloring each point by its digit label and adding a colorbar. Save the figure to outputs/pca_2d_projection.png. Add a comment: do same-digit images tend to cluster together in this 2D space?

plt.figure()
scatter = plt.scatter(scores[:, 0], scores[:, 1], c = y_digits, cmap = "tab10", s = 10)
plt.colorbar(scatter, label='Digit')

# Got help from AI in the best title for my code
plt.title("2D PCA Projection of Handwritten Digits")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig(f"{output_path}/pca_2d_projection.png")
plt.close()

# Same digit images tend to form loose clusters on the scatter plot, but there is a lot of overlap.  PCA groups images based on overall visual differences, not on which digit they are, which is why different digits can overlap in the 2D plot.

# --- PCA Question 3 ---
# Using the PCA object you fit in Question 2, plot cumulative explained variance vs. number of components using np.cumsum(pca.explained_variance_ratio_). Save to outputs/pca_variance_explained.png. Add a comment: approximately how many components do you need to explain 80% of the variance?
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure()

plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.grid(True)

plt.savefig(f"{output_path}/pca_variance_explained.png")
plt.close()

n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
print(n_components_80)
# We would need approximately 13 components to explain 80% of the variance.

# --- PCA Question 4 ---
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Using this function, the PCA object, and the scores from Question 2, reconstruct the first 5 digits in X_digits using reconstruction through principal components n = 2, 5, 15, and 40.
# Build a grid of subplots where rows correspond to each n value and columns show those 5 digits. Add an "Original" row at the top (use images[i], which is already shaped as (8, 8)). Save to outputs/pca_reconstructions.png.
n_components = [2, 5, 15, 40]

fig, axs = plt.subplots(5, 5)

for col in range(5):
    axs[0, col].imshow(images[col], cmap = "gray_r")
    axs[0, col].axis("off")
    axs[0, col].set_title(f"Digit {col}")

for row, n in enumerate(n_components, start = 1):
    for col in range(5):
        reconstructed_digit = reconstruct_digit(col, scores, pca, n)
        axs[row, col].imshow(reconstructed_digit, cmap = "gray_r")
        axs[row, col].axis("off")
        axs[row, col].set_title(f"n = {n}")

plt.tight_layout()        
plt.savefig(f"{output_path}/pca_reconstructions.png")
plt.close()

# Add a comment: at what n do the digits become clearly recognizable, and does that match where the variance curve levels off?
# The digits becomes clearly recognizable at n = 40.  This matches where the variance curve levels off.