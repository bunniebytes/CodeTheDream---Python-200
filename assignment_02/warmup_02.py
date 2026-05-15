import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

output_path = "./outputs"
os.makedirs(output_path, exist_ok=True)

# The scikit-learn API

# --- Question 1 ---
# Create a LinearRegression model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# Initializing model and train the model
model = LinearRegression()
model.fit(years, salary)

# slope (model.coef_[0]), the intercept (model.intercept_)
# View model Parameters
print(f"Slope (Coefficient) : {model.coef_[0]}")
print(f"Intercept : {model.intercept_}")

# Prediction for 4 years experience
prediction_4_years = model.predict([[4]])
print(f"Prediction for years = 4 : {prediction_4_years}")

# Prediction for 8 years experience
prediction_8_years = model.predict([[8]])
print(f"Prediction for years = 8 : {prediction_8_years}")

# --- Question 2 ---
# Start with this 1D array.  Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. Add a comment explaining, in your own words, why scikit-learn needs X to be 2D.
x = np.array([10, 20, 30, 40, 50])
print(f"Shape of x before reshape() : {x.shape}")
x_reshape = x.reshape(-1, 1)
print(f"Shape of x after reshape : {x_reshape.shape}")

# scikit learn expects X to be a 2D array because it treats data as rows and columns (samples and features).  Even if there is only one feature it still needs to be a column for it to understand the structure of the dataset.

# --- Question 3 ---
# Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, and predict a cluster label for each point. Print the cluster centers (kmeans.cluster_centers_) and how many points fell into each cluster using np.bincount(labels).
# Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels. Save the figure to outputs/kmeans_clusters.png.

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# Creating the KMeans model and fit to X_clusters
kmeans_model = KMeans(n_clusters = 3, random_state = 42)
kmeans_model.fit(X_clusters)

# Predict cluster label for points
labels = kmeans_model.predict(X_clusters)

# Print cluster centers and number of points
cluster_centers = kmeans_model.cluster_centers_
print(f"Cluster centers : {cluster_centers}")
print(f"Number of points in each cluster : {np.bincount(labels)}")

plt.figure()
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c = labels)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker = "X", c = "black", label = "Centers")

# Adding title
plt.title("Kmeans Clusters with Their Centers")
plt.xlabel("x")
plt.ylabel("y")

plt.savefig(f"{output_path}/kmeans_clusters.png")
plt.close()

# Linear Regression
# The questions below all use the same synthetic medical costs dataset: 100 patients, each with an age (20 to 65), a smoker flag (0 = non-smoker, 1 = smoker), and an annual medical cost as the target. Generate it once and reuse the variables throughout.

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# --- Question 1 ---
# Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and cost on the y-axis. Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). Add a title "Medical Cost vs Age", label both axes, and save to outputs/cost_vs_age.png.

# Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the smoker variable?

# Create new figure for scatter plot
plt.figure()
scatter = plt.scatter(age, cost, c = smoker, cmap = "coolwarm")

# Adding titles and labels to smoker vs non smoker on scatter plot
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")

# I don't like how the bar looks, but following instructions it is asking for me to use cmap and not create 2 seperate scatter plots for smoker vs non smoker
cbar = plt.colorbar(scatter)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Non-smoker", "Smoker"])

# Saving figure to outputs and closing
plt.savefig(f"{output_path}/cost_vs_age.png")
plt.close()

# I see 2 distinct groups, one for smoker and one for non-smoker.  This suggest the cost of the medical care for a smoker is higher than the cost of a non smoker.

# --- Question 2 ---
# Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42. Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.

age_reshape = age.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(age_reshape, cost, test_size = 0.2, random_state = 42)

print(f"Shape of X_train : {X_train.shape}")
print(f"Shape of X_test : {X_test.shape}")
print(f"Shape of y_train : {y_train.shape}")
print(f"Shape of y_test : {y_test.shape}")

# --- Question 3 ---
# Fit a LinearRegression model to your training data from Question 2. Print the slope and intercept. Then predict on the test set and print:

    # RMSE: np.sqrt(np.mean((y_pred - y_test) ** 2))
    # R² on the test set: model.score(X_test, y_test)

# Add a comment interpreting the slope in plain English -- what does it mean for medical costs?

# Training data = X_train and y_train
model = LinearRegression()
model.fit(X_train, y_train)

# Print slope (coefficient) and intercept and print
print(f"Slope (Coefficient) : {model.coef_[0]}")
print(f"Intercept : {model.intercept_}")

# Predict on test set
y_predict = model.predict(X_test)

# RMSE = Root Mean Squared Error
rmse = np.sqrt(np.mean((y_predict - y_test) ** 2))
print(f"RMSE : {rmse}")

# R Squared
r2 = model.score(X_test, y_test)
print(f"R² : {r2}")

# The slope is telling us that the cost will go up every year of age by approximately $196.58

# --- Question 4 ---
# Now add smoker as a second feature and fit a new model.

# Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding the smoker flag help? Print both coefficients:

# print("age coefficient:    ", model_full.coef_[0])
# print("smoker coefficient: ", model_full.coef_[1])

# Add a comment interpreting the smoker coefficient: what does it represent in practical terms?

X_full = np.column_stack([age, smoker])

X_train, X_test, y_train, y_test = train_test_split(X_full, cost, test_size = 0.2, random_state = 42)

model_full = LinearRegression()
model_full.fit(X_train, y_train)

# R Squared
r2_full = model_full.score(X_test, y_test)
print(f"R² full : {r2_full}")

print(f"Age Coefficient : {model_full.coef_[0]}")
print(f"Smoker Coefficient : {model_full.coef_[1]}")

# Smoking has a higher predicted midecal cost every year.

# --- Question 5 ---
# A predicted vs actual plot is a standard tool for evaluating regression models. Each test observation becomes a dot: the model's prediction goes on the x-axis, the true value goes on the y-axis. A perfect model would place every point on the diagonal line where predicted equals actual.

# Using the two-feature model from Linear Regression Question 4, create this plot for the test set. Add a diagonal reference line, a title "Predicted vs Actual", labeled axes, and save to outputs/predicted_vs_actual.png.

# Add a comment: what does it mean when a point falls above the diagonal? What about below?

y_predict_full = model_full.predict(X_test)

# Creating new figure for scatter plot
plt.figure()
plt.scatter(y_predict_full, y_test)

# Adding diagonal reference line
min_val = min(min(y_predict_full), min(y_test))
max_val = max(max(y_predict_full), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val], color = "red")

# Adding titles
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")

plt.savefig(f"{output_path}/predicted_vs_actual.png")
plt.close()

# If the point falls below the diagonal it means the actual cost was less than the predicted cost.  If the point is above the diagonal it means the actual cost was more than the predicted cost.