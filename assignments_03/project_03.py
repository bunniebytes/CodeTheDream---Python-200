import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from prefect import task, flow
from prefect.logging import get_run_logger

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = "./spambase/spambase.data"
output_path = "./outputs"
os.makedirs(output_path, exist_ok=True)

# Part 2: Mini-Project -- Spam or Ham? A Classifier Shootout

# --- Task 1: Load and Explore ---
@task()
def load_data(file):
    logger = get_run_logger()
    
    logger.info("Loading and splitting data into X, y")
    
    column_names = ["word_freq_make", "word_freq_address", "word_freq_all",
        "word_freq_3d", "word_freq_our", "word_freq_over",
        "word_freq_remove", "word_freq_internet","word_freq_order",	"word_freq_mail", "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you",	"word_freq_credit", "word_freq_your", "word_freq_font",	"word_freq_000", "word_freq_money","word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm",
        "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", "spam"]
    
    # Loading the data into a dataframe
    df = pd.read_csv(data, names = column_names)
    df["spam"] = df["spam"].astype(int)
    
    # print(df.head(10))
    return df
    
@task()
def plot_boxplot(df):
    logger = get_run_logger()
    
    logger.info("Creating box plots for word_freq_free, char_freq_!, and capital_run_length_total")
    
    columns_to_plot = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
    
    spam_emails = df[df["spam"] == 1]
    ham_emails = df[df["spam"] == 0]
    
    for feature in columns_to_plot:
        plt.figure()
        plt.boxplot([ham_emails[feature], spam_emails[feature]], tick_labels = ["Ham", "Spam"])
        plt.title(f"{feature} Distribution")
        plt.savefig(f"{output_path}/{feature}_distribution.png")
        plt.close()
    
# --- Task 2: Prepare Your Data ---
task()
def train_test_split_data(df):
    logger = get_run_logger()
    
    logger.info("Splitting the data into Train and Test sets")
    # Split data into X and y.  Documentation says last column is if it is spam or not
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    return X_train, X_test, y_train, y_test
    
@task()
def scale_data(train_set, test_set):
    logger = get_run_logger()
    
    logger.info("Scaling the train and test sets")
    scaler = StandardScaler()
    
    train_set_scaled = scaler.fit_transform(train_set)
    test_set_scaled = scaler.transform(test_set)
    
    return train_set_scaled, test_set_scaled

# Before we can scale our data we need to make sure the split them into train and test sets.  We only fit_transform the train set and then transform the test set using the same scaler we fit to the train set.  We also used stratify = y to keep the same proporions of spam vs ham in train and test.  We will not use the scaled data on Decision Tree or Random Forest but will use it on KNN and Logistic Regression

# PCA preprocessing
@task()
def evaluate_pca(train_set_scaled, test_set_scaled):
    logger = get_run_logger()
    
    logger.info("Finding the PCA on the scaled train set")
    
    pca = PCA()
    pca.fit(train_set_scaled)
    # scores = pca.transform(train_set_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure()

    plt.plot(cumulative_variance)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")

    plt.grid(True)

    plt.savefig(f"{output_path}/pca_spam_variance_explained.png")
    plt.close()
    
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    print(f"Number of n components : {n_components_90}")

    X_train_pca = pca.transform(train_set_scaled)[:, :n_components_90]
    X_test_pca  = pca.transform(test_set_scaled)[:, :n_components_90]
    
    return X_train_pca, X_test_pca

# --- Task 3: A Classifier Comparison ---
@task()
def evaluate_knn(X_train, X_test, y_train, y_test, test_set_name):
    logger = get_run_logger()
    
    logger.info(f"Finding {test_set_name}.")
    
    # creating dictionary to store values to compare later
    knn_dict = {}
    knn_dict["name"] = test_set_name
    knn = KNeighborsClassifier(n_neighbors = 5)

    # fits on unscaled trainings data
    knn.fit(X_train, y_train)
    
    # predicts on test set and adds to dictionary
    y_predict = knn.predict(X_test)
    knn_dict["y_predict"] = y_predict
    
    # finds accuracy score and classification report
    acc_score = accuracy_score(y_test, y_predict)
    knn_dict["accuracy_score"] = acc_score
    class_report = classification_report(y_test, y_predict)
    knn_dict["classification_report"] = class_report
    

    print(f"Accuracy Score for {test_set_name} : {acc_score}")
    print(f"Classification report for {test_set_name} :\n {class_report}")
    
    return knn

@task()
def evaluate_decision_tree(X_train, X_test, y_train, y_test):
    logger = get_run_logger()
    
    logger.info("Testing different max depths for decision tree and finding the classification report for that")
    max_depth = [3, 5, 10, None]
    
    best_model = None
    best_depth = None
    best_test_predict = None
    best_test_accuracy = -1
    
    for depth in max_depth:
        d_tree_classifier = DecisionTreeClassifier(max_depth = depth, random_state = 42)
        d_tree_classifier.fit(X_train, y_train)

        # Finding train and test predictions
        train_predict = d_tree_classifier.predict(X_train)
        test_predict = d_tree_classifier.predict(X_test)
        
        # Finding accuracy scores for these sets
        train_accuracy = accuracy_score(y_train, train_predict)
        test_accuracy = accuracy_score(y_test, test_predict)
        
        # Find the difference between training and test accuracy to help pinpoint overfitting
        gap = train_accuracy - test_accuracy
        
        print(f"Max depth : {depth}")
        print(f"Training accuracy : {train_accuracy}")
        print(f"Test accuracy : {test_accuracy}")
        print(f"Gap : {gap}")
        
        # Penalize for overfitting (this makes sure that in instances where depth is None (unlimited) but it is overfitting, the program does not choose this over the actual best max depth)
        score = test_accuracy - 0.1 * gap
        
        if score > best_test_accuracy:
            best_test_accuracy = score
            best_model = d_tree_classifier
            best_depth = depth
            best_test_predict = test_predict
        
    # I notice as depth increases the accuracy increases as well.  At None (or unlimited) the training accuracy is almost at exactly 1.  The difference between the training accuracy and the test accuracy became greater at this point, which is a sign of over fitting.

    print(f"Best depth : {best_depth}")
    print(f"Accuracy Score : {accuracy_score(y_test, best_test_predict)}")
    print(f"Classification report :\n {classification_report(y_test, best_test_predict)}")
    
    return best_model

@task()
def evaluate_rand_forest(X_train, X_test, y_train, y_test):
    logger = get_run_logger()
    
    logger.info("Finding random forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_predict = rf.predict(X_test)
    
    print(f"Accuracy Score : {accuracy_score(y_test, y_predict)}")
    print(f"Classification report :\n {classification_report(y_test, y_predict)}")
    
    return rf, y_predict

@task()
def find_top_10_features(X_train, model, model_name):
    logger = get_run_logger()
    
    logger.info(f"Finding the top 10 features of the {model_name}")
    importance_df = pd.DataFrame({"feature" : X_train.columns,
                                  "importance" : model.feature_importances_})
    
    importance_df = importance_df.sort_values(by = "importance", ascending = False)
    
    return importance_df.head(10)

@task()
def plot_importances_bar_chart(df, model_name):
    logger = get_run_logger()
    
    logger.info(f"Plotting best features on a bar chart for {model_name}")
    
    plt.figure()
    plt.bar(df["feature"], df["importance"])
    
    plt.title(f"Top 10 Features of {model_name}")
    plt.xticks(fontsize=6, rotation=45, ha='right')
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_importances.png")
    plt.close()

@task()
def evaluate_log_regression_model(X_train_set, X_test_set, y_train, y_test, train_set_name):
    logger = get_run_logger()
    
    logger.info(f"Finding logistic regression model on {train_set_name}")
    
    log_regression_model = LogisticRegression(C = 1.0, max_iter = 1000, solver = 'liblinear')
    log_regression_model.fit(X_train_set, y_train)
    
    y_predict = log_regression_model.predict(X_test_set)
    
    print(f"Total Size of all coefficients : {np.abs(log_regression_model.coef_).sum()}")
    print(f"Accuracy Score : {accuracy_score(y_test, y_predict)}")
    print(f"Classification report :\n {classification_report(y_test, y_predict)}")
    
    return log_regression_model

@task()
def plot_confusion_matrix(y_test, y_predict):
    logger = get_run_logger()
    
    logger.info("Plotting Confusion Matrix using Random Forest Classifier because it had best accuracy")
    
    c_matrix = confusion_matrix(y_test, y_predict)
    c_matrix_display = ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ["Ham", "Spam"])

    c_matrix_display.plot()
    plt.title("Random Forest Confusion Matrix (k = 2)")

    plt.savefig(f"{output_path}/best_model_confusion_matrix.png")
    plt.close()
    
    # A mistake the Random Forest Classifier makes most often is that it lets in more spam emails than accidently blocking actual emails.  This is okay in my opinion as we would rather spam get through that actual emails missed because they are filtered out.

# The model that performed best was the Random Forest Classifier.  When I compared PCA vs non-PCA, the non-PCA (knn_scaled and log_regression_model_scaled) performed better than PCA (knn_pca and log_regression_model_pca).  I originally thought accuracy would be the right metric to optimize because hypothetically the more accurate the spam filter, the number of false positives and false negatives should go down.  The issue is, the errors (false positives vs false negatives) are no the same cost.  Ideally we would want to minimize false positives, where legitimate emails are incorrectly marked as spam because this could cause the user to miss some important emails.  False negatives, where spam is accidently let through as legitimate mail is still an issue, but would not cause as much harm as a false positive.  So while accuracy is a good measurement for general performance measure, recall and precision are more appropriate for evaluating a spam classifier.

@task()
def evaluate_cross_val(model, X_train_set, y_train, model_name):
    logger = get_run_logger()
    
    logger.info(f"Evaluating the Cross-Validation for {model_name}")
    
    # Cross validation on training data
    cv_scores = cross_val_score(model, X_train_set, y_train, cv = 5)

    # Print each fold score, mean and std
    print(f"Model : {model_name}")
    print(f"Fold scores : {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean()}")
    print(f"Standard deviation: {cv_scores.std()}")
    
    # The model that was the most accurate was Random Forest with a mean of 0.9541.  The model that was the most stable was Logistic Regression using PCA training set with a standard deviation of 0.0033.  The ranking of the accuracy matches the single train/test split, but the standard deviation shows that the Random forest is less stable that some of the other models.

# --- Task 5: Building a Prediction Pipeline ---
def build_tree_pipeline():
    pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state = 42))
])
    return pipeline

def build_non_tree_pipeline():
    pipline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C = 1.0, max_iter = 1000, solver = 'liblinear'))
    ])
    return pipline

@task()
def train_and_evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test):
    logger = get_run_logger()
    
    logger.info("Using pipeline to fit and predict")
    
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)
    print(f"Accuracy Score : {pipeline.score(X_test, y_test)}")
    print(f"Classification Report:\n {classification_report(y_test, y_predict)}")
    
    # The 2 pipelines do not have the same structure.  One calls a scaler, while the other does not.  Tree based models do not use scaled training sets because they don't need normalization while linear/distance based models benefit from normalization.  It is practical to build it this way to prevent data leakage.  This is also packaged neatly into one object so it will give consistent results without second guessing if any steps are missing.  This also makes it easier to hand off to people because again, everything is package neatly into one object and they just need to call predict().
    
@flow()
def flow_function(file):
    df = load_data(file)
    plot_boxplot(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    X_train_pca, X_test_pca = evaluate_pca(X_train_scaled, X_test_scaled)
    
    knn_raw = evaluate_knn(X_train, X_test, y_train, y_test, "knn_raw")
    knn_scaled = evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test, "knn_scaled")
    knn_pca = evaluate_knn(X_train_pca, X_test_pca, y_train, y_test, "knn_pca")
    
    decision_tree = evaluate_decision_tree(X_train, X_test, y_train, y_test)
    random_forest, random_forest_y_predict = evaluate_rand_forest(X_train, X_test, y_train, y_test)
    log_regression_model_scaled = evaluate_log_regression_model(X_train_scaled, X_test_scaled, y_train, y_test, "X_train_scaled")
    log_regression_model_pca = evaluate_log_regression_model(X_train_pca, X_test_pca, y_train, y_test, "X_train_pca")
    
    decision_tree_top_10 = find_top_10_features(X_train, decision_tree, "Decision_Tree")
    random_forest_top_10 = find_top_10_features(X_train, random_forest, "Random_Forest")
    
    plot_importances_bar_chart(random_forest_top_10, "Random_Forest")
    plot_confusion_matrix(y_test, random_forest_y_predict)
    
    evaluate_cross_val(knn_raw, X_train, y_train, "KNN_Raw")
    evaluate_cross_val(knn_scaled, X_train_scaled, y_train, "KNN_Scaled")
    evaluate_cross_val(knn_pca, X_train_pca, y_train, "KNN_PCA")
    evaluate_cross_val(decision_tree, X_train, y_train, "Decision_Tree")
    evaluate_cross_val(random_forest, X_train, y_train, "Random_Forest")
    evaluate_cross_val(log_regression_model_scaled, X_train_scaled, y_train, "Logistic_Regression_Scaled")
    evaluate_cross_val(log_regression_model_pca, X_train_pca, y_train, "Logistic_Regression_PCA")
    
    tree_pipeline = build_tree_pipeline()
    non_tree_pipeline = build_non_tree_pipeline()
    train_and_evaluate_pipeline(tree_pipeline, X_train, X_test, y_train, y_test)
    train_and_evaluate_pipeline(non_tree_pipeline, X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    flow_function(data)