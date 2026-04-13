import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from prefect import task, flow
from prefect.logging import get_run_logger

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr

# If you were loading this with pd.read_csv(), what parameter would you need to specify beyond the filename? Write that observation as a comment at the top of your script before you write the load call.
# We would need to also have the sep parameter because the separator is ";" in the csv file.

file_name = "student_performance_math.csv"
output_path = "./outputs"
# Checks if the outputs folder exists, if not it creates it
os.makedirs(output_path, exist_ok=True)

# --- Task 1: Load and Explore ---
@task(retries = 3, retry_delay_seconds = 2)
def create_df(file_name):
    # Load the dataset with the correct separator. Print the shape, the first five rows, and the data types of all columns.
    # Numeric features -- use directly as numbers
        # age, Medu, Fedu, traveltime, studytime, failures, absences, freetime, goout, Walc
    # Binary features stored as "yes"/"no" -- you will convert these to 1/0
        # schoolsup, internet, higher, activities
    # Binary feature stored as "F"/"M" -- you will convert to 0/1
        # sex - Student sex (F=0, M=1).
    # Grade columns
        # G3 - Final period grade (0-20) -- your prediction target. Note: Some students have G3=0. This represents absence from the final exam, not an actual score of zero. You will need to decide how to handle these rows.
    logger = get_run_logger()
    
    logger.info("Creating dataframe")
    df = pd.read_csv(file_name, sep = ";")
    
    print(f"Shape of dataframe: {df.shape}")
    print(df.head(5))
    print(df.dtypes)
    return df

# Then plot a histogram of G3 with 21 bins (one per possible value, 0-20). Add a title "Distribution of Final Math Grades", label both axes, and save to outputs/g3_distribution.png. You should see a cluster of zeros sitting apart from the main distribution. They represent the students who didn't take the final exam.
@task
def plot_histogram(df):
    logger = get_run_logger()
    logger.info("Plotting G3 in Histogram")
    plt.figure()
    plt.hist(df["G3"], bins = 21)
    
    # Adding titles
    plt.title("Distribution of Final Math Grades")
    plt.xlabel("G3 - Final Grade")
    plt.ylabel("Number of Students")
    
    plt.savefig(f"{output_path}/g3_distribution.png")
    plt.close()
    
# --- Task 2: Preprocess the Data ---
@task(retries = 3, retry_delay_seconds = 2)
def process_df(df):    
    logger = get_run_logger()
    
    # Filtering out the rows of G3 that are 0
    logger.info("Creating df with G3 0's filtered out")
    df_filtered = df[df["G3"] != 0].copy()
    print(df_filtered.head(5))
    
    # We want to filter them out because they would skew the data to a negative.  This is not the actual grade that the students received, they were just absent that day.
    
    # Changing the gender and the yes/no columns to 1/0
    logger.info("Mapping the binary columns to have 1/0 for their values")
    binary_columns = ["schoolsup", "internet", "higher", "activities", "sex"]
    binary_map = {"yes" : 1,
                "no" : 0,
                "F" : 0,
                "M" : 1}

    for column in binary_columns:
        df_filtered[column] = df_filtered[column].map(binary_map)
    
    logger.info("Finding Pearson Correlation for both original and filtered df")
    corr_coef, p_val = pearsonr(df["absences"], df["G3"])
    corr_coef_filtered, p_val_filtered = pearsonr(df_filtered["absences"], df_filtered["G3"])
    print(f"Original dataframe : {corr_coef}")
    print(f"Filtered dataframe : {corr_coef_filtered}")
    
    # The students with 0 as their G3 score made the absences look like a weak predictor because the 0 does not actually reflect their score just that they did not take the test.  When we remove the rows with 0 from the data frame it shows a stronger negative correlation where it reflects more absences mean lower scores.
    
    return df_filtered

# --- Task 3: Exploratory Data Analysis ---
@task(retries = 3, retry_delay_seconds = 2)
def compute_corr(df):
    # Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset, and print them sorted from most negative to most positive. Which feature has the strongest relationship with G3? Are any results surprising?
    logger = get_run_logger()
    
    logger.info("Getting correlation between G3 and all the other columns")
    corrs = df.corr(numeric_only = True)["G3"].sort_values()
    print(corrs)
    
    # The strongest relationships with G3 (not including G1 and G2) is failures and school support as well as absences.  These all have a negative relationship; so the greater they are, G3 tends to be lower.
    
@task()
def plot_scatter(df):
    logger = get_run_logger()
    
    logger.info("Creating scatter plot of correlation of Absences in relation to G3")
    
    plt.figure()
    plt.scatter(df["G3"], df["absences"], color = "blue", label = "Absences")
    
    # Add title
    plt.title("Correlation between Absences and G3")
    plt.xlabel("G3 - Final Grade")
    plt.ylabel("Absences")
    
    plt.savefig(f"{output_path}/correlation_absences_g3.png")
    plt.close()
    
@task()
def plot_boxplot(df):
    logger = get_run_logger()
    
    logger.info("Creating box plot of correlation between School Support and G3")
    
    plt.figure()
    sns.boxplot(x = "schoolsup", y = "G3", data = df)
    
    # Adding titles
    plt.title("School support in relation to G3")
    plt.xlabel("School Support")
    plt.ylabel("G3 - Final Grade")
    
    plt.savefig(f"{output_path}/correlation_schoolsup_g3.png")
    plt.close()
    
# --- Task 4: Baseline Model ---
@task
def build_model(df):
    # Build the simplest possible model: use failures alone to predict G3. Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print the slope, RMSE, and R² on the test set.
    logger = get_run_logger()
    
    logger.info("Building model using failures to predict G3")
    
    X_train, X_test, y_train, y_test = train_test_split(df[["failures"]], df["G3"], test_size = 0.2, random_state = 42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    print(f"Slope (Coefficient) : {model.coef_[0]}")
    
    # RMSE = Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_predict - y_test) ** 2))
    print(f"RMSE : {rmse}")

    # R Squared
    r2 = model.score(X_test, y_test)
    print(f"R² : {r2}")
    
    # given that grades are on a 0-20 scale, what do the slopes and RMSE tell you in plain English? Is R² better or worse than you expected from exploratory data analysis?
    # TODO
    # The model explains very little.  I actually would have thought it would have been better for the R² since failures was the strongest relationship
    
# --- Task 5: Build the Full Model ---
@task
def build_full_model(df, columns):
    # Now build a regression model using all of the numeric and binary features from the Feature Guide
    logger = get_run_logger()
    
    logger.info("Building full model using all the numeric and binary features")
    X_full = df[columns]
    
    X_train, X_test, y_train, y_test = train_test_split(X_full, df["G3"], test_size = 0.2, random_state = 42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    # Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print both train R² and test R², as well as RMSE on the test set. Compare the test R² to your baseline from Task 4 -- how much does adding more features help?
    
    # RMSE = Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_predict - y_test) ** 2))
    print(f"RMSE : {rmse}")
    
    # R Squared train and test
    r2_train = model.score(X_train, y_train)
    print(f"R² train : {r2_train}")
    r2_test = model.score(X_test, y_test)
    print(f"R² test : {r2_test}")
    
    # Adding features helped the prediction model a little, but only a little.  From one feature to all the features the R² improved from 0.09 to 0.15 so it is still not a strong model
    
    # Print each feature name alongside its coefficient:
    for name, coef in zip(columns, model.coef_):
        print(f"{name:12s}: {coef:+.3f}")
        
    # Are any signs (positive or negative) surprising given what you know about the data? For any surprising result, add a comment with your best explanation. Then compare train R² to test R² -- are they close, or is there a gap? What does that tell you about the model?
    # This was not surprising to me, but could be for someone reading the data.  For school support there was a very strong negative relationship (-2.062).  It is a bit misleading since someone with poor grades would be the ones who would need school support.  So it is most likely that they had poor grades first and then received school support, not that they have school support and still received poor grades on top of that.
    # With train R² and test R², they are close with no gap, but overall it is still a weak model because of how low they are.

    # Finally, add a comment answering: if you were deploying this model in production, which features would you keep and which would you drop? Justify your choices based on what you see in the numbers.
    # I would drop freetime, activities, as well as travel time since they are super low (close to 0) so they don't do much to help predict.  I would keep failures and studytime as they were the strongest relationships.  I would also keep higher (education) as that shows the student is invested in their future.  School support should also be kept even though it is a backwards from what we expect because it still tells us a story of if the student is doing well or not.  Mother and father education should also be kept, they are small but still have some impact as well as internet because it represents they have access to resources.
    
    return y_predict, y_test

# --- Task 6: Evaluate and Summarize ---
@task
def plot_predicted_vs_actual(predicted, test):
    logger = get_run_logger()
    logger.info("Creating scatter plot between predicted vs actual data")
    
    # Creating new figure for scatter plot
    plt.figure()
    plt.scatter(predicted, test)
    
    # Adding diagonal reference line
    min_val = min(predicted.min(), test.min())
    max_val = max(predicted.max(), test.max())
    plt.plot([min_val, max_val], [min_val, max_val], color = "red")
    
    # Adding titles
    plt.title("Predicted vs Actual (Full Model)")
    plt.xlabel("Predicted Grade")
    plt.ylabel("Actual Grade")
    
    # Saving figure to outputs
    plt.savefig(f"{output_path}/predicted_vs_actual.png")
    plt.close()
    
    # Add a comment: does the model seem to struggle more at the high end, the low end, or is error roughly uniform across grade levels? What does a value above or below the diagonal mean?
    # The model seems to be the same across all grade levels.  The values above the diagonal mean that the model predicted the grade to be less than the actual grade and when they are below they predicted the grade to be more than the actual grade
    
    # The size of the filtered dataset and the test set
    # The RMSE and R² of your best model in plain language -- on a 0-20 scale, what does a typical prediction error actually mean?
    # Which two features have the largest positive and largest negative coefficients, and what those mean
    # One result that surprised you
    # The size of the dataset only included the students that took the final and the test set was only 20% of this already filtered data.  The RMSE of 2.86 means that the prediction error can be off by almost 3 points, which on a 0-20 scale is significant. The value of 0.15 for R² shows only a small portion of variation in grades.  The feature with the largest negative coefficients was school support, but that is slightly misleading as the students who are already struggling are the ones more likely to receive the additional support, not they are receiving support and still receiving low grades.  The largest positive coefficients were higher (education) and internet showing that those with the means and resources along with motivation for further education performed a little better.
    
@flow
def flow_function(file_name):
    feature_cols = ["failures", "Medu", "Fedu", "studytime",
                    "higher", "schoolsup", "internet", "sex",
                    "freetime", "activities", "traveltime"]
    df = create_df(file_name)
    plot_histogram(df)
    
    df_filtered = process_df(df)
    compute_corr(df_filtered)
    plot_scatter(df_filtered)
    plot_boxplot(df_filtered)
    build_model(df_filtered)
    
    y_predict, y_test = build_full_model(df_filtered, feature_cols)
    plot_predicted_vs_actual(y_predict, y_test)
    
    # --- Neglected Feature: The Power of G1 ---
    with_g1_predict, with_g1_test = build_full_model(df_filtered, feature_cols + ["G1"])
    
    # Add a comment addressing these questions: does a high R² here mean G1 is causing G3? Is this a useful model for identifying students who might struggle? What might educators need to do if they wanted to intervene early, before G1 is even available?
    # High R² does not mean G1 is causing G3.  It is a strong predictor since it is the same student in the same subject meaning they will have the same study habits and motivation.  This model is good for predicting the final grade after the course has already started (first grades are in) but it is not a good model for early intervention since this model is not available until after G1 is.  If they want to intervene early they should look at absences and prior academic performance as well as study habits to predict if they need to intervene before the first grades are in. 

if __name__ == "__main__":
    flow_function(file_name)