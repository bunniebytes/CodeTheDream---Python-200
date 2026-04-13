import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

# --- Pandas ---
# Pandas Question 1
# Create the following DataFrame and print the first three rows, the shape, and the data types of each column.
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# Print the first three rows
print(df.head(3))

# Print the shape of the DataFrame
print(df.shape)

# Print the data types of each column
print(df.dtypes)

# Pandas Question 2
# Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result.
passing_students_df = df[(df["grade"] > 80) & (df["passed"] == True)]
print(passing_students_df)

# Pandas Question 3
# Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows).
df["grade_curved"] = df["grade"] + 5
print(df)

# Pandas Question 4
# Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together.
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# Pandas Question 5
# Group the DataFrame by "city" and compute the mean grade for each city. Print the result.
city_mean = df.groupby("city")["grade"].mean()
print(city_mean)

# Pandas Question 6
# Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas Question 7
# Sort the DataFrame by "grade" in descending order and print the top 3 rows.
df_sorted = df.sort_values(by = "grade", ascending = False)
print(df_sorted.head(3))

# --- NumPy ---
# NumPy Question 1
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.
arr_1d = np.array([10, 20, 30, 40, 50])

# Print shape of array
print(np.shape(arr_1d))

# Print dtype of array
print(arr_1d.dtype)

# Print ndim of array
print(arr_1d.ndim)

# NumPy Question 2
# Create the following 2D array and print its shape and size (total number of elements).
arr_2d = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Print shape of array
print(np.shape(arr_2d))

# Print size of array
print(arr_2d.size)

# NumPy Question 3
# Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].
sliced_arr = arr_2d[:2, :2]
print(sliced_arr)

# NumPy Question 4
# Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.
array_zero = np.zeros((3, 4))
array_one = np.ones((2, 5))

print(array_zero)
print(array_one)

# NumPy Question 5
# Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.
array_arange = np.arange(0, 50, 5)

# I believe that this will be a 1d array, we know it will increment by 5 starting from 0.

print(array_arange)
print(array_arange.shape)
print(np.mean(array_arange))
print(np.sum(array_arange))
print(np.std(array_arange))

# NumPy Question 6
# Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.

# If we don't specify loc(mean) or scale(std) their defaults will be 0 and 1 which is what the question is asking for
array_random = np.random.normal(size = 200)

print(np.mean(array_random))
print(np.std(array_random))


# --- Matplotlib ---
# Matplotlib Question 1
# Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Creating the plot and seperate window
plt.figure()
plt.plot(x, y)

# Adding titles
plt.title("Square")
plt.xlabel("x")
plt.ylabel("y")

# Matplotlib Question 2
# Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

# Creating the plot and making sure it shows seperate from previous plot
plt.figure()
plt.bar(subjects, scores)

# Adding titles
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

# Matplotlib Question 3
# Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure()
plt.scatter(x1, y1, color = "red", label = "Group 1", marker = "o")
plt.scatter(x2, y2, color = "blue", label = "Group 2", marker = "x")

# Adding titles
plt.title("Two Scatter Plots")
plt.xlabel("x")
plt.ylabel("y")

# Adding legend to scatter plot
plt.legend()

# Matplotlib Question 4
# Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
ax1.plot(x, y)
ax1.set_title("Question 1 Sqaure")

ax2.bar(subjects, scores)
ax2.set_title("Question 2 Scores")

plt.tight_layout()

# --- Descriptive Statistics Review ---
# Descriptive Statistics Question 1
# Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

# Saving all values to variables for easier readability
mean_val = np.mean(data)
median_val = np.median(data)
variance_val = np.var(data)
std_val = np.std(data)

# Label and print the results
print(f"Mean : {mean_val}")
print(f"Median : {median_val}")
print(f"Variance : {variance_val}")
print(f"Standard Deviation : {std_val}")

# Descriptive Statistics Question 2
# Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
random_data = np.random.normal(65, 10, 500)
plt.figure()
plt.hist(random_data, bins = 20)

# Adding titles
plt.title("Distibution of Scores")
plt.xlabel("Scores")
plt.ylabel("Count")

# Descriptive Statistics Question 3
# Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

# Prepare data to be plotted
data_to_plot = [group_a, group_b]

plt.figure()
plt.boxplot(data_to_plot, tick_labels = ["Group A", "Group B"])

# Add title
plt.title("Score Comparison")

# Descriptive Statistics Question 4
# Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# Prepare data to be plotted
data_to_plot = [normal_data, skewed_data]

# Did not do a subplot because this says to do a side by side comparison not two graphs next to each other
plt.figure()
plt.boxplot(data_to_plot, tick_labels = ["Normal", "Exponential"])

# Add Title
plt.title("Distribution Comparison")

# Then, add a comment in your code briefly noting which distribution is more skewed, and which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.
# The "Exponential" distribution is more skewed (it has a long tail to the right).  The "Normal" distribution is symmetric. For Normal, the mean is a good measure of the center. For Exponential, the median is better because the mean gets pulled by the extreme values.

# Descriptive Statistics Question 5
# Print the mean, median, and mode of the following:
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

data1_mean = np.mean(data1)
data1_median = np.median(data1)
# Finding mode, need to find the count of the values
vals1, counts1 = np.unique(data1, return_counts= True)
# Finding the index of the value that shows up the most
data1_mode_index = np.argmax(counts1)
data1_mode = vals1[data1_mode_index]

print(f"Data 1 Mean : {data1_mean}")
print(f"Data 1 Median : {data1_median}")
print(f"Data 1 Mode : {data1_mode}")

data2_mean = np.mean(data2)
data2_median = np.median(data2)
# Finding mode, need to find the count of the values
vals2, counts2 = np.unique(data2, return_counts= True)
# Finding the index of the value that shows up the most
data2_mode_index = np.argmax(counts2)
data2_mode = vals2[data2_mode_index]

print(f"Data 2 Mean : {data2_mean}")
print(f"Data 2 Median : {data2_median}")
print(f"Data 2 Mode : {data2_mode}")

# Why are the median and mean so different for data2? Add your answer as a comment in the code.
# The mean and median of data2 differ so much because 150 is an outlier.  The mean is sensitive to extreme values, so it gets pulled up by 150, which is different from the median that reflects the middle value and is resistant to outliers.

# --- Hypothesis Testing Review ---
# Hypothesis Testing Question 1
# Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)

print(f"t-statistic : {t_stat:.4f}")
print(f"p-value : {p_val:.4f}")

# Hypothesis Testing Question 2
# Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05.
alpha = 0.05

if p_val < alpha:
    print("We can assume the null hypothesis is False")
else:
    print("We can assume the null hypothesis is True")

# Hypothesis Testing Question 3
# Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_val = stats.ttest_rel(before, after)

print(f"t-statistic : {t_stat:.4f}")
print(f"p-value : {p_val:.4f}")

# Hypothesis Testing Question 4
# Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.
scores = [72, 68, 75, 70, 69, 74, 71, 73]
benchmark = 70

t_stat, p_val = stats.ttest_1samp(scores, benchmark)

print(f"t-statistic : {t_stat:.4f}")
print(f"p-value : {p_val:.4f}")

# Hypothesis Testing Question 5
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.

t_stat, p_val = stats.ttest_ind(group_a, group_b, alternative = "less")

print(f"t-statistic : {t_stat:.4f}")
print(f"p-value : {p_val:.4f}")

# Hypothesis Testing Question 6
# Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.
print("The average of Group B is significantly higher than Group A. The p-value is much smaller that 0.5 (at 0.0000), this difference is very unlikely to be due to chance.")

# --- Correlation Review ---
# Correlation Question 1
# Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
corr_coef = corr_matrix[0, 1]

print(corr_matrix)
print(f"Correlation Coefficient : {corr_coef:4f}")

# What do you expect the correlation to be, and why? Add your answer as a comment in the code.
# The correlation is expected to be 1 because y increases perfectly linearly with x.  As x goes up, y goes up in exact proportion, so the relationship is a perfect positive correlation.

# Correlation Question 2
# Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr_coef, p_val = pearsonr(x, y)

print(f"Correlation Coefficient : {corr_coef:4f}")
print(f"P Value : {p_val:4f}")

# Correlation Question 3
# Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr_matrix = df.corr()

print(corr_matrix)

# Correlation Question 4
# Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.figure()
plt.scatter(x, y)

# Adding titles
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")

# Correlation Question 5
# Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap"

sns.heatmap(corr_matrix, annot = True, cmap = "cool")

# Adding title
plt.title("Correlation Heatmap")

# --- Pipelines ---
# Pipelines Question 1
# A data pipeline is a sequence of processing steps where each step takes in data, transforms it, and passes the result to the next. You don't need a special framework to build one -- chaining plain functions together is often enough.

# Given the array below, which contains some missing values scattered throughout:
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# takes a NumPy array and returns a pandas Series with the name "values"
def create_series(arr):
    values = pd.Series(arr)
    return values

# takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series
def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series

# takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode". For mode, use series.mode()[0] to get a single value.
def summarize_data(series):
    # had to use lambdas to ensure only received the first value so we could use to_dict()
    data_summary = series.agg(mean = "mean",
                              median = "median",
                              std = "std",
                              mode = lambda x : x.mode()[0]).to_dict()
    return(data_summary)

# calls the three functions above in sequence and returns the summary dictionary.
def data_pipeline(arr):
    values = create_series(arr)
    cleaned_values = clean_data(values)
    data_summary = summarize_data(cleaned_values)
    for k, v in data_summary.items():
        print(f"{k} : {v}")

data_pipeline(arr)

# Show the all the graphs
plt.show()