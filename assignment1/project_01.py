import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path
from prefect import task, flow
from prefect.logging import get_run_logger
from scipy import stats
from scipy.stats import pearsonr

path = "./happiness_project/"
output_path = "./outputs"

# Task 1: Load Multiple Years of Data
@task
def get_files(path):
    folder = Path(path)
    files = [f for f in folder.iterdir() if f.suffix == ".csv"]
    return files

@task(retries=3, retry_delay_seconds=2)
def create_df(files, outputs_path):
    logger = get_run_logger()
    logger.info("Reading files to create data frame")
    dfs = []
    for f in files:
        # Gets the year from the file name
        logger.info("Getting year from the file name")
        year = f.stem.split("_")[-1]
        
        # Data columns are seperated by ; and decimal points are ,
        logger.info(f"Creating data frame for {year}")
        df = pd.read_csv(f, sep = ";", decimal = ",")
        
        logger.info("Adding the year to the data frame")
        df["year"] = int(year)
        
        # Year 2024 has ladder score instead of happiness score, so cleaning that before merging into the combined df
        if "Happiness score" not in df.columns and "Ladder score" in df.columns:
            df["Happiness score"] = df["Ladder score"]
            df = df.drop(columns=["Ladder score"])
        dfs.append(df)
        
    # Combines my list of dfs to a single df
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info("Data frames successfully created and merged")
    
    # Check that the directory exists, creates it if it does not
    os.makedirs(outputs_path, exist_ok=True)
    
    logger.info("Creating new csv file to outputs folder")
    combined_df.to_csv("outputs/combined_data.csv")
 
    return combined_df

# Originally here before confirming Ladder score and Happiness score were the same column
# @task(retries=3, retry_delay_seconds=2)
# def clean_data(df):
#     logger = get_run_logger()
#     logger.info("Replacing all NaN values with 0 in the DataFrame")
    
#     cleaned_df = df.fillna(0)  # replaces NaN with 0 for all columns
    
#     logger.info("Successfully replaced NaN values with 0")
#     return cleaned_df

# Task 2: Descriptive Statistics
@task(retries=3, retry_delay_seconds=2)
def get_overall_happiness_stats(df):
    logger = get_run_logger()
    logger.info("Finding overall happiness mean, median, and standard deviation")
    overall_happiness = {"mean" : df["Happiness score"].mean(),
                      "median" : df["Happiness score"].median(),
                      "standard deviation" : df["Happiness score"].std()}
    logger.info("Successfully found overall happiness stats")
    return overall_happiness

# Originally was going to have this as 2 seperate functions, but then realized same code just column name was different
@task(retries=3, retry_delay_seconds=2)
def get_grouped_happiness_stats(df, column_name):
    logger = get_run_logger()
    logger.info(f"Finding {column_name} happiness mean, median, and standard deviation")
    happiness_stats_by_group = df.groupby(column_name)["Happiness score"].agg(["mean", "median", "std"]).to_dict(orient = "index")
    logger.info(f"Successfully found {column_name} happiness stats")
    return happiness_stats_by_group

# Task 3: Visual Exploration
# A histogram of all happiness scores across all years. Save as happiness_histogram.png.
@task(retries=3, retry_delay_seconds=2)
def plot_histogram(df, outputs_path):
    logger = get_run_logger()
    logger.info("Creating histogram for Happiness Scores Across All Years")
    
    # Makes sure to create a different window for each plot
    plt.figure(figsize=(8,6))
    plt.hist(df["Happiness score"], bins=20, color="skyblue", edgecolor="black")
    
    # Adding titles and labels
    plt.title("Histogram of Happiness Scores Across All Years")
    plt.xlabel("Happiness Score")
    plt.ylabel("Count")
    
    # Adds lines for readability
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    
    # Saving plot to outputs
    plt.savefig(f"{outputs_path}/happiness_histogram.png")
    logger.info("Successfully created and saved histogram")
    # closes the plot after the figure is saved
    plt.close()

# A boxplot comparing happiness score distributions across years (one box per year). Save as happiness_by_year.png.
@task(retries=3, retry_delay_seconds=2)
def plot_boxplot(df, outputs_path):
    logger = get_run_logger()
    logger.info("Creating boxplot for Happiness Score Distribution by Year")
    # Makes sure to create a different window for each plot
    plt.figure(figsize=(10,6))
    plt.boxplot([df[df["year"]==y]["Happiness score"] for y in sorted(df["year"].unique())],
                labels=sorted(df["year"].unique()))
    
    # Adding titles
    plt.title("Happiness Score Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    
    # Rotates text so easier to read/does not overlap
    plt.xticks(rotation=45)
    # Adds lines for readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # Saving plot to outputs
    plt.savefig(f"{outputs_path}/happiness_by_year.png")
    logger.info("Successfully created and saved boxplot")
    # closes the plot after the figure is saved
    plt.close()

# A scatter plot showing the relationship between GDP per capita and happiness score. Save as gdp_vs_happiness.png.
@task(retries=3, retry_delay_seconds=2)
def plot_scatterplot(df, outputs_path):
    logger = get_run_logger()
    logger.info("Creating scatterplot for GDP per Capita vs Happiness Score")
    
    plt.figure(figsize=(8,6))
    plt.scatter(df["GDP per capita"], df["Happiness score"], color="green", alpha=0.7)
    
    # Adding titles
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    
    # Adding lines for readability
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(f"{outputs_path}/gdp_vs_happiness.png")
    logger.info("Successfully created and saved scatterplot")
    
    # closes plot after saving
    plt.close()

# A correlation heatmap (using sns.heatmap() with annot=True) showing the Pearson correlations between all numeric columns. Save as correlation_heatmap.png.
@task(retries=3, retry_delay_seconds=2)
def plot_heatmap(df, outputs_path):
    logger = get_run_logger()
    
    logger.info("Selecting only numeric columns and creating a correlation matrix")
    # selects numeric columns only so we can create a matrix
    numeric_cols = df.select_dtypes(include="number")

    # computes the correlation matrix
    corr = numeric_cols.corr()
    logger.info("Correlation matrix successfully created")

    logger.info("Creating heatmap for Correlation of Numeric Columns")
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    
    # Adding titles
    plt.title("Correlation Heatmap of Numeric Columns")
    plt.tight_layout()
    
    plt.savefig(f"{outputs_path}/correlation_heatmap.png")
    logger.info("Successfully created and saved heatmap")
    
    plt.close()

# Task 4: Hypothesis Testing
@task
def pandemic_happiness_test(df):
    logger = get_run_logger()
    
    # Happiness scores for 2019 and 2020, drops any NaN values
    scores_2019 = df[df["year"] == 2019]["Happiness score"].dropna()
    scores_2020 = df[df["year"] == 2020]["Happiness score"].dropna()
    
    # find the mean for both years
    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()
    
    # independent samples t-test
    t_stat, p_val = stats.ttest_ind(scores_2019, scores_2020, equal_var=False)
    
    # logs the results
    logger.info(f"2019 mean happiness: {mean_2019:.4f}")
    logger.info(f"2020 mean happiness: {mean_2020:.4f}")
    logger.info(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    
    # if/else statement for interpretation at alpha = 0.05
    if p_val < 0.05:
        interpretation = (
            f"The difference in mean happiness between 2019 ({mean_2019:.3f})and 2020 ({mean_2020:.3f}) is statistically significant.  This suggests the pandemic may have impacted global happiness scores."
        )
    else:
        interpretation = (
            f"No statistically significant difference in mean happiness between 2019 ({mean_2019:.2f}) and 2020 ({mean_2020:.2f}) was found at alpha = 0.05."
        )
    
    logger.info(f"Interpretation: {interpretation}")
    
    return {
        "mean_2019": mean_2019,
        "mean_2020": mean_2020,
        "t_stat": t_stat,
        "p_val": p_val,
        "interpretation": interpretation
    }

@task
def region_happiness_comparison_test(df, region1, region2):
    logger = get_run_logger()
    
    # Drops any values that are NaN
    scores1 = df[df["Regional indicator"] == region1]["Happiness score"].dropna()
    scores2 = df[df["Regional indicator"] == region2]["Happiness score"].dropna()
    
    mean1 = scores1.mean()
    mean2 = scores2.mean()
    
    t_stat, p_val = stats.ttest_ind(scores1, scores2, equal_var=False)
    
    logger.info(f"{region1} mean happiness: {mean1:.4f}")
    logger.info(f"{region2} mean happiness: {mean2:.4f}")
    logger.info(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    
    # if/else statement for interpretation at alpha = 0.05
    if p_val < 0.05:
        interpretation = (
            f"The difference in mean happiness between {region1} ({mean1:.3f}) and {region2} ({mean2:.3f}) is statistically significant."
        )
    else:
        interpretation = (
            f"No statistically significant difference in mean happiness between {region1} ({mean1:.2f}) and {region2} ({mean2:.2f}) at alpha = 0.05."
        )
        
    logger.info(f"Interpretation: {interpretation}")
    
    return {
        "region1": region1,
        "region2": region2,
        "mean1": mean1,
        "mean2": mean2,
        "t_stat": t_stat,
        "p_val": p_val,
        "interpretation": interpretation
    }

# Task 5: Correlation and Multiple Comparisons
@task
def correlation_with_happiness(df):
    logger = get_run_logger()
    
    # select numeric explanatory variables (not including Happiness score)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "Happiness score"]
    
    results = []
    
    # run Pearson correlation for each numeric explanatory variable
    for col in numeric_cols:
        x = df[col].dropna() # Drops any values that are NaN
        y = df.loc[x.index, "Happiness score"].dropna()  # align indices
        # align lengths
        common_index = x.index.intersection(y.index)
        x = x.loc[common_index]
        y = y.loc[common_index]
        
        if len(x) > 1:  # need at least 2 points for pearsonr
            r, p = pearsonr(x, y)
            results.append({"variable": col, "r": r, "p_val": p})
            logger.info(f"{col}: r = {r:.4f}, p-value = {p:.4f}")
        else:
            logger.info(f"{col}: Not enough data for correlation")
    
    # Bonferroni correction
    number_of_tests = len(results)
    adjusted_alpha = 0.05 / number_of_tests if number_of_tests > 0 else 0
    
    logger.info(f"Number of tests: {number_of_tests}, Adjusted alpha = {adjusted_alpha:.4f}")
    
    # log significance at original and adjusted alpha
    for res in results:
        sig_original = res["p_val"] < 0.05
        sig_adjusted = res["p_val"] < adjusted_alpha
        logger.info(
            f"{res['variable']} significance - original alpha: {sig_original}, "
            f"after Bonferroni correction: {sig_adjusted}"
        )
    
    return {"results": results, "number_of_tests": number_of_tests, "adjusted_alpha": adjusted_alpha}

# Task 6: Summary Report
@task
def final_report(df, overall_happiness, happiness_by_year, happiness_by_region, pandemic_results, region_happiness_comparison, correlation_results):
    # a summary of the pipeline for people to understand what this code does (Also so I can remember what this code is about xD).
    logger = get_run_logger()
    
    # Total number of countries and years
    num_countries = df["Country"].nunique()
    num_years = df["year"].nunique()
    logger.info(f"Total countries in dataset: {num_countries}")
    logger.info(f"Total years in dataset: {num_years}")
    
    # Overall happiness stats
    logger.info(f"Overall happiness stats: {overall_happiness}")
    
    # Happiness by year
    logger.info(f"Happiness stats by year: {happiness_by_year}")
    
    # Happiness by region
    logger.info(f"Happiness stats by region: {happiness_by_region}")
    
    # Region t-test result
    region_interp = region_happiness_comparison.get("interpretation", "No interpretation available")
    region1 = region_happiness_comparison.get("region1", "Region 1")
    region2 = region_happiness_comparison.get("region2", "Region 2")
    logger.info(f"Region comparison between {region1} and {region2}: {region_interp}")
    
    # Top 3 and bottom 3 regions by mean happiness
    region_means = df.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
    top3 = region_means.head(3)
    bottom3 = region_means.tail(3)
    
    logger.info(f"Top 3 regions by mean happiness score: {top3.to_dict()}")
    logger.info(f"Bottom 3 regions by mean happiness score: {bottom3.to_dict()}")
    
    # Pre/post-2020 t-test interpretation
    ttest_interp = pandemic_results.get("interpretation", "No interpretation available")
    logger.info(f"Pre/post-2020 happiness comparison: {ttest_interp}")
    
    # Variable most strongly correlated with happiness (after Bonferroni)
    # Filter only significant after Bonferroni
    sig_results = [res for res in correlation_results["results"] 
                   if res["p_val"] < correlation_results["adjusted_alpha"]]
    
    if sig_results:
        # find the variable with largest absolute correlation
        strongest = max(sig_results, key=lambda x: abs(x["r"]))
        logger.info(
            f"Variable most strongly correlated with happiness (after Bonferroni correction): "
            f"{strongest['variable']} with r = {strongest['r']:.4f}, p = {strongest['p_val']:.4f}"
        )
    else:
        logger.info("No variables remained significant after Bonferroni correction.")

@flow
def flow_function(path, outputs_path):
    logger = get_run_logger()
    files = get_files(path)
    df = create_df(files, outputs_path)
    
    overall_happiness = get_overall_happiness_stats(df)
    logger.info(f"Overall happiness stats : {overall_happiness}")
    
    happiness_by_year = get_grouped_happiness_stats(df, "year")
    happiness_by_region = get_grouped_happiness_stats(df, "Regional indicator")
    
    # running all the plots now
    plot_histogram(df, outputs_path)
    plot_boxplot(df, outputs_path)
    plot_scatterplot(df, outputs_path)
    plot_heatmap(df, outputs_path)
    
    # running the hypothesis testing now
    pandemic_happiness_stats = pandemic_happiness_test(df)
    region_happiness_comparison = region_happiness_comparison_test(df, "North America and ANZ", "Western Europe")
    
    # running correlation and comparisons
    corr_results = correlation_with_happiness(df)
    
    # final report
    logger.info("Running final report")
    final_report(df,
                 overall_happiness,
                 happiness_by_year,
                 happiness_by_region,
                 pandemic_happiness_stats,
                 region_happiness_comparison,
                 corr_results)

if __name__ == "__main__":
    flow_function(path, output_path)