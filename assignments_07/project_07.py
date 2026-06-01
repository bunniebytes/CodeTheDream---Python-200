import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import pearsonr

from dotenv import load_dotenv
from openai import OpenAI

from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

if load_dotenv():
    print('Successfully loaded environment variables from .env')
    api_key = os.getenv("OPENAI_API_KEY")
else:
    print('Warning: could not load environment variables from .env')

base_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_dir, "outputs")

os.makedirs(output_path, exist_ok=True)

client = OpenAI()
print('OpenAI client created.')

DATA_PATH = "../assignment_01/outputs/combined_data.csv"

class HappinessDataManager:
    def __init__(self, data_path : Path):
        self.data_path = data_path
        self.df = None
    
    def load_happiness_data(self) -> dict:
        """Load World Happiness Data into memory"""
        
        try:
            if os.path.exists(self.data_path):
                self.df = pd.read_csv(self.data_path)
                return {
                    "message": "Loaded merged dataset successfully.",
                    "shape": self.df.shape,
                    "columns": self.df.columns.tolist()
                }

            folder = Path("../assignment_01/happiness_project")
            files = sorted(folder.glob("*.csv"))

            if not files:
                return {"error": "No dataset found in merged file or yearly folder."}

            dfs = []

            for f in files:
                year = f.stem.split("_")[-1]

                temp_df = pd.read_csv(f, sep=";", decimal=",")

                temp_df["year"] = int(year)

                # Normalize column name differences
                if "Happiness score" not in temp_df.columns and "Ladder score" in temp_df.columns:
                    temp_df["Happiness score"] = temp_df["Ladder score"]
                    temp_df = temp_df.drop(columns=["Ladder score"])

                dfs.append(temp_df)

            self.df = pd.concat(dfs, ignore_index=True)

            return {
                "message": "Built dataset from yearly files successfully.",
                "shape": self.df.shape,
                "columns": self.df.columns.tolist(),
                "files_used": [f.name for f in files]
            }
        
        except Exception as e:
            return {
                "error" : f"Failed to load dataset : {e}"
        }
            
    def summarize_column(self, column : str) -> dict:
        """Summarize descriptive statistics for a single column in the loaded dataset"""
        
        if self.df is None:
            return {"error" : "Dataset does not exist.  Please run load_happiness_data()."}
        
        if column not in self.df.columns:
            return {"error" : f"{column} does not exist in the Dataset"}
        
        return self.df[column].describe().to_dict()
    
    def compute_correlation(self, col1: str, col2: str) -> dict:
        """Compute the Pearson correlation coefficient and p-value between two numeric columns.
        """
        
        if self.df is None:
            return {"error" : "Dataset does not exist.  Please run load_happiness_data()."}
        if col1 not in self.df.columns:
            return {"error" : f"{col1} does not exist in the Dataset"}
        if col2 not in self.df.columns:
            return {"error" : f"{col2} does not exist in the Dataset"}
        
        corr_coef, p_val = pearsonr(self.df[col1], self.df[col2])
        results = {"col1" : col1, "col2" : col2, "pearson_r" : round(corr_coef, 4), "p_value" : round(p_val, 4)}
        return results
    
    def get_top_n_countries(self, column: str, year: int, n: int = 5) -> dict:
        """Find the top N countries ranked by a given column for a specific year.
        """
        list_of_dict = []
        if self.df is None:
            return {"error" : "Dataset does not exist.  Please run load_happiness_data()."}
        if column not in self.df.columns:
            return {"error" : f"{column} does not exist in the Dataset"}
        
        countries = self.df[self.df["year"] == year].sort_values(by = column, ascending = False)["Country"].head(n)
        
        for rank, country in enumerate(countries, start = 1):
            list_of_dict.append({"rank" : rank, "country" : country})
        # print(list_of_dict)
        
        return list_of_dict
    
happiness_manager = HappinessDataManager(DATA_PATH)

# --- Task 1: Define Your Tools ---

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.
    
    This function reads and saves the CSV file at DATA_PATH into a DataFrame in memory so other tools can operate on it.
    
    Returns:
        dict : A status message, dataset shape and columns list or an error message
    """
    return happiness_manager.load_happiness_data()
    
@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive stats for a single column in the loaded dataset.

    Args:
        column: Name of the column to be summarized.
        
    Return:
        A dictionary with the summary stats or an error message.
    """
    return happiness_manager.summarize_column(column)
     
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    
    Args:
        col1 : The name of the first column.
        col2 : The name of the second column.
    
    Returns:
        A dictionary containing col1, col2 and the correlation coefficient and p-value
    """
    return happiness_manager.compute_correlation(col1, col2)
    
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Find the top N countries ranked by a given column for a specific year.
    
    Args:
        column : The name of the column to be sorted.
        year : The year used to filter.
        n : the number of rows to find
        
    Returns:
        A list of dictionaries where the dictionaries contain the rank and country.
    """
    return happiness_manager.get_top_n_countries(column, year, n)
    
# --- Task 2: Build the Agent ---
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

# Did get some help from AI to reword/rework the system prompt to ensure happiness data df was being used.  Originally agent kept trying to use the returned metadata.  Then realized agent couldn't see the happiness data df that was created in the HappinessDataManager.
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

DATA ARCHITECTURE RULES (VERY IMPORTANT):
- The actual dataset is stored in a global pandas DataFrame: happiness_manager.df.
- The tool load_happiness_data() ONLY loads data and returns a dictionary with metadata.
- Tool outputs are NOT DataFrames and must NEVER be used for pandas operations.
- Never call .groupby(), .shape(), .columns(), or any pandas method on tool outputs.
- All analysis and plotting must use happiness_manager.df directly.
- Never use a variable named df unless you explicitly create it from happiness_manager.df.

TOOL USAGE RULES:
- Always call load_happiness_data() first if df is not already loaded.
- Use tools for simple tasks (loading, summaries, correlations, rankings).
- Use Python code ONLY when tools are insufficient (especially for plotting or custom aggregation).

PLOTTING RULES (CRITICAL):
- Always use df for plotting, never tool outputs.
- If plotting trends over time:
  - Ensure 'year' column is treated as integer (convert if needed).
  - Sort data by year before plotting.
- If asked "one line per region":
  - Group data using:
    df.groupby(["Regional indicator", "year"])["Happiness score"].mean().reset_index()
  - Each region must be plotted as a separate line.
- Do NOT plot raw rows when a time axis is involved.
- Always call plt.savefig(...) before plt.close().
- Never use plt.show().

LEGEND RULES:
- Never place legends in the middle of the plot.
- Place legends outside the plotting area whenever possible.
- Use:
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
- Always call plt.tight_layout() before saving plots with legends.

CODE QUALITY RULES:
- Ensure no duplicate or overlapping plotting commands.
- Avoid flat lines by aggregating before plotting.
- Ensure x-axis values are sorted before plotting.

OUTPUT RULES:
- Be concise and student-friendly.
- When a tool returns statistical results (e.g., correlation, p-values), do not repeat the raw dictionary if the query requests an explanation.
- Explain the results in clear natural language.
- When a plot is created, return only a final_answer() message with the file path.
"""

agent = CodeAgent(
    tools=[load_happiness_data, 
           summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats", "numpy"],
    max_steps=8,
)

# --- Task 3: Run Guided Queries ---
queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]

def run_queries(queries : list):
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={
        "happiness_manager": happiness_manager},)
        print(response)
    
# --- Task 4: Your Own Questions ---
# Run two additional queries of your own choice. Try to make at least one of them require the agent to write code rather than just call a tool.
# My query 1
my_query_1 = "What is the correlation between GDP per capita and Happiness Score and please explain if the relationship statistically significant?"
# response_1 = agent.run(my_query_1, reset=False)
# print(response_1)
# Comment: Did this trigger tool use, code generation, or both?
# This triggers tool use and reasoning.  I used this to test my added output parameters on System Prompt.  I wanted to try and have the agent provide more of an explanation than a boolean.

# My query 2
my_query_2 = "Please show the correlation between GDP per capita and Happiness Score. Make sure to save the figure to outputs folder"
# response_2 = agent.run(my_query_2, reset=False)
# print(response_2)
# Comment: Did this trigger tool use, code generation, or both?
# This triggered both tool use and code generation.  Used the same
# question as above to find the correlation between GDP per capita and
# Happiness score and then plot a scatter plot (this is to compare to
# the plot I made in Project_01.  I also made this intentionally vague to see how well the Agent was able to follow.

my_query_3 = "Create overlapping histograms of Happiness Score for each region. Use transparency so the distributions can be compared. Save the figure to outputs/happiness_by_region_hist.png."
# This triggers both tool use and code generation.  I really wanted to see how well the agent was able to group and follow requests while still writing custom code to plot the data.

custom_queries = [my_query_1, my_query_2, my_query_3]

if __name__ == "__main__":
    run_queries(queries)
    run_queries(custom_queries)

# --- Task 5: Reflection ---
# 1. In Query 3, how did the agent communicate whether the correlation was statistically significant? Did it use the p-value correctly? What threshold did it apply?
# The agent used a boolean flag to communicate that the correlation was statistically significant:
# Final answer: {'correlation_coefficient': 0.6313, 'p_value': 0.0, 'significant': True}
# It did use the p-value correctly, however it should also be noted
# the p-value was so small that only 0.0 was saved due to formatting
# of rounding to 4 places.  Most likely the agent used the standard
# threshold of less than 0.05 depicting significance.

# 2. Did any of the agent's responses surprise you — either by being more capable than you expected, or less? Describe one specific example.
# The agent's response did not surprise me.  I was expecting it to
# struggle on plotting with the given system prompt in the rubric.
# The agent was having issues plotting the line because it was not
# able to see the df, only the meta data returned originally.  I tried
# to write a System Prompt that had clear parameters when plotting.  I
# did use AI to help tighten up what I originally had.  Once I had my
# system prompt more detailed the agent was able to create all the
# plots successfully meaning it was able to write the code itself as
# well.

# 3. What one additional tool would make this agent meaningfully more useful?  Describe what it would do and what kind of question it would help the agent answer.  (You do not need to implement it.)
# An additional tool that would make this agent better would be a data
# retrieval tool similar to having dataframe query capabilities.
# Example is being able to give specific columns and filter to
# aggregate.  So it would be a useful tool to be able to find the
# average happiness score by region or finding all the countries that
# have a specific columns above something and happiness below a
# certain score.  This would also allow the correlation to be computed
# on specific regions as opposed to the whole data set as well.