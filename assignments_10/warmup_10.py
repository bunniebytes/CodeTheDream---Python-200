# LLMs as Transform
# --- LLMs as Transform Question 1 ---
# For each task below, write a one-sentence comment saying whether you would use an LLM or deterministic code, and why.

#     Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".
# Deterministic code - Parsing a date is predictable and has clear rules and answers.

#     Classify a customer support ticket -- "my card was charged twice" -- into one of: billing, technical, or general.
# LLM - An LLM can understand the meaning and context of freeform text and classify it correctly even when phrased in different ways.

#     Calculate the average of a list of numbers.
# Deterministic code - This is an exact mathematical calculation and results.

#     Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp (contract)".
# LLM - Job titles can appear in many formats, and an LLM is flexible at identifying and extracting the company name from unstructured text.

#     Determine whether a product review is more than 100 words long.
# Deterministic code - Counting the amount of words is a straightforward operation with an objective answer.

# --- LLMs as Transform Question 2 --- The prompt produces
# unstructured output, so the number of sentences, wording, and format
# may vary between responses. This makes it difficult for downstream
# systems to reliably parse, validate, and store the data. Using a
# structured format such as JSON provides a consistent schema, making
# parsing predictable and reducing failures in automated pipelines.

# --- LLMs as Transform Question 3 ---
# Sequential processing would take approximately 50,000 seconds.
# 50,000 seconds / 60 = 833.33 minutes
# 833.33 minutes / 60 = 13.89 hours

# One practical strategy would be to process requests in parallel
# using asynchronous workers or a queue system. This allows multiple
# classification calls to run at the same time and greatly reduces the
# total processing time.

# Azure OpenAI
# --- Azure OpenAI Question 1 ---
# In a comment block, name two reasons an organization might use Azure OpenAI instead of calling the OpenAI API directly. Be specific -- "it's better" is not an answer.
# One reason an organization might use Azure OpenAI is that it can
# integrate with existing Azure services, making it easier to manage
# security, networking, and access controls in one environment.

# Another reason is compliance and data residency requirements.
# Organizations may need their data processed in specific regions or
# under certain regulatory standards that Azure can help support.

# --- Azure OpenAI Question 2 ---
# When you switch from OpenAI to AzureOpenAI, the client initialization takes three Azure-specific parameters. In a comment block, name them and describe what each one is. (Do not include the standard api_key -- describe the Azure-specific ones.)
# azure_endpoint - The URL of the Azure OpenAI resource that the
# application will send requests to.

# api_version - The version of the Azure OpenAI API being used. Azure
# requires an explicit API version for requests.

# azure_deployment - The name of the model deployment created in
# Azure. Requests are sent to this deployment instead of directly
# specifying a model name.

# --- Azure OpenAI Question 3 ---
# In a comment block, answer: when using AzureOpenAI, the model parameter in chat.completions.create() does not take a value like "gpt-4o-mini". What does it take instead, and where do you find the right value to use?
# When using AzureOpenAI, the model parameter takes the name of your
# Azure deployment rather than a model name like "gpt-4o-mini".

# The correct value can be found in the Azure Portal under your Azure
# OpenAI resource's Deployments section, where you create and manage
# model deployments.