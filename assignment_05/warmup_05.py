from dotenv import load_dotenv
from openai import OpenAI

if load_dotenv():
    print("Successfully loaded api key")
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

# The Chat Completions API
# --- API Question 1 ---
# Set up your OpenAI client and make your first chat completion call. Use the model "gpt-4o-mini" and send this prompt: "What is one thing that makes Python a good language for beginners?". Print the model's response.

# Print just the text of the response (not the whole object). Then print the name of the model that responded and the total number of tokens used. Label each output.

print(f"The model's response : {response.choices[0].message.content}")
print(f"The name of the model : {response.model}")
print(f"Total number of tokens used : {response.usage}")

# --- API Question 2 ---
# Run the same prompt three times with three different temperature settings: 0, 0.7, and 1.5. Print each response, labeled with its temperature.

# What do you notice about how the outputs differ? Which temperature would you use if you needed a consistent, reproducible output?
prompt = "Suggest a creative name for a data engineering consultancy."
# temperature controls the level of randomness and creativity
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role" : "user", "content" : prompt}], temperature = temperature)
    print(f"The model's response with temperature {temperature} : {response.choices[0].message.content}")
    
# The outputs differ that some are just the suggestion and others are are as though it is trying to have a conversation.  I feel like 0 gives the most consistent outputs.  Testing this out a few times, 0.7 would sometimes give just the suggestion and other times give a conversational output.

# --- API Question 3 ---
# Use n=3 with temperature=1.0 to get three different completions in a single API call. Print all three
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for idx, choice in enumerate(response.choices, start = 1):
    print(f"The model's response {idx} : {choice.message.content}")
    
# --- API Question 4 ---
# Set max_tokens=15 and send a prompt that would normally produce a long response (for example, "Explain how neural networks work."). Print the result. Add a comment: What happened, and why might you want to use max_tokens in a real application?
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print(response.choices[0].message.content)

# The response stopped mid sentence because it ran out of tokens.  We would want to limit the tokens in a real application to keep answers short and concise.  Setting max_tokens can also help limit the cost and prevent more usage than expected.

# System Messages and Personas
# --- System Question 1 ---
# Use a system message to give the model a personality, then ask it a question. Print the response.
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(f"Patient and encouraging model's response : {response.choices[0].message.content}")

# Now change the system message to give the model a completely different personality (your choice) and ask the same question. Print that response too. Add a comment noting what changed.
messages = [
    {"role": "system", "content": "You are a burned out and out of patience. You try to stay patient but are short and sarcastic, but then feel bad at the end."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(f"Burned out and sarcastic model's response : {response.choices[0].message.content}")

# The tone of the response as well as the length.  The first one was a lot more in depth while my burned out and sarcastic model was more concise.

# --- System Question 2 ---
# The completions API is stateless — it has no memory of previous calls. The way to give a model context is to pass the conversation history yourself as a list of messages.
# Build the following conversation manually (no loop, no user input — just construct the list) and send it in a single API call.
# Print the response.  Why does the model know Jordan's name, even though it's stateless?
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)

# The model know's Jordan's name because it was included in the conversation history sent within the same API call.  The model is stateless, so it doesn't have any memory between requests.  It used all the messages provided in the single call to generate the response.

# Prompt Engineering
# --- Prompt Question 1 — Zero-Shot ---
# Ask the model to classify the sentiment of each review below as positive, negative, or mixed. Give it no examples — just the task description and the reviews. Print each result labeled with the review number.

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""Please classify the sentiment of each provided review as positive, negative, or mixed.

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}

Return each on labeled.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content":prompt}]
)

print(f"Prompt without Example:\n{response.choices[0].message.content}")

# --- Prompt Question 2 — One-Shot ---
# Repeat the same task, but this time add one example before the reviews to show the model the format you want.
# Print the results. Did adding one example change the format or consistency of the output compared to Q1?
prompt = f"""Please classify the sentiment of each provided review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}

Return each on labeled.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content":prompt}]
)

print(f"Prompt with 1 Example:\n{response.choices[0].message.content}")

# The format was changed after the example was given.  The format matched the example format I provided.  The model reiterated the review and then provided the sentiment below it.  The model's sentiments however did not alter and remained consisten.

# --- Prompt Question 3 — Few-Shot ---
# Repeat the task again, this time with three examples. At least one example should be positive, one negative, and one mixed. Print the results. Add a comment comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?

prompt = f"""Please classify the sentiment of each provided review as positive, negative, or mixed.

Example:
Review: "Great customer support, they were very knowledgable."
Sentiment: positive

Review: "Item did not work on arrival and the seller did nothing to resolve the issue."
Sentiment: negative

Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}

Return each on labeled.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content":prompt}]
)

print(f"Prompt with 3 Examples:\n{response.choices[0].message.content}")

# The zero-shot results were consistent to the one-shot and three-shot results.  I would use zero-shot when the task is clear and simple and we are confident the model already understands it.  If I want a specific format I would use one-shot or three-shot.  One-shot would be useful when the task is not as clear and an example helps clarify what I am looking for.  Three-shot would be used when I wasnt to ensure high consistency with examples of all outcomes.  It is also useful in teaching the model patterns to look for.

# --- Prompt Question 4 — Chain of Thought ---
# Ask the model to solve the following problem, but instruct it to show its reasoning step by step before giving a final answer. Label the final answer clearly.

prompt = """Please help me answer the following question.  Explain what you did for each step and why.  Please label the final answer clearly at the end.

Example:
Problem: 4(2 + 2) - 3
Step 1: I added 2 + 2 to find the sum of 4.
Step 2: I multiplied 4 * 4 for the product 16 because of the parenthesis.
Step 3: I subtracted 3 from 16 to find the difference of 13.
Final Answer: 13

A data engineer earns $85,000 per year.  She gets a 12% raise, then 6 months later takes a new job that pays $7,500 more per year than her post-raise salary.  What is her final annual salary?"""

# Print the full response including the reasoning. Add a comment: Why does asking the model to reason step by step tend to improve accuracy on problems like this?

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content":prompt}]
)

print(f"Prompt with Chain of Thought:\n{response.choices[0].message.content}")

# It helps improve accuracy because it needs to follow the step from before and it will help limit hallucinations?  The prompt requires the model to break down what it is doing instead of jumping straight to an answer.  This also allows the user to validate the work the model does to confirm the accuracy.

# --- Prompt Question 5 — Structured Output ---
# Ask the model to analyze the review below and return the result only as valid JSON with keys sentiment, confidence (a float from 0 to 1), and reason (one sentence). Print the raw response, then parse it with json.loads() and print each field separately, labeled.

import json

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

prompt = f"""Please analyze the review provided and return the result as a JSON with the following keys.  Sentiment should be if the review is positive, negative, or mixed.  Confidence is a float from 0 to 1, and reason is one sentence as to why you chose the sentiment.

review: {review}
keys: sentiment, confidence, and reason
"""

# Add a try/except block to handle the case where the response is not valid JSON. If it fails, print the raw response so you can debug the prompt.

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content":prompt}]
)

raw_response = response.choices[0].message.content

try:
    print(f"Raw Response : {raw_response}")
    clean_response = raw_response.replace("```", "").replace("json", "")
    parsed_response= json.loads(clean_response)
    print(f"JSON successfully parsed")
    for k, v in parsed_response.items():
        print(f"{k} : {v}")
    
except Exception as e:
    print(f"Raw Response : {raw_response}")
    print(f"Error : {e}")
    
# --- Prompt Question 6 — Delimiters ---
# Use triple backticks as delimiters to clearly separate the user's text from your instructions. Send the prompt below and print the result.

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content":prompt}]
)

print(f"Prompt with Instructions :\n{response.choices[0].message.content}")

# Then send a second prompt using a passage that is not a set of instructions (any sentence or two of regular prose). Confirm that the model returns "No steps provided." Add a comment: What problem do delimiters help prevent?

user_text = "I can't even think of a sentence to provide.  So I am just writing this."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user", "content":prompt}]
)

print(f"Prompt without Instructions :\n{response.choices[0].message.content}")

# It helps make it clear that the information between the deliminators is just data to process, not instructions to follow.

# --- Ollama Question 1 ---
# In your terminal, run the following prompt using Ollama and then run the same prompt using the OpenAI API in Python.
# Paste the Ollama output as a multi-line string comment in your code. Then add another comment answering: What differences did you notice between the two responses? What is one advantage and one disadvantage of running a model locally?
prompt = "Explain what a large language model is in two sentences."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content":prompt}]
)

print(f"ChatGPT response : {response.choices[0].message.content}")

# Ollama Response
# Thinking...
# Okay, the user wants me to explain a large language model in two sentences. Let me start by recalling what I know about them. Large language models, like BERT, are neural networks that can understand and generate text. They process vast amounts of text, so they can understand context and learn from a lot of data. That's the first sentence. Now, the second sentence needs to elaborate. Maybe mention their ability to it concise and highlight their key features. Let me check if I'm not missing anything, but I think that's all.
# ...done thinking.

# A large language model is a type of artificial intelligence that can understand and generate human-like text, learning from vast amounts of data to improve its understanding and performance. It processes and analyzes large volumes of information, enabling it to understand context, learn from diverse datasets, and perform tasks like translation or writing with high accuracy.

# Chatgpt Response
# A large language model (LLM) is an artificial intelligence system designed to understand, generate, and manipulate human language by utilizing vast amounts of textual data. These models use deep learning techniques, particularly neural networks, to learn patterns and structures in language, allowing them to produce coherent and contextually relevant responses.

# The local model is less technical and more general. Ollama focuses on what AI does while Chatgpt focuses on how it works. An advantage of running a local model is it provides more privacy and control.  A disadvantage is that it can require a strong setup and hardware.  It can also be slower and less capable than the larger cloud models.