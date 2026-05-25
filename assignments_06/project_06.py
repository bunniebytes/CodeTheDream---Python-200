import os

from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# --- Step 1: Setup ---
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("./groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# --- Step 2: Load the Documents ---
# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("./groundwork_docs").load_data()

# How many documents were loaded
num_docs = len(docs)
print(f"Number of Documents : {num_docs}")

# The file name of each document
doc_names = [doc.metadata["file_name"] for doc in docs]
print(f"Document Names : {', '.join(doc_names)}")

# --- Step 3: Build the Index and Query Engine ---
index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k = 3)

print("Index built successfully. Ready to answer questions.")

# --- Step 4: Query the Assistant ---
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

def get_answers_and_source_nodes(questions):
	for question in questions:
		print(f"Question : {question}")
		response = query_engine.query(question)
		print(f"Answer : {response}")
		
		print("\nRETRIEVED SOURCE NODES:")
		for i, node in enumerate(response.source_nodes, start=1):
			score = round(node.score, 4) if node.score else "N/A"
			preview = node.text[:200].replace("\n", " ")

			print(f"\nNode {i}")
			print(f"Similarity Score: {score}")
			print(f"Chunk Preview: {preview}...")
			print("-" * 30)
   
get_answers_and_source_nodes(questions)
    
# Did the assistant sound confident and accurate? Did any of the answers surprise you?

# The assistant sounded confident and accurate.  The answers for hours, milk options, loyalty, and origin story are clearly grounded and detailed.  The catering/wholesale answer actually surprised me.  I found that compared to the
# other answers it was a lot more vague and minimal.  It answered the question
# directly, but I would have thought it would have said to contact Groundwork
# for a quote.  This likely means that the retrieved context did not contain
# detailed information so the model defaulted to the short statement rather than
# expanding on it's answer.  Overall the system performs well when relevant
# context is available and did not hallucinate when the retrieved documents did
# not provide an adequate answer.

# --- Step 5: Find a Failure ---
questions = ["When is it the busiest time to go to Groundwork?",
             "What was last season's top seller?",
             "What is this season's most popular drink?",
             "How long is the wait usually at 9am on a Saturday?"
             ]

get_answers_and_source_nodes(questions)

# What you asked and why you expected it to be hard
# I asked questions that were not readily available in the documents like most popular drinks or busiest times.  I expected it to be hard because this information is not posted in the documents.

# What went wrong — wrong retrieval, missing information, the model guessed anyway?
# The model guessed anyways.

# When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting
# AI-generated responses?

# The model still sounded so confident.  It suggests that AI-generated responses
# may not be correct even though the tone sounds confident and is recommended to
# always double check their answers. What you would change about the system to
# improve it. I would improve the system by requiring evidence based answers.  A
# lot of the failed answers were short and brief, thus signalling the model made
# up the answer but did not expand on it. I also think that if the model can not
# find the answer in the documents to have it respond that the documents do not
# have the information.  I would also improve it so that the tone is not so
# confident when answering below a certain threshold, such as 0.85.

# The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using a framework?

# The LlamaIndex implementation took fewer than five lines of code in my entire
# project. This demonstrates the value of using a framework because it provides
# pre-built components that significantly speed up development. Frameworks also
# help enforce standardized coding practices, improve maintainability, and can
# enhance security.

# You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.

# This approach would be extremely beneficial for companies that manage
# thousands of documents and must follow regulations that are frequently
# updated. Examples include law firms and medical offices. A RAG-based system
# would allow employees to quickly retrieve accurate information from internal
# documents while also improving security, since sensitive documents would not
# need to be uploaded into a public AI model.# 

# What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?

# One failure mode that RAG (Retrieval-Augmented Generation) cannot fully
# prevent is hallucination or incorrect reasoning, even when the correct
# documents are retrieved. Although the system may retrieve relevant
# information, the language model can still misinterpret the content or make
# incorrect assumptions while generating a response. In addition, if the source
# documents are outdated, incomplete, or inaccurate, the model may still provide
# incorrect answers because it relies on the information it was given.
