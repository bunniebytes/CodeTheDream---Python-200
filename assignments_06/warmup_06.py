from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
# RAG Concepts

# --- Concepts Question 1 ---
# prompt engineering, fine-tuning, or RAG
# Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
# The best approach would be RAG (Retrieval-Augmented Generation).
# Since this is a legal team dealing with internal information, the
# team will be able to ask the systen a question and the system will
# search the PDFs.  Because the documents are updated every quarter,
# using this method will ensure that the team receives the most up to
# date data.

# Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. They have 3,000 examples their in-house writers produced over the years.
# The best approach would be fine-tuning.  This is the best because
# the voice they are looking for is not common and they want their
# model to procure this voice for every product copy.  They can use
# all the examples that their in-house writers have produced to train
# it.

# Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.
# The approach that would be best for this scenario is prompt
# engineering because she only wants the information from one specific
# document.  She is not looking to change the behavior of the agent
# for any other documents.

# --- Concepts Question 2 ---
# Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where a confident hallucination could cause harm.
# Think about the tone of the response as well as its content — why does the way the model expresses an answer affect how much we trust it?
# When an agent hallucinates and is confidently wrong it can cause a
# lot of harm.  There are people who go to LLM's for medical advice,
# such as can this medication be taken with this medication.  This can
# cause issues when the agent sounds super confident and people take
# the advice at face value.  They take the medications together and
# serious side effects happen.  When the tone is confident and says
# "Yes you can", instead of reminding the user to check with an actual
# medical professional; people are more likely to trust it.

# --- Concepts Question 3 ---
# The steps below make up a complete RAG pipeline, but they are out of order. Copy the list into your code as a comment, arrange them in the correct order, and add a one-sentence description of what happens at each step.
# steps = [
#     "Receive the user's query",
#     "Embed the user's query",
#     "Extract text from source documents",
#     "Split text into chunks",
#     "Convert text chunks into embeddings",
#     "Retrieve the most relevant chunks",
#     "Inject retrieved chunks into the prompt",
#     "Generate a response from the LLM",
# ]

# Receive the user's query - System receives the question or request
# Embed the user's query - User's query is converted to a numerical vector representation
# Extract text from source documents - System pulls raw text from files like PDFs or databases
# Split text into chunks - Data is split into smaller chunks for the system to process and retrieve efficiently
# Convert text chunks into embeddings - Text chunks are converted into numerical vector representation that captures its meaning
# Retrieve the most relevant chunks - System searches the vector database for chunks whose embeddings most closely matches the user's query embedding
# Inject retrieved chunks into the prompt - The retrieved text is injected into the prompt as information
# Generate a response from the LLM - The model uses the prompt and the retrieved data to product an answer

# Keyword RAG
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# --- Keyword Question 1 ---
# Run simple_keyword_retrieval with verbose=True on the query and documents below. Print the name of the selected document.

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

selected_document = simple_keyword_retrieval(query, documents)
print(f"The best selected document : {selected_document}")

# The best selected document with the code provided is loyalty.txt
# (but this doesn't seem right as it should be hours.txt).  Looking at
# the code the reason why is the content word in hours is weekends
# while the query has weekend.  The keywords also are not checked in
# the title of the files.

# --- Keyword Question 2 ---
query = "Do you have anything without caffeine?"

selected_document = simple_keyword_retrieval(query, documents)
print(f"The best selected document : {selected_document}")

# No documents were selected.  Keyword RAG did not get this correct
# because none of the documents mention caffeine or have the other
# words in the query, it only checks for token overlaps not related
# concepts.  An embedding based retrieval would be better because
# embeddings capture semantic similarity.

# --- Keyword Question 3 ---
query = "How do I sign up for rewards?"

# I do not believe it will find an overlap.  The query itself would have the words "sign" and "rewards" and looking at the documents there are none that overlap.

selected_document = simple_keyword_retrieval(query, documents)
print(f"The best selected document : {selected_document}")

# My prediction was correct.

# Semantic RAG Concepts

# --- Semantic Question 1 ---
# What is a vector embedding? (1-2 sentences)
# Vector embedding is a numerical representation of text that capture
# semantic meaning.  Texts with similar meanings end up with vectors
# that are close together in vector space.

# Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
# The chunk with the cosine similarity score of 0.85 is more relevant
# because higher cosine similarity means the vectors are more closely
# aligned semantically.  The closer the score is to 1 means the texts
# are similar in meaning, while a score closer to 0 indicates weak
# semantic similarity.

# Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
# They can find a relevant chunk because embedding captures meaning
# and context not just the words.  An example from the previous
# questions would be "weekend hours" and "Saturday and Sunday
# schedule" because their embeddings would represent similar concepts.

# --- Semantic Question 2 ---
# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Vector/embedding similarity between meanings   |
# | What is retrieved?         | Full document                     | Semantically relevant chunks                   |
# | Can it handle synonyms?    | No                                | Yes                                            |
# | Storage format             | Plain text dictionary             | Vector embeddings in a vector database/index   |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity (vector similarity score)    |

# LlamaIndex
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
# from llama_index.readers.file import PyMuPDFReader
# from pathlib import Path

# --- LlamaIndex Question 1 ---
# Load documents directly from PDFs in the folder
doc = SimpleDirectoryReader("./brightleaf_pdfs").load_data()

# This loads the PDFs properly
# loader = PyMuPDFReader()
# docs = []

# pdf_folder = Path("./brightleaf_pdfs")

# for pdf_file in pdf_folder.glob("*.pdf"):
#     file_docs = loader.load(file_path=str(pdf_file))
#     docs.extend(file_docs)
# for doc in docs:
#     print(doc.text[:1000])

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(doc)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print(f"\nQuestion: {question}")
    response = query_engine.query(question)
    print("Answer:", response)
    
    print("\nRETRIEVED SOURCE NODES:")
    for i, node in enumerate(response.source_nodes, start=1):
        score = round(node.score, 4) if node.score else "N/A"
        preview = node.text[:150].replace("\n", " ")

        print(f"\nNode {i}")
        print(f"Similarity Score: {score}")
        print(f"Chunk Preview: {preview}...")
        print("-" * 30)
        
# Do the retrieved chunks look relevant to the question?
# The retrieved chunks are very relevant to the question.  (I
# installed llama_index.readers.file because I was getting badly
# extracted data)

# Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.
# The model's response sounds very confident and specific

# Did anything unexpected get retrieved?
# Originally I was receiving previews like
# "@,,bD<:t;XK\SnFVa-N.'+]]mG][knfN:Nf+n^5pD]VbeI1<@0t=:Ek/89j3FeVPn!";#N1GY8!CAdR/2]D93hd&ohPt+qggPXk+"PsGOj/+-6QG1U'>_OGih@LLtBf7%UY?!94]!LrZ#ce_r\K1U..."
# but I trouble shot and resolved the issue.

# --- LlamaIndex Question 2 ---
# Re-run one of the queries from Q1 twice: once with similarity_top_k=1 and once with similarity_top_k=5. Print the response and source node scores for both runs.

# Add a comment explaining how the response changed (if at all) and whether more retrieved context is always better

def use_query_engine(query, k = 3):
    query_engine = index.as_query_engine(similarity_top_k=k)

    for question in questions:
        print(f"\nQuestion: {question}")
        response = query_engine.query(question)
        print("Answer:", response)
        
        print("\nRETRIEVED SOURCE NODES:")
        for i, node in enumerate(response.source_nodes, start=1):
            score = round(node.score, 4) if node.score else "N/A"
            preview = node.text[:150].replace("\n", " ")

            print(f"\nNode {i}")
            print(f"Similarity Score: {score}")
            print(f"Chunk Preview: {preview}...")
            print("-" * 30)
            
use_query_engine(questions, 1)
use_query_engine(questions, 5)

# The responses changed a little with k = 5 providing a bit more
# information than k = 1.  The overall information was on the same
# subject, and k = 5 had all the information k = 1 had and more.  More
# retrieved content will provide more information usually it seems.

# --- LlamaIndex Question 3 ---
# Try a query you think the pipeline might struggle with — something vague, something that spans multiple documents, or something where the information might not be in the documents at all. Print the response and all retrieved chunks.

questions = ["Is Brightleaf a good company to work for long term?"]

use_query_engine(questions)

# Add a comment explaining what you expected, what actually happened, and what you would change about the system to handle this kind of query better.
# I expected the system to struggle with this question because it is
# very subjective.  It would require taking some information from
# multiple documents and analyzing them to determine if it was a good
# place to work.  The system sounded very confident in its answer to
# if the company is a good place to work for long term, which again is
# a subjective topic.  I would put up guards so that the model can
# differentiate between questions answered with evidence from
# documents and subjective conclusions/inference.

# --- LlamaIndex Question 4 ---
# Import and instantiate a FaithfulnessEvaluator and a RelevancyEvaluator, both using gpt-4o-mini as the judge LLM (refer to the "RAG Evaluation using LlamaIndex" section of lesson 4 for the exact import and setup pattern). Run them on this query.

# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

def get_faithfulness_relevancy_result(q):
    response = query_engine.query(q)

    # Evaluate faithfulness and relevancy
    faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
    print("Faithfulness Evaluation: " + str(faithfulness_result.score))

    relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
    print("Relevancy Result: " + str(relevancy_result.score))

q = "What employee benefits does BrightLeaf offer?"
get_faithfulness_relevancy_result(q)

q = "What programming language does BrightLeaf use for its mobile app backend?"
get_faithfulness_relevancy_result(q)

# What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
# A faithfulness score of 1.0 means that the answer is fully supported
# by retrieved context while 0.0 means the answer was not found in the
# documents at all.  The model most likely made up the data or the
# answer provided is completely irrelevant.

# What does a relevancy score measure, and how is it different from faithfulness?
# A relevancy score is the measurement of how well the retrieved
# documents match the user's query.  Faithfulness is if the model
# stuck to the retrieved data while relevancy is if the information is
# even correct.

# Did the scores change between your two queries? If so, why do you think that happened?
# The scores did change, the second question regarding programming
# language was not a piece of information that is available in the
# documents so relevancy and faithfulness were both 0.0 as opposed to
# the question about benefits which were both a score of 1.0.

# What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
# LLM-as-a-judge approach is using another language model to evaluate
# the outputs instead of having metrics and parameters hard coded.  It
# asks the LLM if the answer is supported by the context as opposed to
# having to have exact match or keyword overlap.  Since RAG outputs
# are not standardized answers and require reasoning or summarization.
# RAG doesn't work well with traditional metrics like precision or
# recall.  LLM judges are used because they can evaluate reasoning
# quality as well as if the answer is partially correct.
