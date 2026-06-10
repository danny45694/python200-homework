from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# Step 1: Setup
from pathlib import Path
docs_dir = Path("groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# Step 2: Load the Documents


docs = SimpleDirectoryReader("groundwork_docs", filename_as_id= True).load_data()
index = VectorStoreIndex.from_documents(docs)

print(docs[0])
# Step 3: Build the Index and Query Engine

query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready for answer questions")



# Step 4: Query the Assistant

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    

    for node_with_score in response.source_nodes:
        file_name = node_with_score.node.metadata.get("file_name")
        print(f"Source Document: {file_name}")
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
        print("-" * 30)

# The assistant sounds confident and self assured. None of the answers surprised me.

# Step 5: Find a Failure

questions = [
    "What drink is good for diabetics?",
    "Can I book Groundwork Coffee for a wedding with 150 guests, and how much would it cost?"
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        file_name = node_with_score.node.metadata.get("file_name")
        print(f"Source Document: {file_name}")
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)
        


        # Text Snippet is coming out as gibberish. Tried this to fix.
        #text = node_with_score.node.get_content()
        #clean_text = " ".join(text.split())

"""

I asked the model 2 different questions. For the wedding question, if it offered pricing without evidence, it hallucinated. In my case, it answered correctly, asking the user to contact Groundwork directly.

For the diabetes question, it suggested an americano. I looked up an americano and those are typically made without sugar. 

"""

# Step 6: Reflection

"""
1. Creating the LlamaIndex implementation took maybe 10 lines at most. It is very efficient and helpful using a framework.

2. In my current role as a procurement specialist, it would be very helpful in pulling price information from different invoices and providing me that cost information. In addition, I can set it up to alert me when there are funny charges like a recent HDPE surcharge we got. 

3. Hallucinations. It continues to be one of the biggest reasons human oversight is required for implementation and use of AI.

"""
