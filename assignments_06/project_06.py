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


docs = SimpleDirectoryReader("groundwork_docs").load_data()
index = VectorStoreIndex.from_documents(docs)
print(docs)
print(len(docs))


# Step 3: Build the Index and Query Engine

"""
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready for answer questions")

"""

"""
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
        print("-" * 30)
        


        # Text Snippet is coming out as gibberish. Tried this to fix.
        #text = node_with_score.node.get_content()
        #clean_text = " ".join(text.split())

        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {clean_text[:150]}...")

        print(node_with_score.node.metadata)
"""



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
        print(f"Source Document: {file_name}")
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)



# Step 5: Find a Failure

questions = [
    "How did Groundwork's wholesale program cause their hours on weekend to change?"
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
        print("-" * 30)
        


        # Text Snippet is coming out as gibberish. Tried this to fix.
        #text = node_with_score.node.get_content()
        #clean_text = " ".join(text.split())

        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {clean_text[:150]}...")

        print(node_with_score.node.metadata)


# Step 6: Reflection

"""
Step 6: Reflection
Add a comment block at the end of project_06.py answering the following:

The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using a framework?

You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.

What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?


"""


"""

Step 6 comments here. 

"""