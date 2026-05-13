from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------- RAG Concepts --------------------------------------

#Q1

"""

Scene A: 
RAG would be best for this situation. With an ever changing database, RAG is much easier to implement and change each time. Fine-tuning is too expensive for something that will change in 3 months and prompt engineering will blow the context window.

Scene B: 
This would require fine-tuning. To achieve a certain voice, you need to provide the AI many examples of the kind of language and writing style it must possess.

Scene C:
Simple prompt engineering would work here. It is faster to paste the documents into the prompt than to setup code and implement RAG for it. We need not say more about fine-tuning.

"""


#Q2

"""

AI hallucinations can have very drastic consequences. Because they produce output with so much confidence, it is easier for a user to take the output at face value vs doing their own research. One recent example is Delotte. They provided a report where the AI cited false citations. The Canadian government spent around $1.6 million on the report. No bueno

"""


#Q3

"""
steps = [
    "Extract text from source documents",
    "Split text into chunks",
    "Convert text chunks into embeddings",
    "Receive the user's query",
    "Embed the user's query",
    "Retrieve the most relevant chunks",
    "Inject retrieved chunks into the prompt",
    "Generate a response from the LLM",
]

"""



# --------------------------------- Keyword RAG -------------------------------------

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
    

#query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


#simple_keyword_retrieval(query, documents, verbose=True)

# Loyalty.txt was selected. The query is looking for 'your'.


# Keyword Question 2

"""
query = "Do you have anything without caffeine?"
simple_keyword_retrieval(query, documents, verbose=True)
"""



"""
 No document was selected. It technically got it right. Source documents do not list anything without caffeine.
"""


# Keyword Question 3


"""
query = "How do I sign up for rewards?"

# It will return nothing. There are no content matches

simple_keyword_retrieval(query, documents, verbose=True)

# I was right. No keywords matches so the model returned nothing.

"""

# --------------------------------- Semantic RAG Concepts -----------------------------



#Q1

"""
1. Vector embedding are essentially representations of data in a numerical format. Things such as text, images, audio etc.


2. With Cosine, the chunk with the higher score is deemed more relevant. 1 is a similar (total match), 0 means no match, -1 means it is the inverse of what you are looking for. 

3. It looks for similar/like words. With vectors, the meaning is captured. Hence someone looking for the word display may get results for tv's, projectors and other type of imaging equipment that produces an output.

"""

#Q2
"""


| Feature                    | Keyword RAG                       | Semantic RAG |
|----------------------------|-----------------------------------|--------------|
| What is compared?          | Exact word overlap              | Word meaning   |
| What is retrieved?         | Full document                   | Context chunks |
| Can it handle synonyms?    | No                              | Yes            |
| Storage format             | Plain text dictionary       |Multi-layer approach|
| Relevance score            | Number of overlapping keywords |Cosign Similarity|



"""




# --------------------------------- LlamaIndex -----------------------------------



"""
def question_query(questions):
    for q in questions:
        print(f"\nQ: {q}")
        response = query_engine.query(q)
        print("A:", response)
        print_response_details(response)

def print_response_details():
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)
"""




# LlamaIndex Question 1
import llama_index.readers.file.pymupdf 
import PyMuPDFReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI

reader = SimpleDirectoryReader(
    input_dir="brightleaf_pdfs",
    file_extractor={".pdf": PyMuPDFReader()},
)

docs = reader.load_data()

index = VectorStoreIndex.from_documents(docs)


print(type(index._vector_store).__name__)


query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
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




# Llamaindex Question 2

n = [1, 5]

for i in n:
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(similarity_top_k=i)

    for q in questions:
        response = query_engine.query(q)
        print("A:", response)
        print(f"\nQ: {q}")

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
        print("-" * 30)

        
        
"""



# Llamaindex Question 3






#Llamaindex Question 4

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

#Get response to query
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))


"""