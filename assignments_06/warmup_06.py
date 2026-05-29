from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------- RAG Concepts --------------------------------------

# Concepts Q1



concepts_q1 = """

Scene A: 
RAG would be best for this situation. With an ever changing database, RAG is much easier to implement and change each time. Fine-tuning is too expensive for something that will change in 3 months and prompt engineering will blow the context window.

Scene B: 
This would require fine-tuning. To achieve a certain voice, you need to provide the AI many examples of the kind of language and writing style it must possess.

Scene C:
Simple prompt engineering would work here. It is faster to paste the documents into the prompt than to setup code and implement RAG for it. We need not say more about fine-tuning.

"""


# Concepts Q2

concepts_q2 = """

The reason confidently wrong answers are more harmful than saying "I am not sure" is because humans are attracted to confidence. Something may not be particularly good for us, but if you say it with such confidence, you can get someone to believe in your answer. The problem is AI does not grasp logic and unable to take accountability. 

Asking AI something like health advice can have very bad consequences because it will not grasp all the variables and it can hallucinate. The web surfing capability does not help either as there is a vast amount of misinformation on the internet already. 

"""

# Concepts Q3


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

steps explained in order:

Text extraction - AI reviews the documentation and extracts text from it

Split text - AI breaks down the text into smaller portions. Allows efficient search and retrieval.

Convert text - The Chunks are transformed into vector embeddings (list of numbers unique to the chunk)

Receive User query - Get user's query

Embed user query - transform user query into vector embedding.

Retrieve relevant chunks - similarity score is used to retrieve the context based on number of word matches. 

Inject retrieved chunks - AI then passes the relevant chunks back to the llm so it can generate a response.

Generate a response - LLM reviews the underlying information and then responds to the users query/prompt.

"""


# --------------------------------- Keyword RAG -------------------------------------

import string


def simple_keyword_retrieval(query, documents, verbose=True):
    #Keyword retrieval using token overlap scoring.
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
    

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


simple_keyword_retrieval(query, documents, verbose=True)

# Loyalty.txt was selected. The query is looking for 'your'.

"""
AI output

Query tokens (filtered): ['hours', 'weekend', 'what', 'your']
[menu.txt] overlap=0 -> []
[hours.txt] overlap=0 -> []
[hiring.txt] overlap=1 -> ['your']
[loyalty.txt] overlap=1 -> ['your']

Selected best match: loyalty.txt

"""


# Keyword Question 2

query = "Do you have anything without caffeine?"
simple_keyword_retrieval(query, documents, verbose=True)

print(simple_keyword_retrieval)


"""

 No document was selected. 
 A keyword rag system would fail here.
 Semantic would work far better here. Converting the text into its semantic meaning and searching for that allows it to return relevant information the user is looking for.

"""

# Keyword Question 3

query = "How do I sign up for rewards?"

"""
It will return nothing. Keyword rag will fail here. If it does return something, it will likely be on the non-essential words in the query like do and for.
The phrase search is exact. Does not derive meaning, just word text.

Prediction was correct. Since the documentation does not have the exact wording, it found no overlap and returned nothing.

"""


simple_keyword_retrieval(query, documents, verbose=True)

# I was right. No keywords matches so the model returned nothing.



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
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader
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

# The retrieve chunks are relevant to the question asked.
# The model sounds very confident and specific
# I don't see anything unexpected.


# Llamaindex Question 2

index = VectorStoreIndex.from_documents(docs)

for i in [1, 5]:
    print(f"\n=== similarity_top_k={i} ===")

    query_engine = index.as_query_engine(similarity_top_k=i)

    for q in questions:
        print(f"\nQ: {q}")

        response = query_engine.query(q)
        print("A:", response)

        for node_with_score in response.source_nodes:
            print(f"Node ID: {node_with_score.node.node_id}")
            print(f"Similarity Score: {node_with_score.score:.4f}")
            print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
            print("-" * 30)

# Responses seem the same overall. The only difference is K=5 produced more technical detail in its output.
        


# Llamaindex Question 3


questions = [
    "What is the motivation of paying brightleaf employees low wages if you really value their impact and cooperation?"
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

# I kind of expected the AI to upsell the benefits provided by the company or the low-pay in lieu of advancement opportunities. Instead, the AI said it pays the employee fair wages.

#Llamaindex Question 4

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

#Get response to query
queries = ["What employee benefits does BrightLeaf offer?",
           "What impact does BrightLeaf have the economy and overall political climate?"]
for q in queries:
    response = query_engine.query(q)

    # Evaluate faithfulness and relevancy
    faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
    print("Faithfulness Evaluation: " + str(faithfulness_result.score))

    relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
    print("Relevancy Result: " + str(relevancy_result.score))


#Faithfulness score of 1 means the AI referred the data almost completely, if not entirely, from the source material.

# Relevancy measures how well an AI's response directly answers the user's specific question.

# The scores did not change between the 2 queries. 

# Uses a capable LLM to act as a judge the output of another AI application.

