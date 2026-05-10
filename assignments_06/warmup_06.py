from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")



# ----------------------------------- RAG Concepts --------------------------------------

#Q1

"""

A. RAG would be best for this situation. With RAG, 


"""


#Q2

"""


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
    

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


simple_keyword_retrieval(query, documents, verbose=True)

# Loyalty.txt was selected. The query is looking for 'your'.


# Keyword Question 2

query = "Do you have anything without caffeine?"
simple_keyword_retrieval(query, documents, verbose=True)

"""
 No document was selected. It technically got it right. Source documents do not list anything without caffeine.
"""


# Keyword Question 3

query = "How do I sign up for rewards?"

# It will return nothing. There are no content matches

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
| What is compared?          | Exact word overlap                | ?            |
| What is retrieved?         | Full document                     | ?            |
| Can it handle synonyms?    | No                                | ?            |
| Storage format             | Plain text dictionary             | ?            |
| Relevance score            | Number of overlapping keywords    | ?            |



"""




# --------------------------------- LlamaIndex -----------------------------------



SimpleDirectoryReader("brightleaf_pdfs")