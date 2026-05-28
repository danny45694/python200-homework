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

print(len(docs))

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
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
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

I asked the model 2 different questions. For the wedding question, if it offered pricing without evidence, it hallucinated. In my case, it answered correctly, asking the user to contact Groundwork directly. I expected it to be hard since it is a specific scenario that a general policy document would not have an answer for. 

For the diabetes question, it suggested an americano. I looked up an americano and those are typically made without sugar. I expected the question to be challenging as it is a health related question. In this case, the AI did answer. While the answer may have been correct, it should have avoided answering the question entirely as it may hallucinate or this particular coffee shop may actually add sugar to their various drinks. I would change the AI to stick strictly to the source material and advise them to contact the company for specific questions not covered in the FAQ. 

"""

"""

Q: What drink is good for diabetics?
A: The Horchata Latte would be a good option for diabetics as it is made with rice milk, cinnamon, and vanilla, and is dairy-free.
Source Document: seasonal_specials.txt
Node ID: cbc1280c-98b9-4fcb-ad4b-bb3b2ea38e06
Similarity Score: 0.7613
Text Snippet: Seasonal Specials — Current Menu

These drinks are available for a limited time only.

Iced Lavender Lemonade — $5.00
Freshly squeezed lemonade with lavender syrup and a splash of cold brew. Dair...
------------------------------
Source Document: menu.txt
Node ID: f781bb78-e851-4726-a685-5d93699e66bd
Similarity Score: 0.7220
Text Snippet: Groundwork Coffee Co. — Menu

Drinks
- Espresso (single or double): $2.50 / $3.00
- Americano: $3.00
- Latte (hot or iced): $4.50
- Cappuccino: $4.00
- Cold brew: $4.50
- Pour-over (rotating s...
------------------------------
Source Document: wholesale_catering.txt
Node ID: ed8845d0-dcaa-4c8e-99ae-c1e884bdc099
Similarity Score: 0.7057
Text Snippet: Wholesale and Catering

Wholesale Coffee
We sell our house blends and single-origin beans in bulk to local restaurants, offices, and retailers. Wholesale pricing is available for orders of 5 pounds...
------------------------------

Q: Can I book Groundwork Coffee for a wedding with 150 guests, and how much would it cost?
A: Yes, you can book Groundwork Coffee for a wedding with 150 guests. To inquire about booking catering for events, you need to email hello@groundworkcoffee.com with your event date, location, estimated guest count, and preferred package. Pricing for catering varies by package and event size, and they are happy to provide a quote based on your specific requirements.
Source Document: wholesale_catering.txt
Node ID: ed8845d0-dcaa-4c8e-99ae-c1e884bdc099
Similarity Score: 0.8245
Text Snippet: Wholesale and Catering

Wholesale Coffee
We sell our house blends and single-origin beans in bulk to local restaurants, offices, and retailers. Wholesale pricing is available for orders of 5 pounds...
------------------------------
Source Document: our_story.txt
Node ID: a5142d8d-89c9-4dec-905e-10b5f0597c2b
Similarity Score: 0.7967
Text Snippet: Our Story

Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala....
------------------------------
Source Document: menu.txt
Node ID: f781bb78-e851-4726-a685-5d93699e66bd
Similarity Score: 0.7893
Text Snippet: Groundwork Coffee Co. — Menu

Drinks
- Espresso (single or double): $2.50 / $3.00
- Americano: $3.00
- Latte (hot or iced): $4.50
- Cappuccino: $4.00
- Cold brew: $4.50
- Pour-over (rotating s...
------------------------------


"""


# Step 6: Reflection

"""
1. Creating the LlamaIndex implementation took maybe 10 lines at most. It is very efficient and helpful using a framework.

2. In my current role as a procurement specialist, it would be very helpful in pulling price information from different invoices and providing me that cost information. In addition, I can set it up to alert me when there are funny charges like a recent HDPE surcharge we got. 

3. Hallucinations. It continues to be one of the biggest reasons human oversight is required for implementation and use of AI.

"""
