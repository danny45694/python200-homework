
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)



# ------------------------------- The Chat Completions API -----------------------



#Q1 

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(response.choices[0].message.content)
print(response.usage.total_tokens)
"""




"""
#Q2

temps = [0, 0.7, 1.5]
prompt = "Suggest a creative name for a data engineering consultancy"
for i in temps:
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature = i,
    messages=[{"role": "user", "content": prompt}]
    )
    print(response.choices[0].message.content)

# Using different temperates changes the output of the AI model. To keep consistent outputs, I would use 0. This makes the model will pick the most likely next token vs introducing randomness.
"""




"""
#Q3

response = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
n=3,
temperature=1.0
)

i=0
while i < 3:
    print(response.choices[i].message.content)
    i += 1

"""



"""
#Q4

response = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "Explain how neural networks work."}],
max_tokens=15
)

print(response.choices[0].message.content)

# Setting max_tokens limits response length. When using AI in real applications, you want the model to offer short, clear answers vs research papers. Shorter, clear responses are cheaper to maintain and take up less compute.

"""

# ----------------------------- System Messages and Personas --------------------------


#Q1 


"""
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages= messages,
)

print(response.choices[0].message.content)



messages1 = [

    {"role": "system", "content": "You are a strict, no-nonsense tutor. You have expect students to adapt or they fail your class. A student better do research and have attempted the problem first"},
    {"role": "user", "content": "I don't understand what a list comprehension is."}

]

response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages= messages1,
)

print(response1.choices[0].message.content)

"""



#Q2


"""
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages= messages,
)

print(response.choices[0].message.content)


#The model knows Jordan's name because we told it. In the code above, messages serves as a record of the conversation. Each time we make a prompt or query, we refeed the information back into the model. Without this ledger, the AI would not have the context required to answer subsequent messages accurately. 
"""



# ---------------------------- Prompt engineering ---------------------------------

#Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = """

Look at each review in the label it if it is positive, negative or mixed. Print each result labeled with the review number

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

"""

"""
messages = [
    {"role": "user", "content": prompt}
]


response = client.chat.completions.create(
model="gpt-4o-mini",
messages= messages
)
print(response.choices[0].message.content)

"""

#Q2

prompt1 = """

Look at each review in the label it if it is positive, negative or mixed. Print each result. Follow the format as shown in the example below

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

"""

"""
messages = [
    {"role": "user", "content": prompt1}
]


response = client.chat.completions.create(
model="gpt-4o-mini",
messages= messages
)
print(response.choices[0].message.content)
"""

# Adding the example changes the format provided by the AI.


#Q3



#Q4

prompt_q4 = """
Solve the following problem. Solve the problem step by step, explaining your logic along the way. Label the final answer clearly.
"""

messages = [
    {"role": "user", "content": prompt_q4}
]


response = client.chat.completions.create(
model="gpt-4o-mini",
messages= messages
)
print(response.choices[0].message.content)

# Asking forces the AI to break down complex tasks into smaller, manageable parts. It is technique known as Chain-of-Thought prompting.




#Q5

import json

prompt_q5 = """

analyze the review below and return the result only as valid JSON with keys sentiment, confidence (a float from 0 to 1), and reason (one sentence). Print the raw response.


review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
"""

"""
messages = [
    {"role": "user", "content": prompt_q5}
]


response = client.chat.completions.create(
model="gpt-4o-mini",
messages= messages
)

try:
        result = json.loads(response)
        print("Parsed sentiment:", result["sentiment"])
        print("Confidence:", result["confidence"])
except json.JSONDecodeError:
        print("Error: response was not valid JSON")
"""


#6



user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt_q6 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""


messages_q6 = [
    {"role": "user", "content": prompt_q6}
]

response = client.chat.completions.create(
model="gpt-4o-mini",
messages= messages_q6
)
print(response.choices[0].message.content)

#Q6 Part 2

user_text2 = "The old oak tree stood tall in the center of the meadow, its branches swaying gently in the afternoon breeze."

prompt_part2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."
```{user_text2}```
"""

messages_pt2 = [
    {"role": "user", "content": prompt_part2}
]


response = client.chat.completions.create(
model="gpt-4o-mini",
messages= messages_pt2
)
print(response.choices[0].message.content)



# -------------------- Local Models with Ollama ------------------

"""
Thinking...
Okay, the user wants me to explain a large language model in two 
sentences. Let me start by recalling what I know. A large language 
model is a type of AI model that can understand and generate 
human-like text. I should mention its ability to learn from a vast 
amount of text to improve performance. Also, it's used in various 
applications like language translation and content creation. Need to 
make sure the sentences flow and are concise. Let me check if I 
covered all key points without being too technical. Yes, that should 
do it.
...done thinking.

A large language model is a type of AI model designed to understand 
and generate human-like text, learning from vast amounts of data to 
improve its understanding and performance. It can perform tasks like 
language translation, content creation, and problem-solving, making it 
versatile for various applications.
"""

Prompt = "Explain what a large language model is in two sentences."

messages_gpt = [
    {"role": "user", "content": Prompt}
]


response = client.chat.completions.create(
model="gpt-4o-mini",
messages= messages_gpt
)
print(response.choices[0].message.content)

# Running the local model provided an output almost immediately. Ollama returned a description that is easier to read. GPT returned a description that is much more detailed and complex. The huge advantage of running a model locally is privacy. However the disadvantage is the model is constrained by your system capabilities. 