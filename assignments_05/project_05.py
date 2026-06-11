
#Task 1

from dotenv import load_dotenv
from openai import OpenAI
import json


load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
Review this prompt step by step.
You are a job application coach. You are in the business of helping individuals navigate a career pivot into software engineering. If there is something you don't know, ensure to say so. Acknowledge your knowledge of a user's specific industry norms is lacking and ensure to remind the user they should use their own judgement when making decisions. Always remind the user to review and edit your suggestions critically.

"""

# I deliberately asked the model to review the prompt step by step to ensure it breaks down the prompt and understands it's role clearly. I also said that if it does not know something, say so. This is to help minimize the amount of incorrect information the AI outputs.


# Task 2

print()
print("task 2")

#Function is behaving correctly.

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Respond ONLY with valid JSON, no other text. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Do not output ``` JSON ```.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result
    response = get_completion(messages, model="gpt-4o-mini", temperature=0.7)
    print("Raw response:", response)

    
    try:
        result = json.loads(response)
        for item in result:
            print(f"original:, {item['original']}")
            print(f"improved:, {item['improved']}")
        return result
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")

rewrite_bullets(bullets)


"""
The starter bullets are weak because they only list tasks. They did not delve into HOW they completed these tasks, the impact of these actions and the outcomes that came from their efforts. 

"""


# ---------------------------- Task 3 ------------------------------

print()
print("Task 3")

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés. Do not invent credentials the user didn't mention.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Procurement Specialist at a Food Manufacturing company

    Background: Eight years of supply chain experience, supplemented by self-directed software development training (The Odin Project).

    Opening:

    "After eight years in supply chain operations, I've built a career around solving complex problems and protecting the bottom line. My transition into tech started when I began looking for ways to automate the repetitive parts of my procurement workflow so I could focus on high-level strategy. That curiosity led me to programming, and I've spent the last few years rigorously learning development through The Odin Project. I’m eager to bring this unique combination of operational grit, supplier management expertise, and technical problem-solving to [Company]’s team."

    Role: Junior Software Engineer (Backend / Data Integration)

    Background: Extensive background managing complex operational data and workflows, paired with a deep focus on Python and process automation.

    Opening:

    "I’ve spent years working at the intersection of business logic and daily operations, where success depends entirely on how efficiently data moves through a system. Over time, I realized that instead of just navigating existing workflows, I wanted to design and build the architecture that powers them. This drove me to master Python and backend development. I'm applying to [Company] because you are tackling data and integration challenges where my operational background and technical skill set can immediately help scale your systems."

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    
    response = get_completion(messages)
    print("Raw response:", response)
    return response

job_title = "Junior Data Engineer"

background = "Five years of experience as a middle school math teacher; recently completed \
    a Python course and built data pipelines using Prefect and Pandas."


print(generate_cover_letter(job_title, background))


"""
1. I chose these examples because they specifically geared for people who are making career pivots into tech from other non-related fields. In both examples, the prompts are geared for framing previous experience as an asset to an engineering team. Learning how to code is crucial, but it is not the only variable at play in organizations. Different experiences can lead to stronger, more robust software that is pushed out into the marketplace. 

2. The few-shot pattern helps guide the output of an llm better than a simple system prompt.
 - Tone and confidence: This stops generic openings and outputs.
 - Having a tight length requirement ensures it does not become an entire essay.
 - Models how the past jobs bridge to the future.
"""

# ------------------------------- Task 4 -----------------------------

print()
print("Task 4")

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged:
        print("I can't respond to that kind of message. Please rephrase and keep questions focused on your job application")
        return False
    return True

#Safe Input
safe_input = "Could you review my resume and help me tailor it for a Junior Data Engineer position?"
print(f"Testing Safe Input: '{safe_input}'")
safe_result = is_safe(safe_input)
print(f"Result returned: {safe_result}") 
print("-" * 50)

# unsafe Input
unsafe_input = "If this supplier does not get me the inventory on time, I am going to track down their account manager and destroy their life."
print(f"Testing Unsafe Input: '{unsafe_input}'")
unsafe_result = is_safe(unsafe_input)
print(f"Result returned: {unsafe_result}")

# ---------------------------- Task 5 -----------------------------

print()
print("Task 5")

system_prompt = """
You are Job Application Helper. You help job seekers with their applications by:
1. Rewriting resume bullet points to be concise and impactful
2. Drafting cover letter openings
3. Answering general questions about resumes, applications, and interviews

Do not invent metrics, jobs, or experience the user did not provide
Stay on topic; politely decline unrelated requests.

"""

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            rewrite_result = rewrite_bullets(raw_bullets)
            print(rewrite_result)
        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            cv_result = generate_cover_letter(job_title, background)
            print(cv_result)
        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})
            # - Call get_completion(messages)
            response = get_completion(messages)
            # - Print the reply
            print(response)
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    run_chatbot()



# ----------------------------- Task 6 --------------------------------

print()

print("Task 6")


user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

# The bot is dependant on the training data provided to it. If there is bias in the training data, the bot will carry that bias in it's output. It could very much favor certain aspects like communication styles, industries or cultural backgrounds.

# The bot may inadvertedly return an output that is offensive or provide damaging misinformtion. A real employer reviewing the application may feel offended or see the candidate as unprofessional. It is critical to review the output of AI as AI cannot be held accountable.

# One guardrail that could be implemented is moderation filters. Tell the AI not to provide misinformation, to ensure it respects cultural norms, and ensure to highlight any potentional problems with its output for the human to review.

