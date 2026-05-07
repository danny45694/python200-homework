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
You are a job application coach. You are in the business of helping individuals navigate a career pivot into software engineering. If there is something you don't know, ensure to say so. Acknowledge your knowledge of a user's specific industry norms is lacking and ensure to remind the user they should use their own judgement when making decisions.

"""

# I deliberately asked the model to review the prompt step by step to ensure it breaks down the prompt and understands it's role clearly. I also said that if it does not know something, say so. This is to help minimize the amount of incorrect information the AI outputs.


# Task 2


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
        print("Parsed sentiment:", result["sentiment"])
        print("Confidence:", result["confidence"])
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")


rewrite_bullets(bullets)


# ---------------------------- Task 3 ------------------------------


def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result


    job_title = "Junior Data Engineer"
    background = "Five years of experience as a middle school math teacher; recently completed \
    a Python course and built data pipelines using Prefect and Pandas."