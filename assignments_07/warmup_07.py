from dotenv import load_dotenv
if load_dotenv():
    print("Successfully loaded api key")

from openai import OpenAI
from pprint import pprint

client = OpenAI()




# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


tools = [
    {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Converts a Celsius temperature to Fahrenheit and returns it as a string.',
            'parameters': {
                'type': 'object',
                'properties': {"number"},
                'required': [],
            },
        },
    }
]
print('Tools list defined with one tool: celsius_to_fahrenheit')

list = [0, 100, -40]

for num in list:
    print(celsius_to_fahrenheit(num))


#Q2

import json

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # Adjusting code to use celsius function
            if function_name == 'celsius_to_fahrenheit':
                tool_result = celsius_to_fahrenheit()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''


"""
 1. I think calling run_agent("Convert 100 degrees Celsius to Fahrenheit") will trigger a tool call. The model LLM will look at the prompt, derive the semantic meaning, convert string into a format so it can use the function. Finally it will return the result.
 2. I think it will perform 4 calls in total

"""

answer = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(answer)