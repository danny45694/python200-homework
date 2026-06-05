


# ------------------------------------ LLMs as Transform -------------------------------


#Q1 

"""

Deterministic - When parsing dates into ISO date format - use deterministic code. LLMs are bad at this

LLM - When classifying something open-ended like reviews or issues, LLM is better. It handles variables/combination of information that deterministic code (regex) cannot.

Deterministic - LLMs should be used when judgement is or comprehension is required. Since the average of a list of numbers can be solved with code, code is better. 

LLM - Regex alone cannot extract the company name from a freeform job title. LLM is better suited. 

Deterministic - You can use code to decide if a product review is more than 100 words. You use split and len.

"""


#Q2 

"""

In pipelines, prompts should be constrained to output a single response. Open-ended responses are useful to humans but extremely difficult to work with in a pipeline. For complex outputs, the LLM should output JSON. JSON is easier to parse through with json.loads()

"Classify this review and extract the main topic. Reply with valid JSON only, using this exact format:

 {\"sentiment\": \"positive\", \"topic\": \"shipping\"}"
"""

#Q3

"""
1. If each call takes 1 second on average when processing 50K records, it will take 13 hours of processing.

2. One practical strategy is batching. You split the work into smaller batches and spread it out among different machines. Smaller models may be used.

"""

# ---------------------------------- Azure OpenAI ---------------------------------------


#Q1

"""
2 reasons:

1. When using Azure OpenAI, requests stay within Azure's infrastructure. This makes it subject to the same compliance controls as the rest of the cloud environment. This is good for companies that have strict data governance policies or requirements. 

2. Azure OpenAI costs appear on the same bill as the rest of the Azure infrastructure. Support also goes through the same company than 3rd parties. This is good for compliance and easier to manage.

"""

#Q2 

"""

The Azure OpenAI takes the following parameters: azure_endpoint, api_version, model.

azure_endpoint takes the unique url provided by Azure for your specific AI resource. It replaces the default OpenAI base URL to route requests to the Azure Cloud space.

api_version - Specific API version you are targeting. Azure requires this for correct routing and feature support. 

model takes the deployment name, not a model name. In Azure OpenAI, you call the named deployment the admin created and configured. 

"""

#Q3

"""

model takes deployment name, not a model name. YOu call a named deployment that is setup by an admin. The name is chosen by whoever set it up and can be anything. 

When you start a job and need to connect to Azure OpenAI, you use:
1. endpoint url and deployment name. Both are found in the Azure AI Foundry
2. Platform and team may also supply this to you.

"""