import openai
import asyncio

# OpenAI-compatible API setup for suggestion and analysis models
suggestion_openai = openai.OpenAI(api_key="not-needed", base_url="http://10.0.141.160:8010/v1")
analysis_openai = openai.OpenAI(api_key="not-needed", base_url="http://10.0.141.160:8011/v1")

# Format chat messages into Qwen-style prompt
def format_chat_prompt(messages):
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"
    return prompt

# Generate response from model (using OpenAI-compatible API)
def generate_response(client, prompt_messages, token_size=512):
    prompt_text = format_chat_prompt(prompt_messages)

    response = client.chat.completions.create(
        model="model",  # `model` is a placeholder for llama.cpp server
        messages=[
            {"role": "user", "content": prompt_text}
        ],
        max_tokens=token_size,
        temperature=0.7,
        top_p=0.95,
        timeout=3600,
        stop=["<|endoftext|>", "<|user|>", "<|system|>"]
    )

    return response.choices[0].message.content.strip()

# Build prompt to analyze a source file
def get_prompt_for_analysis(context, file_content):
    system_message = "You are an assistant that migrates dotnet monolithic web api apps into microservices architecture."
    user_prompt = f"""Context from related files: {context}

Below is one of the files in the java monolithic web api app:

{file_content}

Analyse the file and retrieve the precise information such as:
- Type of file (controller/service)
- Functional summary of the class
- Functional summary of each method
- Internal/external dependencies."""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

# Analyze a file
async def analyse_file(context, file):
    prompt = get_prompt_for_analysis(context, file)
    response = generate_response(analysis_openai, prompt, 512)
    print(response)
    return response

# Prompt for microservice identification
def GetPromptForMicroserviceSuggestion(analysis_summary, previous_response):
    system_message = (
        "You are an assistant that migrates java monolithic web api apps into microservices architecture. "
        "All the classes from the monolithic app have to be considered and listed in the output response. "
        "Generate the response in JSON format."
    )
    if previous_response == "":
        user_prompt = f"""
Below is the analysis of the important files in a java monolithic web api app.

Suggest microservices based on Domain-Driven Design principles in JSON with the corresponding grouping of classes from the monolithic app.

- Provide name key for each microservice name in the JSON strictly
- Include full path of the class after the repository name (do not write 'fullpath' word in JSON)
- Include test classes under associated microservice only and dont create seperate test microservice.
- If any class doesn't fit, put it under shared services microservice.
- Provide the description of each microservice.
- Mention interconnections and interdependencies between microservices.
- Mention programming language as well.
- Provide the output in strict json format.
- Do not provide any extra information.

{analysis_summary}
"""
    else:
        user_prompt = f"""
Below is the partially analyzed microservice boundaries based on a few files in a monolithic project:

{previous_response}

Along with the above mentioned JSON, combine the result of the following section.

Suggest microservices based on Domain-Driven Design principles in JSON with the corresponding grouping of classes from the monolithic app.

- Provide name key for each microservice name in the JSON strictly
- Include full path of the class after the repository name (do not write 'fullpath' word in JSON)
- Include test classes under associated microservice only and dont create seperate test microservice.
- If any class doesn't fit, put it under shared services microservice.
- Provide the description of each microservice.
- Mention interconnections and interdependencies between microservices.
- Mention programming language as well.
- Provide the output in strict json format.
- Do not provide any extra information.

{analysis_summary}
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

# Identify microservices from combined summary
async def identify_microservices(combined_summary, previous_response=""):
    prompt = GetPromptForMicroserviceSuggestion(combined_summary, previous_response)
    response = generate_response(suggestion_openai, prompt, token_size=15000)
    print(response)
    return response

# Prompt for microservice code generation
def GetPromptForMicroserviceCodeGeneration(context_code):
    system_message = "You are an assistant that migrates dotnet monolithic web api apps into microservices architecture. All the classes from the monolithic app have to be considered. Generate output in JSON format."

    user_prompt = f"""
You are an expert .NET developer. Generate microservice project C# code with:

- startup.cs
- controller.cs
- service.cs

Based on the following context:

{context_code}
"""
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

# Generate C# code for microservice
async def genrate_microservices(context_code: str):
    prompt = GetPromptForMicroserviceCodeGeneration(context_code)
    return generate_response(suggestion_openai, prompt, token_size=2048)
