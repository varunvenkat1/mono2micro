from llama_cpp import Llama

# Path to your GGUF model file
#model_path = "C:/Users/Administrator/Downloads/Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf"
model_path = "C:/Users/Administrator/Downloads/qwen2.5-coder-3b-instruct-q8_0.gguf"


# Initialize llama.cpp model globally (for reuse)
llm = Llama(
    model_path=model_path,
    n_ctx=32768,
    n_threads=8,     # Tune based on your CPU
    n_gpu_layers=0,  # CPU only
    verbose=True
)

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

# Generate text using llama-cpp
def generate_response(prompt_messages, token_size=512):
    prompt_text = format_chat_prompt(prompt_messages)
    output = llm.create_completion(
        prompt=prompt_text,
        max_tokens=token_size,
        temperature=0.7,
        top_p=0.95,
        stop=["<|endoftext|>", "<|user|>", "<|system|>"]
    )
    return output["choices"][0]["text"].strip()

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
    response = generate_response(prompt, 512)
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

- Include full path of the class after the repository name (do not write 'fullpath' word in JSON)
- Include each class under a single microservice only
- Include test classes under the corresponding microservice
- If any class doesn't fit, put it under shared services
- Mention interconnections and interdependencies between microservices
- Do not provide any extra information

{analysis_summary}
"""
    else:
        user_prompt = f"""
Below is the partially analyzed microservice boundaries based on a few files in a monolithic project:

{previous_response}

Along with the above mentioned JSON, combine the result of the following section.

Below is the analysis of more files in a java monolithic web api app. Follow the same Domain-Driven Design rules:

{analysis_summary}
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

# Identify microservices from combined summary
async def identify_microservices(combined_summary, previous_response=""):
    prompt = GetPromptForMicroserviceSuggestion(combined_summary, previous_response)
    return generate_response(prompt, token_size=10000)

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
    return generate_response(prompt, token_size=2048)
