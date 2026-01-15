import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from prompts import repl_system_prompt
from rlm.repl import REPLEnvironment, extract_repl_code

# Sub-agent uses REPL only - no file operation tools
config=types.GenerateContentConfig(
        system_instruction=repl_system_prompt
    )

def run_sub_rlm(client, task, verbose=False, depth=0, max_depth=1):
    if depth > max_depth:
        return f"[Max recursion depth {max_depth} reached. Returning without further processing.]"
    
    messages = [types.Content(role="user", parts=[types.Part(text=task)])]
    repl = REPLEnvironment(context=task, llm_client=client, depth=depth, max_depth=max_depth)
    while True:
        result = call_sub_rlm(client, messages, verbose, config, repl=repl)
        if result is not None:
            return result

def call_sub_rlm(client, messages, verbose, config, repl=None):

    model_name = os.environ.get("GEMINI_SUB_RLM_MODEL")

    try:

        response = client.models.generate_content(
            model=model_name,
            contents=messages,
            config=config
        )

        if response.text:
            code = extract_repl_code(response.text)
            if code:
                output = repl.execute(code)
                if repl.finished:
                    return repl.result
                messages.append(types.Content(
                    role="user",
                    parts=[types.Part(text=f"REPL Output:\n{output}")]
                ))
                return None
                    


        

        if response.candidates:
            for candidate in response.candidates:
                messages.append(candidate.content)

        # No function calls - sub-agent uses REPL only
        # Return text response if no REPL code was found
        return response.text
            
    except Exception as e:
        print(f"Error: {e}")


schema_call_sub_rlm = types.FunctionDeclaration(
    name="call_sub_rlm",
    description="Call a sub recursive language model to handle complex tasks that require decomposition, analysis of large files, or multi-step reasoning. Use this when the task is too complex to handle directly.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "task": types.Schema(
                type=types.Type.STRING,
                description="The task or query to delegate to the sub-RLM agent. Be specific about what you want analyzed or accomplished.",
            ),
        },
        required=["task"],
    ),
)