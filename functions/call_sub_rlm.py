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
        print(f"[Depth {depth}] Max recursion depth reached")
        return f"[Max recursion depth {max_depth} reached. Returning without further processing.]"
    
    messages = [types.Content(role="user", parts=[types.Part(text=task)])]
    repl = REPLEnvironment(context=task, llm_client=client, depth=depth, max_depth=max_depth)
    while True:
        result = call_sub_rlm(client, messages, verbose, config, repl=repl, depth=depth)
        if result is not None:
            return result

def call_sub_rlm(client, messages, verbose, config, repl=None, depth=0):
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
                print(f"Executing REPL code ({len(code)} chars)...")
                output = repl.execute(code)
                if repl.finished:
                    return repl.result
                messages.append(types.Content(
                    role="user",
                    parts=[types.Part(text=f"REPL Output:\n{output}")]
                ))
                return None
            else:
                # No REPL code found - log what the model said
                print(f"No REPL code found. Response: {response.text[:200]}...")
                    


        

        if response.candidates:
            for candidate in response.candidates:
                if candidate.content is not None:
                    messages.append(candidate.content)

        # No function calls - sub-agent uses REPL only
        # Return text response if no REPL code was found
        return response.text or "[No response from sub-agent]"

    except Exception as e:
        print(f"Error: {e}")
        return f"[Sub-agent error: {e}]"


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