import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functions.get_files_info import schema_get_files_info
from functions.run_any_file import schema_run_any_file
from functions.write_file import schema_write_file
from functions.get_file_content import schema_get_file_content
from functions.delete_file import schema_delete_file
from functions.call_functions import call_function
from prompts import repl_system_prompt

available_functions = types.Tool(
        function_declarations=[schema_get_files_info, schema_run_any_file, schema_write_file, schema_get_file_content, schema_delete_file],
    )
config=types.GenerateContentConfig(
        tools=[available_functions], system_instruction=repl_system_prompt
    )

def run_sub_rlm(client, task, verbose=False):
    """Run sub-agent to completion and return result."""
    messages = [types.Content(role="user", parts=[types.Part.from_text(task)])]
    while True:
        result = call_sub_rlm(client, messages, verbose, config)
        if result is not None:
            return result

def call_sub_rlm(client, messages, verbose, config):
        
    model_name = os.environ.get("GEMINI_SUB_RLM_MODEL")
    
    function_responses = []

    try:
        

        response = client.models.generate_content(
            model=model_name,
            contents=messages,
            config=config
        )



        if response.candidates:
            for candidate in response.candidates:
                messages.append(candidate.content)

        if not response.function_calls:
                return response.text


        for function_call_part in response.function_calls:
            function_call_result = call_function(function_call_part, verbose=verbose)
            function_responses += function_call_result.parts
            # just print the function result if verbose


        messages.append(types.Content(role="user", parts=function_responses))
        return None
            
    except Exception as e:
        print(f"Error: {e}")