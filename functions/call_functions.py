from google import genai
from google.genai import types
from functions.get_file_content import get_file_content
from functions.get_files_info import get_files_info
from functions.run_any_file import run_any_file
from functions.write_file import write_file
from functions.delete_file import delete_file
from functions.call_sub_rlm import call_sub_rlm, run_sub_rlm



def call_function(function_call_part, verbose=False, client=None):
    if verbose:
        print(f"Calling function: {function_call_part.name}({function_call_part.args})")
    else:
        print(f" - Calling function: {function_call_part.name}")

    function_name = function_call_part.name

    function_map = {
        "get_files_info": get_files_info,
        "get_file_content": get_file_content,
        "run_any_file": run_any_file,
        "write_file": write_file,
        "delete_file": delete_file,
        "call_sub_rlm": call_sub_rlm

    }

    if function_name == "call_sub_rlm":
        task = function_call_part.args.get("task", "")
        result = run_sub_rlm(client, task, verbose)
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response={"result": result},
                )
            ],
        )

    if function_name not in function_map:
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
            response={"error": f"Unknown function: {function_name}"},
        )
    ],
)
    else:
    
        
        
        args = dict(function_call_part.args)
        args["working_directory"] = "./working_directory"
        result = function_map[function_call_part.name](**args)
        
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response={"result": result},
                )
            ],
        )
        
    
        
        

        