import os
import random
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from prompts import system_prompt
from functions.get_files_info import schema_get_files_info
from functions.run_any_file import schema_run_any_file
from functions.write_file import schema_write_file
from functions.get_file_content import schema_get_file_content
from functions.delete_file import schema_delete_file
from functions.call_sub_rlm import schema_call_sub_rlm
from functions.call_functions import call_function
import click


def generate_massive_context(num_lines: int = 100_000, answer: str = "1298418") -> tuple[str, int]:
    """Generate a massive context with a hidden magic number and store in env var."""
    print(f"Generating context with {num_lines:,} lines...")

    random_words = ["blah", "random", "text", "data", "content", "information", "sample",
                    "word", "line", "stuff", "thing", "item", "value", "entry"]

    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))

    # Insert the magic number at a random position (somewhere in the middle)
    magic_position = random.randint(int(num_lines * 0.4), int(num_lines * 0.6))
    lines[magic_position] = f"The magic number is {answer}"

    print(f"Magic number '{answer}' inserted at line {magic_position:,}")

    context = "\n".join(lines)

    # Store in environment variable instead of passing in prompt
    os.environ["HAYSTACK_DATA"] = context
    print(f"Context stored in HAYSTACK_DATA env var ({len(context):,} characters)")

    return len(context), magic_position


def run_main_agent(client, config, prompt, verbose=False, max_turns=20):
    """Run the main agent loop (same as main.py but for benchmark)."""
    messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    model_name = os.environ.get("GEMINI_MODEL")

    for turn in range(max_turns):
        print(f"[Main Agent] Turn {turn + 1}...")

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=messages,
                config=config
            )

            if verbose:
                print(f"  Prompt tokens: {response.usage_metadata.prompt_token_count}")
                print(f"  Response tokens: {response.usage_metadata.candidates_token_count}")

            if response.candidates:
                for candidate in response.candidates:
                    messages.append(candidate.content)

            if not response.function_calls:
                print("[Main Agent] Finished with text response")
                return response.text

            # Handle function calls
            function_responses = []
            for function_call_part in response.function_calls:
                print(f"[Main Agent] Calling function: {function_call_part.name}")
                function_call_result = call_function(function_call_part, verbose=verbose, client=client)
                function_responses += function_call_result.parts

            messages.append(types.Content(role="user", parts=function_responses))

        except Exception as e:
            print(f"Error: {e}")
            return f"[Error: {e}]"

    return "[No response after max turns]"


@click.command()
@click.option("--num-lines", default=100_000, help="Number of lines to generate")
@click.option("--max-turns", default=20, help="Max turns for main agent")
@click.option("--verbose", is_flag=True, help="Enable verbose output")
def benchmark(num_lines, max_turns, verbose):
    """Needle-in-haystack benchmark for RLM system."""

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # Setup main agent with tools
    available_functions = types.Tool(
        function_declarations=[
            schema_get_files_info,
            schema_run_any_file,
            schema_write_file,
            schema_get_file_content,
            schema_delete_file,
            schema_call_sub_rlm
        ],
    )
    config = types.GenerateContentConfig(
        tools=[available_functions],
        system_instruction=system_prompt
    )

    print("=" * 60)
    print("RLM Needle-in-Haystack Benchmark")
    print("=" * 60)

    # Generate random answer and context (stored in HAYSTACK_DATA env var)
    answer = str(random.randint(1000000, 9999999))
    context_size, magic_position = generate_massive_context(num_lines=num_lines, answer=answer)

    print(f"Context size: {context_size:,} characters")
    print(f"Looking for magic number: {answer}")
    print("=" * 60)

    # Create a SMALL prompt for the MAIN agent - data is in env var
    prompt = f"""Find the magic number hidden in a large text.

The text contains {num_lines:,} lines of random words ({context_size:,} characters total).
Somewhere in the middle there is a line that says "The magic number is X" where X is a 7-digit number.

IMPORTANT: The text data is stored in the environment variable HAYSTACK_DATA.

Call call_sub_rlm with this EXACT task:
\"\"\"
Search for a magic number in the HAYSTACK_DATA environment variable.

In your REPL, do this:
1. import os
2. data = os.environ["HAYSTACK_DATA"]
3. Search for the pattern "The magic number is " followed by digits
4. Use regex: import re; match = re.search(r"The magic number is (\\d+)", data)
5. Return match.group(1) using FINAL()

The data is {context_size:,} characters. Search it and return ONLY the 7-digit number.
\"\"\"

Return ONLY the magic number (just the digits).
"""

    print("\nStarting Main Agent...\n")
    start_time = time.time()

    # Run the main agent
    result = run_main_agent(client, config, prompt, verbose=verbose, max_turns=max_turns)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Expected answer: {answer}")
    print(f"Agent result: {result}")
    print(f"Time taken: {elapsed:.2f}s")

    # Check if correct
    result_str = str(result).strip()
    if answer in result_str:
        print("STATUS: SUCCESS - Magic number found!")
        return True
    else:
        print("STATUS: FAILED - Wrong answer")
        return False


if __name__ == "__main__":
    benchmark()
