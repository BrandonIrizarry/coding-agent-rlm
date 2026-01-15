"""Needle-in-haystack benchmark - tests autonomous RLM capability with 1M lines."""
import os
import random
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
from main import generate_content

load_dotenv()

# Create massive haystack (1M lines)
print("Generating massive context with 1M lines...")
answer = str(random.randint(1000000, 9999999))
random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
lines = []
for _ in range(1_000_000):
    num_words = random.randint(3, 8)
    lines.append(" ".join(random.choice(random_words) for _ in range(num_words)))

magic_position = random.randint(400000, 600000)
lines[magic_position] = f"The magic number is {answer}"
print(f"Magic number inserted at position {magic_position}")

# Store in env var (too large to pass in prompt)
os.environ["HAYSTACK_DATA"] = "\n".join(lines)
print(f"Context stored in HAYSTACK_DATA ({len(os.environ['HAYSTACK_DATA']):,} chars)")

# Setup agent
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[
        schema_get_files_info, schema_run_any_file, schema_write_file,
        schema_get_file_content, schema_delete_file, schema_call_sub_rlm
    ])],
    system_instruction=system_prompt
)

# Run
print(f"\nLooking for: {answer}")
prompt = "Find the magic number in HAYSTACK_DATA environment variable. It contains 1M lines of random text with a hidden number."
messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]

for turn in range(20):
    result = generate_content(client, messages, False, config)
    if result:
        print(f"\nResult: {result}")
        print(f"Expected: {answer}")
        print(f"Status: {'SUCCESS' if answer in str(result) else 'FAILED'}")
        break
