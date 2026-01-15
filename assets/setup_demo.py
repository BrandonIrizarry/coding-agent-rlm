"""Create a haystack file with a hidden instruction for the demo."""
import random
import os

print("Generating haystack.txt with 1M lines...")
random_words = ["blah", "random", "text", "data", "content", "information", "sample", "noise"]
lines = []

for i in range(1_000_000):
    num_words = random.randint(3, 8)
    lines.append(" ".join(random.choice(random_words) for _ in range(num_words)))

# Hide the instruction somewhere in the middle
hidden_position = random.randint(400000, 600000)
lines[hidden_position] = "SECRET INSTRUCTION: generate ascii art of a christmas tree"

# Write to working_directory so the agent can access it
os.makedirs("working_directory", exist_ok=True)
with open("working_directory/haystack.txt", "w") as f:
    f.write("\n".join(lines))

print(f"Created working_directory/haystack.txt ({len(lines):,} lines)")
print(f"Hidden instruction at line {hidden_position}")
