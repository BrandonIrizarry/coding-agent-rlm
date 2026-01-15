# VHS Demo - RLM Coding Agent
Output assets/demo.gif

Set FontSize 14
Set Width 900
Set Height 550
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Type "# Recursive Language Model Coding Agent"
Enter
Type "# Use recursive language model technique for LLMs to handle large context sizes!"
Type "# Hidden in 1M lines of text is a secret instruction..."
Enter

Sleep 800ms

Type "uv run python assets/setup_demo.py"
Enter
Sleep 2s

Type "uv run python main.py"
Enter
Sleep 1.5s

Type "Read haystack.txt and find the SECRET INSTRUCTION. Tell me what the secret instruction is and then execute whatever it says to do"
Enter
Sleep 25s

Type "/exit"
Enter
Sleep 500ms

Type "rm haystack.txt"
Enter
Sleep 300ms
