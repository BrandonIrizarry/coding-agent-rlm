# coding-agent-rlm
Coding agent based on RLM (Recursive Language Model) implementation

![Demo](assets/demo.gif)

## Overview

This is a coding agent that uses [recursive language models](https://alexzhang13.github.io/blog/2025/rlm/) to solve tasks with large contexts. The main agent can delegate subtasks to sub-agents that have access to a Python REPL environment. When an agent reads a file with get_file_contents that is too large, it can delegate the task of analyzing the file to a sub-agent, which then summarizes the information and passes it back to the main agent.

All code generated from the agent itself will be stored in the working_directory folder as that is where the agent has access to the file system to read, write, delete and run files. This does not apply to the Sub RLM agent, which runs in a separate environment and does not have access to the working_directory.

## Installation

To run the project, you can run main.py. Make sure to set up your environment variables first.

```bash
uv sync
```

This installs dependencies from the lockfile (`uv.lock`) ensuring reproducible builds.

For environment variables, create a .env file in the root directory with the following content:

```
GEMINI_API_KEY="your_gemini_api_key"
GEMINI_MODEL="gemini-2.5-flash" #or any model that you prefer
GEMINI_SUB_RLM_MODEL="gemini-2.5-flash-lite" #or any smaller model that you prefer
```

To run the coding agent, use:

```uv python main.py```


To test it against a benchmark, you can run:

benchmark.py (benchmark is based on Alex Zhang's RLM minimal implementation)

```uv python benchmark.py```


Max recursion is set to 1, meaning the agent can only call itself once to refine its solution. To change this, modify
the max_depth parameter in call_sub_rlm.py:

```
run_sub_rlm(client, task, verbose=False, depth=0, max_depth=1)
```

Here are some resources that were used for the project:

https://github.com/alexzhang13/rlm-minimal
https://huggingface.co/datasets/oolongbench/oolong-real
