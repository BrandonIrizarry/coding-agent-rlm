import io
import re
import contextlib
import traceback


class REPLEnvironment:

    def __init__(self, context, llm_client=None, depth=0, max_depth=1):
        """
        Initialize the REPL environment.

        Args:
            context: The input data available as 'context' variable in the REPL
            llm_client: The Gemini client for llm_query function (optional)
        """
        self.namespace = {}
        self.finished = False
        self.result = None
        self._llm_client = llm_client
        self._depth = depth
        self._max_depth = max_depth

        self.namespace['context'] = context
        self.namespace['llm_query'] = self._make_llm_query()
        self.namespace['FINAL'] = self._make_final()

    def _make_final(self):
        """
        Create the FINAL function that terminates execution with a result.

        Returns:
            A function that when called with an answer, sets self.finished = True
            and stores the answer in self.result

        Example usage in REPL:
            FINAL("The answer is 42")
        """

        def FINAL(answer):
            self.finished = True
            self.result = answer
            
        return FINAL


    def _make_llm_query(self):
        """
        Create the llm_query function for querying sub-LLMs.

        Returns:
            A function with signature: llm_query(query: str, context: str = "") -> str
            that sends a query to the LLM and returns the response text

        Example usage in REPL:
            summary = llm_query("Summarize this text", chunk)
        """

        def llm_query(query, context=""):
            from functions.call_sub_rlm import run_sub_rlm
            task = f"{query}\n\nContext:\n{context}" if context else query
            return run_sub_rlm(self._llm_client, task, depth=self._depth + 1, max_depth=self._max_depth)
        return llm_query

    def execute(self, code: str) -> str:
        """
        Execute Python code in the REPL environment.

        Args:
            code: The Python code string to execute

        Returns:
            A string containing:
            - Captured stdout output (from print statements)
            - Or error traceback if execution failed

        Side effects:
            - Updates self.namespace with any new variables
            - May set self.finished and self.result if FINAL/FINAL_VAR called
        """
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            try:
                exec(code, self.namespace)
            except Exception as e:
                return traceback.format_exc()
        return buffer.getvalue()
            
        
        
        


def extract_repl_code(response_text: str) -> str | None:
    """
    Extract Python code from a ```repl or ```python code block in LLM response.

    Args:
        response_text: The full text response from the LLM

    Returns:
        The code inside the code block, or None if no block found

    Example:
        text = '''Here's my code:
        ```repl
        x = 1 + 1
        print(x)
        ```
        '''
        code = extract_repl_code(text)
        # code == "x = 1 + 1\\nprint(x)"
    """
    # Try ```repl first, then ```python as fallback
    result = re.search(r"```repl\n(.*?)```", response_text, flags=re.DOTALL)
    if result:
        return result.group(1).strip()

    # Fallback to ```python
    result = re.search(r"```python\n(.*?)```", response_text, flags=re.DOTALL)
    if result:
        return result.group(1).strip()

    return None
