from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class RunpodBaseLLM(LLM):
    """RunposBaseLLM class."""

    url: str = None
    """Runpod API endpoint."""

    sync_url: str = None
    """Runpod API stream endpoint."""

    stream_url: str = None
    """Runpod API stream endpoint."""

    apikey: str = "YOUR_API_KEY"
    """Runpod API key."""

    verbose: bool = False
    """Whether to print verbose output."""

    stop: Optional[List[str]] = None
    """Sets the stop tokens to use."""

    def __init__(self,
                 apikey: str = None,
                 verbose: bool = False,
                 **kwargs: Any,
                 ):
        super().__init__()
        self.verbose = verbose
        if apikey == None:
            raise ValueError(
                "Please set your Runpod API key in the RunpodLlama2LLM class.")
        self.apikey = apikey

    def _print_prompts(self,
                       prompt: str
                       ):
        if self.verbose:
            print(
                f"\n\r{bcolors.OKGREEN}[RunpodLlama2LLM:PROMPTS]{bcolors.ENDC}\n{prompt}\n{bcolors.OKGREEN}[/PROMPTS]{bcolors.ENDC}")
