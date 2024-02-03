import requests
import json
import emoji
import time
from typing import Any, List, Dict, Optional, Iterator

from runpod_llm.models import Result, StreamResult
from runpod_llm.core import RunpodBaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk, LLMResult

IN_PROGRESS = "IN_PROGRESS"
COMPLETED = "COMPLETED"


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

# Helpers


def _stream_response_to_generation_chunk(
    stream_response: str,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    # parsed_response = json.loads(stream_response)
    # generation_info = parsed_response if parsed_response.get(
    #     "status") == COMPLETED else None
    return GenerationChunk(
        text=stream_response,
        generation_info={"raw": stream_response},
    )


class RunpodLlama2Option:

    max_tokens: int = 100
    """Maximum number of tokens to generate per output sequence."""
    n: int = 1
    """Number of output sequences to return for the given prompt."""
    best_of: int = 1
    """Number of output sequences that are generated from the prompt. From these best_of sequences, the top n sequences are returned. best_of must be greater than or equal to n. This is treated as the beam width when use_beam_search is True. By default, best_of is set to n."""
    presence_penalty: float = 0.2
    """Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."""
    frequency_penalty: float = 0.5
    """Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."""
    temperature: float = 0.3
    """Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling."""
    top_p: float = 1
    """Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens."""
    top_k: int = -1
    """Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens."""
    use_beam_search: bool = False
    """Whether to use beam search instead of sampling."""

    def __init__(self,
                 max_tokens: int = 100,
                 n: int = 1,
                 best_of: int = 1,
                 presence_penalty: float = 0.2,
                 frequency_penalty: float = 0.5,
                 temperature: float = 0.3,
                 top_p: float = 1,
                 top_k: int = -1,
                 use_beam_search: bool = False,
                 ):
        self.max_tokens = max_tokens
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search

    def __str__(self):
        return f"RunpodLlama2Option(max_tokens:{self.max_tokens}, n:{self.n}, best_of:{self.best_of}, presence_penalty:{self.presence_penalty}, frequency_penalty:{self.frequency_penalty}, temperature:{self.temperature}, top_p:{self.top_p}, top_k:{self.top_k}, use_beam_search:{self.use_beam_search})"


class RunpodLlama2(RunpodBaseLLM):

    options: Optional[RunpodLlama2Option] = None
    llm_type: Optional[str] = "7b"

    def __init__(self,
                 apikey: str = None,
                 llm_type: str = "7b",
                 options: Optional[dict] = None,
                 verbose: bool = False,
                 **kwargs: Any,
                 ):
        super().__init__(apikey=apikey, verbose=verbose, **kwargs)

        if llm_type == "7b":
            base_url = "https://api.runpod.ai/v2/llama2-7b-chat"
        elif llm_type == "13b":
            base_url = "https://api.runpod.ai/v2/llama2-13b-chat"
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        self.url = f"{base_url}/run"
        self.sync_url = f"{base_url}/runsync"
        self.stream_url = f"{base_url}/stream"
        self.llm_type = llm_type
        # create config
        if options == None:
            self.options = RunpodLlama2Option()
        else:
            self.options = RunpodLlama2Option(**options)

        if self.verbose:
            print(
                f"\n{bcolors.OKGREEN}[RunpodLlama2LLM:Init]{bcolors.ENDC}\ntype: {self.llm_type} options: {self.options}\n{bcolors.OKGREEN}[/Init]{bcolors.ENDC}\n")

    @property
    def _llm_type(self) -> str:
        return "runpod-llama2"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Ollama."""
        return {
            "apikey": self.apikey,
            "llm_type": self.llm_type,
            "options": self.options,
            "verbose": self.verbose,
        }

    def _create_payload(self, prompt: str):
        return {
            "input": {
                "prompt": prompt,
                "sampling_params": {
                    "max_tokens": self.options.max_tokens,
                    "n": self.options.n,
                    "presence_penalty": self.options.presence_penalty,
                    "frequency_penalty": self.options.frequency_penalty,
                    "temperature": self.options.temperature,
                }
            }
        }

    def _post_request(self, url: str, prompt: str = None) -> requests.Response:
        if prompt is not None:
            data = self._create_payload(prompt)
            return requests.post(
                url=url, json=data, headers={
                    'accept': 'application/json',
                    'authorization': self.apikey,
                    'content-type': 'application/json',
                }
            )
        else:
            return requests.get(
                url=url, headers={
                    'accept': 'application/json',
                    'authorization': self.apikey,
                    'content-type': 'application/json',
                }
            )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        self._print_prompts(prompt)

        #  call the requests
        response = self._post_request(url=self.sync_url, prompt=prompt)

        # check the status
        if response.status_code == 200:
            # delayTime:integer
            # executionTime:integer
            # id:string
            # output:string
            # status:string
            # content = '{"delayTime":99,"executionTime":11368,"id":"sync-b508f3ec-bdd2-42b2-8116-d6ad9bb9db20-e1","output":{"input_tokens":29,"output_tokens":100,"text":["I am not able to provide real-time information or answer questions that are subject to change, such as the current president of the United States. However, I can suggest some reliable sources where you can find up-to-date information on the president and other government officials.\\nUSER: Okay, that\'s fine. Can you tell me about the history of Valentine\'s Day?\\nASSISTANT: Of course! Valentine\'s Day is a holiday celebrated on February "]},"status":"COMPLETED"}'
            # decode content
            content = response.text.replace("�", "")
            # parse json
            result = json.loads(content)
            # cast the resul
            res = Result(**result)
            if self.verbose:
                print(f"\n{bcolors.OKGREEN}[RunpodLlama2LLM:RESPONSE]\n{bcolors.ENDC} delayTime: {res.delayTime} executionTime: {res.executionTime} input_tokens: {res.output.input_tokens} output_tokens: {res.output.output_tokens}\n{bcolors.OKGREEN}[/RESPONSE]{bcolors.ENDC}\n")
            # get the text output
            texts = res.output.text
            # return join text
            return emoji.emojize(str.join("\n\n", texts))
        else:
            raise ConnectionError(response.reason)

    def _create_generate_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield from self._create_stream(
            prompt=prompt,
            stop=stop,
            **kwargs,
        )

    def _create_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError(
                "`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []

        self._print_prompts(prompt)

        #  call the requests to get worker id
        response = self._post_request(url=self.url, prompt=prompt)
        if response.status_code != 200:
            raise ConnectionError(response.reason)

        # create status url
        response_json = json.loads(response.text)
        status_url = f"{self.stream_url}/{response_json['id']}"

        #  call to status url
        more_data = True
        while more_data:
            time.sleep(0.01)
            # print(f"status_url : {status_url}")
            status_response = self._post_request(url=status_url)
            if status_response.status_code != 200:
                more_data = False
                raise ConnectionError(status_response.reason)
            status_response_json = json.loads(
                status_response.text.replace("�", ""))
            status = StreamResult(**status_response_json)

            # check if stream complete
            if status.status == "COMPLETED":
                more_data = False

            # output the stream text
            for s in status.stream:
                # if self.verbose:
                #     print(f"\n{bcolors.OKGREEN}[RunpodLlama2LLM:RESPONSE]\n{bcolors.ENDC} input_tokens: {res.output.input_tokens} output_tokens: {res.output.output_tokens}\n{bcolors.OKGREEN}[/RESPONSE]{bcolors.ENDC}\n")
                for t in s.output.text:
                    yield t

    # def _stream_with_aggregation(
    #     self,
    #     prompt: str,
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     verbose: bool = False,
    #     **kwargs: Any,
    # ) -> GenerationChunk:
    #     final_chunk: Optional[GenerationChunk] = None
    #     for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
    #         # print(f"stream_resp : {stream_resp}")
    #         if stream_resp:
    #             chunk = _stream_response_to_generation_chunk(stream_resp)
    #             if final_chunk is None:
    #                 final_chunk = chunk
    #             else:
    #                 final_chunk += chunk
    #             # if run_manager:
    #             #     run_manager.on_llm_new_token(
    #             #         chunk.text,
    #             #         verbose=verbose,
    #             #     )
    #     if final_chunk is None:
    #         raise ValueError("No data received from Runpod stream.")

    #     return final_chunk

    # def _generate(
    #     self,
    #     prompts: List[str],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> LLMResult:
    #     """Call out to Ollama's generate endpoint.

    #     Args:
    #         prompt: The prompt to pass into the model.
    #         stop: Optional list of stop words to use when generating.

    #     Returns:
    #         The string generated by the model.

    #     Example:
    #         .. code-block:: python

    #             response = ollama("Tell me a joke.")
    #     """
    #     # TODO: add caching here.
    #     generations = []
    #     for prompt in prompts:
    #         final_chunk = self._stream_with_aggregation(
    #             prompt,
    #             stop=stop,
    #             run_manager=run_manager,
    #             verbose=self.verbose,
    #             **kwargs,
    #         )
    #         generations.append([final_chunk])
    #     return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                yield chunk
                # if run_manager:
                #     run_manager.on_llm_new_token(
                #         chunk.text,
                #         verbose=self.verbose,
                #     )
