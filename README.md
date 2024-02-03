## Installation

```
# Install the latest release version
pip install runpod-llm

# or

# Install the latest development version (main branch)
pip install git+https://https://github.com/tsangwailam/langchain-runpod-llm
```

## Get Runpod API key

1. Goto www.runpod.io. Create a RunPod account.
2. From the portal, goto Settings>APIKeys
3. Create a new API key by click the "+ API Key" button.

## Usage

```python
from runpod_llm import RunpodLlama2

llm = RunpodLlama2(
        apikey="YOU_RUNPOD_API_KEY",
        llm_type="7b|13b",
        config={
            "max_tokens": 500, 
            #Maximum number of tokens to generate per output sequence.
            "n": 1,  # Number of output sequences to return for the given prompt.
            "best_of": 1,  # Number of output sequences that are generated from the prompt. From these best_of sequences, the top n sequences are returned. best_of must be greater than or equal to n. This is treated as the beam width when use_beam_search is True. By default, best_of is set to n.
            "Presence penalty": 0.2,  # Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
            "Frequency penalty": 0.5,  # Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
            "temperature": 0.3,  # Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.
            "top_p": 1,  # Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
            "top_k": -1,  # Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.
            "use_beam_search": False,  # Whether to use beam search instead of sampling.
        },
        verbose=True, # verbose output
    )

    some_prompt_template = xxxxx
    output_chain = some_prompt_template | llm
    output_chain.invoke({"input":"some input to prompt template"})
```