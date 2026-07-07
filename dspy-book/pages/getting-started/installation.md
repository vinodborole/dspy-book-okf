---
type: Web Page
title: Setting up DSPy - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/getting-started/installation
timestamp: '2026-07-07T10:31:54.390135+00:00'
---

# Setting up DSPy

Install DSPy with `pip install dspy` or run `uv add dspy` to add DSPy to a virtual environment. DSPy works in Python 3.10+ environments.

## Connecting to a language model

DSPy connects to language models with the `dspy.LM` class. To set up a language model, we provide a `"provider/model"` format string and an API key:

```
import dspy
# Pass the key explicitly...
lm = dspy.LM("openai/gpt-5-nano", api_key="YOUR_OPENAI_API_KEY")
dspy.configure(lm=lm)
```
Behind the scenes, DSPy uses the LiteLLM library to normalize inference providers into a single format. This allows you to provide a LiteLLM model string and connect to nearly any model and its provider. Click here to search for the full list.

For this tutorial, we could replace the `"openai/gpt-5-nano"` model string and our OpenAI API key with options from Anthropic, Google, OpenRouter, and more. All the examples below will work with any LiteLLM model string, without code changes.

Once we have an `LM`, calling `dspy.configure(lm=lm)` sets our `LM` as the default provider for every DSPy program in the process. This sets our `LM` globally, but we can selectively override this with `dspy.context` when more granular control is needed.

Let’s ensure everything works by manually calling the model:

```
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]
print(lm(messages = messages))
```
Which returns:

With an `LM` configured, we can proceed to writing our first program.

**Next:** Your first program →

# Citations

1. Source page: https://dspy.ai/getting-started/installation
