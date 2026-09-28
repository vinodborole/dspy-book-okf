---
type: Web Page
title: Advanced Tool Use - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/current/tutorials/tool_use
timestamp: '2026-09-28T13:20:17.892585+00:00'
---

# Tutorial: Advanced Tool Use

 Let’s walk through a quick example of building and prompt-optimizing a DSPy agent for advanced tool use. We’ll do this for the challenging task [ToolHop](https://arxiv.org/abs/2501.02506) but with an even stricter evaluation criteria.

Install the latest DSPy via `pip install -U dspy` and follow along. You will also need to `pip install func_timeout datasets`. This tutorial uses `dspy.SIMBA`, which requires numpy: `pip install dspy[numpy]`.

## Recommended: Set up MLflow Tracing to understand what's happening under the hood.

### MLflow DSPy Integration

 [MLflow](https://mlflow.org/) is an LLMOps tool that natively integrates with DSPy and offer explainability and experiment tracking. In this tutorial, you can use MLflow to visualize prompts and optimization progress as traces to understand the DSPy’s behavior better. You can set up MLflow easily by following the four steps below.

1. Install MLflow

1.  Start MLflow UI in a separate terminal
2.  Connect the notebook to MLflow
3.  Enabling tracing.

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

In this tutorial, we’ll demonstrate the new experimental `dspy.SIMBA` prompt optimizer, which tends to be powerful for larger LLMs and harder tasks. Using this, we’ll improve our agent from 35% accuracy to 60%.

Let’s now download the data.

```
Downloading 'ToolHop.json'...
```
Then let’s prepare a cleaned set of examples. The ToolHop task is interesting in that the agent gets a *unique set* of tools (functions) to use separately for each request. Thus, it needs to learn how to use *any* such tools effectively in practice.

And let’s define some helpers for the task. Here, we will define the `metric`, which will be (much) stricter than in the original paper: we’ll expect the prediction to match exactly (after normalization) with the ground truth. We’ll also be strict in a second way: we’ll only allow the agent to take 5 steps in total, to allow for efficient deployment.

Now, let’s define the agent! The core of our agent will be based on a ReAct loop, in which the model sees the trajectory so far and the set of functions available to invoke, and decides the next tool to call.

To keep the final agent fast, we’ll limit its `max_steps` to 5 steps. We’ll also run each function call with a timeout.

Out of the box, let’s assess our `GPT-4o`-powered agent on the development set.

```
2025/03/23 21:46:10 INFO dspy.evaluate.evaluate: Average Metric: 105.0 / 300 (35.0%)
35.0
```
Now, let’s optimize the agent using `dspy.SIMBA`, which stands for **Stochastic Introspective Mini-Batch Ascent**. This prompt optimizer accepts arbitrary DSPy programs like our agent here and proceeds in a sequence of mini-batches seeking to make incremental improvements to the prompt instructions or few-shot examples.

Having completed this optimization, let’s now evaluate our agent again. We see a substantial 71% relative gain, jumping to 60% accuracy.

```
2025/03/23 21:46:21 INFO dspy.evaluate.evaluate: Average Metric: 182.0 / 300 (60.7%)
60.67
```

# Citations

1. Source page: https://dspy.ai/current/tutorials/tool_use
