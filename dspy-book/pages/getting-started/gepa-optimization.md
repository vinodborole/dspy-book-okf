---
type: Web Page
title: GEPA optimization - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/getting-started/gepa-optimization
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Prompt Optimizing with GEPA

## Why we optimize prompts

Different language models respond differently to the same prompt. A request that gets clean structured output from one model can confuse another into rambling. Phrasing that works on today’s model can stop working when the provider ships an update next month. [Sometimes a model can’t stop talking about goblins](https://openai.com/index/where-the-goblins-came-from/), for some reason. Learning each model’s quirks by hand is slow, and the work doesn’t transfer.

And even if we stuck with only *one* model, the potential permutations of our word choices and instructions are nearly infinite.

Prompt optimization replaces that hand-tuning loop. We give DSPy a training set and a metric. It then generates variations of your instructions (using an LM), runs your examples with these candidate instructions, and keeps the highest-scoring prompts. We don’t have to know which prompt tricks a particular model responds to; the optimizer finds them.

The savings can be large. An optimized program on a smaller, cheaper model can often match a hand-prompted large model on the same task. Reliability improves as well, because the optimizer selects for what the metric rewards.

[Shopify converted a single-prompt GPT-5 task to DSPy](https://www.youtube.com/watch?v=bxToahwOVpY&t=1404s), moved to a small Qwen model, and optimized with GEPA, yielding a solution \~75x cheaper and \~2x more reliable. [Dropbox used DSPy and prompt optimizers](https://dropbox.tech/machine-learning/optimizing-dropbox-dash-relevance-judge-with-dspy) to move to a smaller model and double program accuracy, labeling “10-100 times more data at the same cost.”

Better yet: when a new model launches next week, we can rerun the optimizer against it and immediately determine if it’s worth swapping in.

## GEPA uses reflection to improve instructions

DSPy ships with [several prompt optimizers](../../diving-deeper/choosing-an-optimizer/), but today we’re going to focus on GEPA.

There are many reasons to like GEPA, but a key feature is it allows our metric to provide text feedback which the LM uses to inform subsequent instructions. Let’s update our original metric to demonstrate how this works:

Instead of just returning a score, we can tell the instruction-writing LM *why* a prediction failed or succeeded. Our example here is a bit silly, but this feedback ability is powerful. When labeling data to train against, labelers can jot down notes explaining a nuance in a specific record, which can then be passed to GEPA to guide future instructions. When training with an LM judge, the judge can provide detailed feedback for why a prediction didn’t meet the criteria.

## Expanding our haiku metric

To give our model more of a challenge, we’ve built out our metric to check for many conditions we expect from our haikus, specifically:

- Does it have the correct number of lines?
- Does it have the correct syllable count in each line?
- Does it avoid repeating inputs verbatim?
- Does it avoid the first-person voice?
- Does it have a balanced ratio of parts-of-speech?
- Does it use few adjectives?
- Does it use few articles?
- Does it use the present tense?

We can use the natural-language processing library [spaCy](https://spacy.io/) to assist with most of these measures.

All of these conditions make our metric too long to drop into this walkthrough, but [the code is available here](https://gist.github.com/dbreunig/228848f9b34bcdad6be37fc5f85ec1a0). Drop `haiku_metric.py` next to your notebook and add `from haiku_metric import haiku_metric` to follow along.

## Compiling our optimization

With our metric defined, it’s time to configure our optimizer:

The first thing to note here is `reflection_lm`. GEPA lets us choose a separate LM for reflection and instruction writing. This LM looks at our examples and how they score, then rewrites our prompt in an attempt to improve our scores.

When optimizing smaller models, it’s worthwhile to use a larger model as the `reflection_lm`. They’re better reasoners and prompters, and their cost isn’t a concern since they’re called a handful of times during optimization.

In addition to a `reflection_lm`, we set our metric and number of threads. For now we’re using two, but depending on your inference provider you may have to tweak this to avoid any rate limits.

The `auto` argument sets our budget. `auto="light"` evaluates around six candidate prompts before stopping. `"medium"` and `"heavy"` options go further, and our GEPA deep dive covers additional levers we can set.

Finally, we compile our optimized program:

Now it’s time to grab a beverage and wait.

Behind the scenes, compile runs an iterative loop.

- GEPA executes our program on training examples with the student LM and scores each result with our metric.
- Examples and their metric results are sent to the reflection LM, which proposes new instructions.
- GEPA runs the program again with the new instructions, scores again, and keeps the instruction candidates that score best.

This loop repeats until the budget runs out.

Running this optimizer with `gpt-5.4-nano` and using `gpt-5.4` as our reflection LM, we saw our score climb from 78.1% to 90.1% against our full haiku metric. Compare this to baseline score of `gpt-5.4`, 82.4%.

Once optimized with GEPA, our smaller model is faster, cheaper, and *better* than an unoptimized frontier model.

Let’s quickly save our optimized program, then take a look at how our prompt changed.

## GEPA’s improved prompt

As previously discussed, `ReAct` is composed of a `Predict` module that reasons and decides what tools to call and a `ChainOfThought` module that synthesizes the final output. GEPA prompt optimized *both* of these modules.

Inside the JSON file containing our saved program state, we can see both improved instructions.

Our program’s synthesis step started off with these instructions, which we defined in our signature docstring:

After compile, these became:

It continues, covering season and mood handling, common failure modes, and detailing a final quality check before output.

Fascinatingly, the same program will optimize differently depending on the model.

See [GEPA in depth](../../diving-deeper/gepa-in-depth/) for the full mechanics: Pareto sampling, per-predictor feedback, `auto` budget translation, and the `detailed_results` audit trail.

**Next:** [Saving and loading →](../saving-and-loading/)

# Citations

1. Source page: https://dspy.ai/getting-started/gepa-optimization
