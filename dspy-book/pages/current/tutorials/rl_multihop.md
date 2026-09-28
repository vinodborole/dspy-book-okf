---
type: Web Page
title: RL for Multi-Hop Research - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/current/tutorials/rl_multihop
timestamp: '2026-09-28T13:20:17.892585+00:00'
---

# Tutorial: Online RL for Multi-Hop Research

 WARNING: This feature is new and extremely EXPERIMENTAL. Unlike almost everything else in DSPy, it’s currently in pure proof of concept and development mode, but we release it to encourage community involvement.

For this tutorial, you will also need [DSPy’s Arbor RL framework](https://github.com/Ziems/arbor) which you can install with:

You may also have to install DSPy from the main branch:

### Install dependencies and download data

 To do the retrieval, we’ll use the cool BM25S library, as it’s pretty lightweight. You can replace this components with whatever you like.

Next, we’ll download a snapshot abstracts (i.e., first paragraphs) of all 5,000,000 Wikipedia pages as of 2017. We’ll use this as our retrieval corpus.

This is 500MB compressed, so the download and decompression may take 2-3 minutes.

And then let’s index it for BM25 retrieval! This will take 2-3 minutes.

### Load the HoVer dataset.

 Let’s load a dataset for our task. We’ll load examples from the HoVer multi-hop task, where the input is a (really!) complex claim and the output we’re seeking is the set of Wikipedia pages that are required to fact-check that claim.

You may have to install an older version of the dataset to get it working properly…

Now, let’s define a function to do the search in Wikipedia. This will use our BM25 index.

## A DSPy program for multi-hop research

 Now, let’s define the multi-hop program in DSPy. It’s going to be super simple, composed of `generate_query` and `append_notes` modules. We’ll define the instructions carefully, though they are typically not necessary.

### Define metrics for success in this task

 ## Optimize the `ResearchHop` system with `dspy.GRPO`

 Now, you can use the GRPO’ed program.

In our preliminary experiments, training about 18 hours boosts the recall (devset) from 61.8% to 66.2%. This is *typically* worse on cost/quality basis than you’d get from running prompt optimizers dspy.MIPROv2 or dspy.SIMBA, but it’s still a very solid start for online RL over arbitrary LM programs for small LMs.

# Citations

1. Source page: https://dspy.ai/current/tutorials/rl_multihop
