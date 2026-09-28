---
type: Web Page
title: RL for Privacy-Conscious Delegation - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/current/tutorials/rl_papillon
timestamp: '2026-09-28T13:20:17.892585+00:00'
---

# Tutorial: Online RL over a Multi-Module DSPy Program

 WARNING: This feature is new and extremely EXPERIMENTAL. Unlike almost everything else in DSPy, it’s currently in pure proof of concept and development mode, but we release it to encourage community involvement.

In this tutorial, we optimize the LM weights of [PAPILLON](https://dspy.ai/tutorials/papillon/) with `ArborGRPO`, a generalization of the popular GRPO online RL algorithm of LLMs to sophisticated multi-module LM programs.

PAPILLON is a system for privacy-preserving delegation, where we will teach a tiny model (1.5B parameters) to use an “untrusted” external LLM, which is more powerful but may save your private data, to balance high-quality and private chat.

For this tutorial, you will also need [DSPy’s Arbor RL framework](https://github.com/Ziems/arbor) which you can install with: 

You may also have to install DSPy from the main branch:

### Define metrics for success in this task

 What does it mean for a PAPILLON system to be successful?

1. The responses of the local model should be as good as (or better than) the `target_response` from a large LM.
2. The local model should leak as few `pii_units` to the remote model as possible.

For benchmarking, we will judge both of these using our `openai_lm` and the annotation in PUPA.

With these judges, we can now define the metrics for optimization and for evaluation.

### Evaluate zero-shot PAPILLON

 Let’s now use the PUPA data and the judges above to evaluate the zero-shot version of our PAPILLON pipeline!

### Optimize PAPILLON with `dspy.GRPO`

 Let’s run the `dspy.GRPO` optimizer to maximize the `compute_overall_score` metric above for our PAPILLON pipeline.

We ran this on 4xH100 GPUs for a couple of hours. But first, you’ll need to set up Arbor (as above).

Now, you can use the GRPO’ed program.

In our preliminary experiments, training three hours boosts the composite score (devset) from 54.6% to 60.0%. This is *typically* worse on cost/quality basis than you’d get from running prompt optimizers like dspy.MIPROv2 or dspy.SIMBA, but it’s still a very solid start for online RL over arbitrary LM programs for tiny LMs.

# Citations

1. Source page: https://dspy.ai/current/tutorials/rl_papillon
