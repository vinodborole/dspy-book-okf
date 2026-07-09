---
type: Web Page
title: Output Refinement - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Output Refinement: BestOfN and Refine

Both `BestOfN` and `Refine` are DSPy modules designed to improve the reliability and quality of predictions by making multiple `LM` calls with different rollout IDs to bypass caching. Both modules stop when they have reached `N` attempts or when the `reward_fn` returns an award above the `threshold`.

## BestOfN

`BestOfN` is a module that runs the provided module multiple times (up to `N`) with different rollout IDs. It returns either the first prediction that passes a specified threshold or the one with the highest reward if none meets the threshold.

### Basic Usage

Lets say we wanted to have the best chance of getting a one word answer from the model. We could use `BestOfN` to try multiple rollout IDs and return the best result.

### Error Handling

By default, if the module encounters an error during an attempt, it will continue trying until it reaches `N` attempts. You can adjust this behavior with the `fail_count` parameter:

## Refine

`Refine` extends the functionality of `BestOfN` by adding an automatic feedback loop. After each unsuccessful attempt (except the final one), it automatically generates detailed feedback about the module’s performance and uses this feedback as hints for subsequent runs.

### Basic Usage

### Error Handling

Like `BestOfN`, `Refine` will try up to `N` times by default, even if errors occur. You can control this with the `fail_count` parameter:

## Comparison: BestOfN vs. Refine

Both modules serve similar purposes but differ in their approach:

- `BestOfN`simply tries different rollout IDs and selects the best resulting prediction as defined by the- `reward_fn`.
- `Refine`adds an feedback loop, using the lm to generate a detailed feedback about the module’s own performance using the previous prediction and the code in the- `reward_fn`. This feedback is then used as hints for subsequent runs.

## Practical Examples

### Ensuring Factual Correctness

### Summarization - Controlling Response Length

## Migration from `dspy.Suggest` and `dspy.Assert`

`BestOfN` and `Refine` are the replacements for `dspy.Suggest` and `dspy.Assert` as of DSPy 2.6.

# Citations

1. Source page: https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine
