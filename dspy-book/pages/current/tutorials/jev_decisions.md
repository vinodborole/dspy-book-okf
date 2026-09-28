---
type: Web Page
title: Decision-Making with Jev Types - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/current/tutorials/jev_decisions
timestamp: '2026-09-28T13:20:17.892585+00:00'
---

# Decision-Making with Jev Types

 Experimental API

`Noul`, `Score`, `Choice`, `TypeSafe`, and `ReAnchor` are experimental and may change without warning. See the [API reference](../../api/experimental/DecisionTypes/) for details.

In this tutorial, we will walk you through building **structured decision-making programs** in DSPy using the three experimental decision types — `Noul`, `Score`, and `Choice` — and calibrating them automatically with the `ReAnchor` optimizer.

Decision types let you move beyond free-text LLM outputs. Instead of asking an LLM “Is this urgent?” and parsing “Yes” from a string, you get a **probability-backed boolean** with a tunable threshold. Instead of asking “Rate severity from 1 to 5”, you get a **continuous score** with level boundaries you can adjust. This is the core idea behind DSPy’s Jev integration.

## Prerequisites

  If you want to use the TypeSafe System One backend (optional):

## Define Decision Types

 Decision types describe the **possible outcomes** of a structured judgment. Each type uses bracket syntax to declare its options:

**Key points:**

- `Noul` is a boolean with probability.`if result.urgent:` works — it delegates to`.value` .
- `Score` produces a continuous float in`[0, N-1]` plus a discrete`.level` . For 3 labels, the value range is`[0.0, 2.0]` .
- `Choice` selects from a fixed set. The value preserves its Python type (`str` ,`int` ,`bool` , or`None` ).

## Build a Signature

 A DSPy signature defines your program’s inputs and outputs. Decision types go in output fields just like any other type:

Descriptions are required

Every decision output field must have a `desc=` or an explicit `instructions` entry in `predict.fields`. Without one, DSPy raises an error before calling the LM.

## Run with a Generative LM

 You don’t need a TypeSafe API key to use decision types. Any DSPy-supported LM works:

Behind the scenes, DSPy asks the LM to produce probability evidence (not just a label) and derives the final value locally using thresholds, cuts, and weights.

## Tune Decision Boundaries

 The default threshold for `Noul` is 0.5 — if P(True) ≥ 0.5, the result is `True`. But what if your use case needs to be more conservative (e.g., only flag as urgent when the LM is very confident)?

You can set parameters directly:

| Type | Parameter | Effect | 
|---|---|---|
| `Noul` | `threshold` | P(True) must be ≥ this value to return `True` . Range: [0, 1]. | 
| `Score` | `cuts` | N−1 boundaries that divide the continuous value into discrete levels. | 
| `Choice` | `weights` | Multipliers on each option’s probability. Higher = preferred in close calls. | 

## Automatic Calibration with ReAnchor

 Manually tuning thresholds is tedious. `ReAnchor` does it automatically by fitting parameters to a labeled training set:

**How ReAnchor works:**

1. Runs your program on the training set and collects probability evidence from each call
2. For each decision output, tries candidate parameter values (threshold midpoints, cut positions, weight multipliers)
3. Cross-validates with 5-fold checks to avoid overfitting to a single example
4. Returns a copy of your program with the best-scoring parameters

Tip

ReAnchor works with cached LM calls. Run your training set once, then iterate on calibration without re-calling the LM. Set `require_cache=False` if you want to allow uncached calls.

## Save and Load

 Calibrated programs save and load like any DSPy program, including the fitted decision parameters:

## Use the TypeSafe Backend (Optional)

 If you have a TypeSafe API key, you can switch the same predictor to the System One backend for probability-native inference:

The same signature, demos, and field parameters work with both backends. The difference is in how probabilities are produced:

- **Generative LMs** : DSPy asks the LM to generate probability evidence as structured JSON, then decodes it locally.
- **TypeSafe System One** : The API returns calibrated probabilities directly. No generation settings (temperature, etc.) are supported.

You can even mix backends within a single program — use Jev for critical decisions and a generative LM for free-text outputs.

## Native Annotations

 If you want the Python result to be a plain `bool` or `Literal` member (not the rich `Noul`/`Choice` object), use `Annotated`:

With `Annotated[bool, Urgent]`, the result is a plain `bool` — you still get probability-backed decisions and threshold tuning, but `result.urgent` is `True` or `False` directly.

## Rich Criteria

 For complex decisions, you can attach structured criteria that guide the LM’s judgment:

Criteria must be valid JSON (strings, dicts, lists, or null). For `Noul`, the keys are `"true"` and `"false"`. For `Score`, use a list matching the number of levels. For `Choice`, use a dict mapping option values to their criteria.

## Putting It All Together

 Here’s a complete end-to-end example:

## FAQ

 **Q: Do I need a TypeSafe API key?** No. Decision types work with any generative LM that DSPy supports. The TypeSafe backend is an optional, probability-native alternative.

**Q: Can I mix decision outputs with regular string outputs?** Yes. You can have `summary: str` alongside `urgent: Noul[...]` in the same signature. The generative LM handles both; TypeSafe requires all outputs to be decision types.

**Q: What’s the difference between `Noul` and just returning a `bool`?** A bare `bool` output uses direct generation — the LM picks True or False. A `Noul` output asks the LM for a probability and applies a threshold locally. This gives you confidence scores and tunable boundaries. You can opt a bare `bool` into evidence decoding by adding it to `predict.fields`.

**Q: How many training examples does ReAnchor need?** ReAnchor requires a nonempty trainset and uses up to 5-fold cross-validation (fewer folds for smaller sets). More data gives more reliable calibration. Since it reuses cached LM calls, adding more examples is cheap after the first pass.

**Q: Can I use this with async?** Yes. `TypeSafe` supports `acall()` for async inference. Decision types work with DSPy’s async support.

# Citations

1. Source page: https://dspy.ai/current/tutorials/jev_decisions
