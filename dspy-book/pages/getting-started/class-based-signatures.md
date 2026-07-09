---
type: Web Page
title: Class-based signatures - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/getting-started/class-based-signatures
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Writing a class-based signature

## Adding instructional nuance with class-based signatures

A class-based signature details the same structure a string signature can, but adds a few levers for adding additional nuance. Here’s our haiku writer string signature rephrased as a class-based signature:

We have the same fields (`location`, `mood`, and the `haiku` output) typed as strings, but we now have the ability to add descriptions to each. Field descriptions allow us to add nuance that might not fit within a field name.

Class-based signatures also let us write a docstring, the string at the start of the class, which DSPy uses as task instructions when preparing prompts.

We pass class-based signatures to modules just like we passed string signatures:

This call renders and sends similar instructions to the LM, with two exceptions. The docstring and any field descriptions are used when building the system instructions, like so:

Signature docstrings and field descriptions are optional, but they are handy levers when field names don’t provide sufficient context for a task. However, resist the urge to restate what the signature already says or write prescriptive tutorials. Expansive rules, watch-outs, and guidance are what optimizers are for (more on that later).

Though it’s worth noting: field descriptions are not touched by the optimizers, so mind your naming. A poorly chosen field name can’t be adjusted by optimizers.

## Tightening signature fields with richer types

Sometimes a plain `str` is too loose. When a value should come from a small fixed set, we’d rather pin it down so the LM (and the caller) can’t drift outside it. This is the unit-test framing from Section 3 made stricter: not just *some string*, but *one of these specific strings*.

We can use `typing`, from Python’s standard lib, to add richer types.

Typing `season` as `Literal["spring", "summer", "autumn", "winter"]` does exactly that. DSPy now accepts only those four values, both at call time and when parsing the LM’s response.

This yields:

But if we pass `season="fall"`, we get a warning explaining the mismatch:

See [Signatures in depth](../../diving-deeper/signatures-in-depth/) for the rest of the surface — output validators, multi-output composition, and richer Pydantic patterns.

**Next:** [Changing modules →](../changing-modules/)

# Citations

1. Source page: https://dspy.ai/getting-started/class-based-signatures
