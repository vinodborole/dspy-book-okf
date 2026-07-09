---
type: Web Page
title: Expanding signatures - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/getting-started/expanding-signatures
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Expanding your signature: more inputs and adding types

## Adding additional inputs and outputs

Adding more fields to signature strings is as easy as separating field names with commas. For example, let’s update our program to accept two inputs, a `location` and `mood`:

This yields:

Defining additional outputs works identically:

Yielding:

## Hone your signature by mindfully naming your fields

The field names we choose aren’t just for our own readability. Unlike traditional programming, where variable names are purely identifiers, the LM reads them too, and uses them to infer what each input and output means.

If we replaced `"location, mood -> haiku"` with `"a, b -> c"`, the LM would be lost. Let’s try it:

This produces:

Our model doesn’t know we want a haiku, so it just makes a guess.

Naming is the cheapest optimization in DSPy. A field called `research_request` will produce better completions than one called `request`, with no other changes. Signatures are easy to edit; take advantage.

## Typing your fields yields more reliable programs

We can add more specificity to our task by *typing* our fields using the format `name: type`. For example, the signature `"location, mood, contains_pun: bool -> haiku"` accepts a boolean to indicate whether we want our poem to include a pun:

Which yields:

Inline types instruct DSPy to coerce the LM’s output into the types we ask for, and surface clear warnings when they can’t. This catches a class of silent failures that prompt-only systems hide.

Types also let us communicate structural details that are easier to express in code than in natural language. [Richer types](../../diving-deeper/signatures-in-depth/) – like Pydantic models, `TypedDicts`, or `dataclasses` – can pack plenty of details that help LMs correctly complete a task.

This is especially helpful when typing output fields. For example, if we wanted to modify our program to generate several haikus we could make our output field name plural and type it as a `list[str]`:

Which yields:

Once a program accrues several fields or we want to add nuanced instructions, it’s likely time to graduate to a class-based signature.

**Next:** [Class-based signatures →](../class-based-signatures/)

# Citations

1. Source page: https://dspy.ai/getting-started/expanding-signatures
