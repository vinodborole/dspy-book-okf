---
type: Web Page
title: Changing modules - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/getting-started/changing-modules
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Change inference strategies by changing the module

In our previous examples, we used the `Predict` module to execute our signature.

Other modules define different strategies for executing a task, and trying them out is very simple:

Which yields:

Instead of `Predict` we used `ChainOfThought`, a module that prompts the LM to reason before delivering a final answer. Our signature, the class-based `HaikuBot`, is the same. The function call, identical.

But behind the scenes, DSPy modified our signature to prompt the model to reason before producing its final poem. The newly added signature output field, `reasoning`, is now populated by the model’s rationale.

We can view this by printing `result.reasoning`:

`ChainOfThought` is simple, but nicely demonstrates the flexibility of DSPy’s module layer. [Other modules](../../diving-deeper/built-in-module-variants/) implement other strategies, including model ensembles, coding sandboxes, and *tool-calling*, which the next section explores.

**Next:** [Tools with ReAct →](../react-and-tools/)

# Citations

1. Source page: https://dspy.ai/getting-started/changing-modules
