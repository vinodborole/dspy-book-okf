---
type: Web Page
title: Privacy-Conscious Delegation - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/tutorials/papillon
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Privacy-Conscious Delegation

Please refer to [this tutorial from the PAPILLON authors](https://colab.research.google.com/github/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb) using DSPy.

This tutorial demonstrates a few aspects of using DSPy in a more advanced context:

- It builds a multi-stage `dspy.Module`that involves a small local LM using an external tool.
- It builds a multi-stage *judge*in DSPy, and uses it as a metric for evaluation.
- It uses this judge for optimizing the `dspy.Module`, using a large model as a teacher for a small local LM.

# Citations

1. Source page: https://dspy.ai/tutorials/papillon
