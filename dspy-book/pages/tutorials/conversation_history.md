---
type: Web Page
title: Managing Conversation History - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/tutorials/conversation_history
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Managing Conversation History

Maintaining conversation history is a fundamental feature when building AI applications such as chatbots. While DSPy does not provide automatic conversation history management within `dspy.Module`, it offers the `dspy.History` utility to help you manage conversation history effectively.

## Using `dspy.History` to Manage Conversation History

The `dspy.History` class can be used as an input field type, containing a `messages: list[dict[str, Any]]` attribute that stores the conversation history. Each entry in this list is a dictionary with keys corresponding to the fields defined in your signature. See the example below:

There are two key steps when using the conversation history:

- **Include a field of type**- `dspy.History`in your Signature.
- **Maintain a history instance at runtime, appending new conversation turns to it.**Each entry should include all relevant input and output field information.

A sample run might look like this:

Notice how each user input and assistant response is appended to the history, allowing the model to maintain context across turns.

The actual prompt sent to the language model is a multi-turn message, as shown by the output of `dspy.inspect_history`. Each conversation turn is represented as a user message followed by an assistant message.

## History in Few-shot Examples

You may notice that `history` does not appear in the input fields section of the prompt, even though it is listed as an input field (e.g., “2. `history` (History):” in the system message). This is intentional: when formatting few-shot examples that include conversation history, DSPy does not expand the history into multiple turns. Instead, to remain compatible with the OpenAI standard format, each few-shot example is represented as a single turn.

For example:

The resulting history will look like this:

As you can see, the few-shot example does not expand the conversation history into multiple turns. Instead, it represents the history as JSON data within its section:

This approach ensures compatibility with standard prompt formats while still providing the model with relevant conversational context.

# Citations

1. Source page: https://dspy.ai/tutorials/conversation_history
