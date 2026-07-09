---
type: Web Page
title: Streaming - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/tutorials/streaming
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Streaming

In this guide, we will walk you through how to enable streaming in your DSPy program. DSPy Streaming consists of two parts:

- **Output Token Streaming**: Stream individual tokens as they’re generated, rather than waiting for the complete response.
- **Intermediate Status Streaming**: Provide real-time updates about the program’s execution state (e.g., “Calling web search…”, “Processing results…”).

## Output Token Streaming

DSPy’s token streaming feature works with any module in your pipeline, not just the final output. The only requirement is that the streamed field must be of type `str`. To enable token streaming:

- Wrap your program with `dspy.streamify`
- Create one or more `dspy.streaming.StreamListener`objects to specify which fields to stream

Here’s a basic example:

To consume the streamed output:

This will produce output like:

Note: Since `dspy.streamify` returns an async generator, you must use it within an async context. If you’re using an environment like Jupyter or Google Colab that already has an event loop (async context), you can use the generator directly.

You may have noticed that the above streaming contains two different entities: `StreamResponse`
and `Prediction.` `StreamResponse` is the wrapper over streaming tokens on the field being listened to, and in
this example it is the `answer` field. `Prediction` is the program’s final output. In DSPy, streaming is
implemented in a sidecar fashion: we enable streaming on the LM so that LM outputs a stream of tokens. We send these
tokens to a side channel, which is being continuously read by the user-defined listeners. Listeners keep interpreting
the stream, and decides if the `signature_field_name` it is listening to has started to appear and has finalized.
Once it decides that the field appears, the listener begins outputting tokens to the async generator users can
read. Listeners’ internal mechanism changes according to the adapter behind the scene, and because usually
we cannot decide if a field has finalized until seeing the next field, the listener buffers the output tokens
before sending to the final generator, which is why you will usually see the last chunk of type `StreamResponse`
has more than one token. The program’s output is also written to the stream, which is the chunk of `Prediction`
as in the sample output above.

To handle these different types and implement custom logic:

### Understand `StreamResponse`

`StreamResponse` (`dspy.streaming.StreamResponse`) is the wrapper class of streaming tokens. It comes with 3
fields:

- `predict_name`: the name of the predict that holds the- `signature_field_name`. The name is the same name of keys as you run- `your_program.named_predictors()`. In the code above because- `answer`is from the- `predict`itself, so the- `predict_name`shows up as- `self`, which is the only key as your run- `predict.named_predictors()`.
- `signature_field_name`: the output field that these tokens map to.- `predict_name`and- `signature_field_name`together form the unique identifier of the field. We will demonstrate how to handle multiple fields streaming and duplicated field name later in this guide.
- `chunk`: the value of the stream chunk.

### Streaming with Cache

When a cached result is found, the stream will skip individual tokens and only yield the final `Prediction`. For example:

### Streaming Multiple Fields

You can monitor multiple fields by creating a `StreamListener` for each one. Here’s an example with a multi-module program:

The output will look like:

### Streaming the Same Field Multiple Times (as in dspy.ReAct)

By default, a `StreamListener` automatically closes itself after completing a single streaming session.
This design helps prevent performance issues, since every token is broadcast to all configured stream listeners,
and having too many active listeners can introduce significant overhead.

However, in scenarios where a DSPy module is used repeatedly in a loop—such as with `dspy.ReAct` — you may want to stream
the same field from each prediction, every time it is used. To enable this behavior, set allow_reuse=True when creating
your `StreamListener`. See the example below:

In this example, by setting `allow_reuse=True` in the StreamListener, you ensure that streaming for “next_thought” is
available for every iteration, not just the first. When you run this code, you will see the streaming tokens for `next_thought`
output each time the field is produced.

#### Handling Duplicate Field Names

When streaming fields with the same name from different modules, specify both the `predict` and `predict_name` in the `StreamListener`:

The output will be like:

## Intermediate Status Streaming

Status streaming keeps users informed about the program’s progress, especially useful for long-running operations like tool calls or complex AI pipelines. To implement status streaming:

- Create a custom status message provider by subclassing `dspy.streaming.StatusMessageProvider`
- Override the desired hook methods to provide custom status messages
- Pass your provider to `dspy.streamify`

Example:

Available hooks:

- lm_start_status_message: status message at the start of calling dspy.LM.
- lm_end_status_message: status message at the end of calling dspy.LM.
- module_start_status_message: status message at the start of calling a dspy.Module.
- module_end_status_message: status message at the end of calling a dspy.Module.
- tool_start_status_message: status message at the start of calling dspy.Tool.
- tool_end_status_message: status message at the end of calling dspy.Tool.

Each hook should return a string containing the status message.

After creating the message provider, just pass it to `dspy.streamify`, and you can enable both
status message streaming and output token streaming. Please see the example below. The intermediate
status message is represented in the class `dspy.streaming.StatusMessage`, so we need to have
another condition check to capture it.

Sample output:

## Synchronous Streaming

By default calling a streamified DSPy program produces an async generator. In order to get back
a sync generator, you can set the flag `async_streaming=False`:

# Citations

1. Source page: https://dspy.ai/tutorials/streaming
