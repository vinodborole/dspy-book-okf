---
type: Web Page
title: Async - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/tutorials/async
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Async DSPy Programming

DSPy provides native support for asynchronous programming, allowing you to build more efficient and scalable applications. This guide will walk you through how to leverage async capabilities in DSPy, covering both built-in modules and custom implementations.

## Why Use Async in DSPy?

Asynchronous programming in DSPy offers several benefits: - Improved performance through concurrent operations - Better resource utilization - Reduced waiting time for I/O-bound operations - Enhanced scalability for handling multiple requests

## When Should I use Sync or Async?

Choosing between synchronous and asynchronous programming in DSPy depends on your specific use case. Here’s a guide to help you make the right choice:

Use Synchronous Programming When

- You’re exploring or prototyping new ideas
- You’re conducting research or experiments
- You’re building small to medium-sized applications
- You need simpler, more straightforward code
- You want easier debugging and error tracking

Use Asynchronous Programming When:

- You’re building a high-throughput service (high QPS)
- You’re working with tools that only support async operations
- You need to handle multiple concurrent requests efficiently
- You’re building a production service that requires high scalability

### Important Considerations

While async programming offers performance benefits, it comes with some trade-offs:

- More complex error handling and debugging
- Potential for subtle, hard-to-track bugs
- More complex code structure
- Different code between ipython (Colab, Jupyter lab, Databricks notebooks, …) and normal python runtime.

We recommend starting with synchronous programming for most development scenarios and switching to async only when you have a clear need for its benefits. This approach allows you to focus on the core logic of your application before dealing with the additional complexity of async programming.

## Using Built-in Modules Asynchronously

Most DSPy built-in modules support asynchronous operations through the `acall()` method. This method
maintains the same interface as the synchronous `__call__` method but operates asynchronously.

Here’s a basic example using `dspy.Predict`:

### Working with Async Tools

DSPy’s `Tool` class seamlessly integrates with async functions. When you provide an async
function to `dspy.Tool`, you can execute it using `acall()`. This is particularly useful
for I/O-bound operations or when working with external services.

#### Using Async Tools in Synchronous Contexts

If you need to call an async tool from synchronous code, you can enable automatic async-to-sync conversion:

For more details on async tools, see the [Tools documentation](../../learn/programming/tools/#async-tools).

Note: When using `dspy.ReAct` with tools, calling `acall()` on the ReAct instance will automatically
execute all tools asynchronously using their `acall()` methods.

## Creating Custom Async DSPy Modules

To create your own async DSPy module, implement the `aforward()` method instead of `forward()`. This method
should contain your module’s async logic. Here’s an example of a custom module that chains two async operations:

# Citations

1. Source page: https://dspy.ai/tutorials/async
