---
type: Web Page
title: Cache - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/tutorials/cache
timestamp: '2026-07-09T12:16:40.130937+00:00'
---

# Use and Customize DSPy Cache

In this tutorial, we will explore the design of DSPy’s caching mechanism and demonstrate how to effectively use and customize it.

## DSPy Cache Structure

DSPy’s caching system is architected in three distinct layers:

- **In-memory cache**: Implemented using- `cachetools.LRUCache`, this layer provides fast access to frequently used data.
- **On-disk cache**: Leveraging- `diskcache.FanoutCache`, this layer offers persistent storage for cached items.
- **Prompt cache (Server-side cache)**: This layer is managed by the LLM service provider (e.g., OpenAI, Anthropic).

While DSPy does not directly control the server-side prompt cache, it offers users the flexibility to enable, disable, and customize the in-memory and on-disk caches to suit their specific requirements.

## Using DSPy Cache

By default, both in-memory and on-disk caching are automatically enabled in DSPy. No specific action is required to start using the cache. When a cache hit occurs, you will observe a significant reduction in the module call’s execution time. Furthermore, if usage tracking is enabled, the usage metrics for a cached call will be `None`.

Consider the following example:

A sample output looks like:

## Using Provider-Side Prompt Caching

In addition to DSPy’s built-in caching mechanism, you can leverage provider-side prompt caching offered by LLM providers like Anthropic and OpenAI. This feature is particularly useful when working with modules like `dspy.ReAct()` that send similar prompts repeatedly, as it reduces both latency and costs by caching prompt prefixes on the provider’s servers.

You can enable prompt caching by passing the `cache_control_injection_points` parameter to `dspy.LM()`. This works with supported providers like Anthropic and OpenAI. For more details on this feature, see the [LiteLLM prompt caching documentation](https://docs.litellm.ai/docs/tutorials/prompt_caching#configuration).

This is especially beneficial when:

- Using `dspy.ReAct()`with the same instructions
- Working with long system prompts that remain constant
- Making multiple requests with similar context

## Restricting Pickle Deserialization

By default, DSPy’s on-disk cache uses Python’s `pickle` for serialization. While this handles arbitrary Python objects, `pickle.load` can execute arbitrary code – meaning a corrupted or malicious cache file could be dangerous.

DSPy provides an opt-in `restrict_pickle` mode that restricts which types the cache is allowed to deserialize:

When enabled, the cache only allows:

- **LiteLLM and OpenAI response types**(- `litellm.types.*`,- `openai.types.*`) – the pydantic data models that DSPy caches for LM calls, embeddings, and the Responses API.
- **NumPy array reconstruction helpers**– the specific internal functions needed to deserialize- `numpy.ndarray`(used by embedding caches).
- **User-registered types**via- `safe_types`– any additional types you explicitly trust.

If you cache custom types (dataclasses, pydantic models, etc.), register them:

If a type is missing from the allowlist, the cache treats it as a miss and returns `None`. The log message will name the exact type that was rejected:

### Nested types

If your registered type contains nested custom types, you must register all of them. For example, if `MyResult` contains a `Metadata` field, register both:

The error message will tell you exactly which nested type is missing.

## Disabling/Enabling DSPy Cache

There are scenarios where you might need to disable caching, either entirely or selectively for in-memory or on-disk caches. For instance:

- You require different responses for identical LM requests.
- You lack disk write permissions and need to disable the on-disk cache.
- You have limited memory resources and wish to disable the in-memory cache.

DSPy provides the `dspy.configure_cache()` utility function for this purpose. You can use the corresponding flags to control the enabled/disabled state of each cache type:

In additions, you can manage the capacity of the in-memory and on-disk caches:

Please note that `disk_size_limit_bytes` defines the maximum size in bytes for the on-disk cache, while `memory_max_entries` specifies the maximum number of entries for the in-memory cache.

## Understanding and Customizing the Cache

In specific situations, you might want to implement a custom cache, for example, to gain finer control over how cache keys are generated. By default, the cache key is derived from a hash of all request arguments sent to `litellm`, excluding credentials like `api_key`.

To create a custom cache, you need to subclass `dspy.clients.Cache` and override the relevant methods:

To ensure seamless integration with the rest of DSPy, it is recommended to implement your custom cache using the same method signatures as the base class, or at a minimum, include `**kwargs` in your method definitions to prevent runtime errors during cache read/write operations.

Once your custom cache class is defined, you can instruct DSPy to use it:

Let’s illustrate this with a practical example. Suppose we want the cache key computation to depend solely on the request message content, ignoring other parameters like the specific LM being called. We can create a custom cache as follows:

For comparison, consider executing the code below without the custom cache:

The time elapsed will indicate that the cache is not hit on the second call. However, when using the custom cache:

You will observe that the cache is hit on the second call, demonstrating the effect of the custom cache key logic.

# Citations

1. Source page: https://dspy.ai/tutorials/cache
