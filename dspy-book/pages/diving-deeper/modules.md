---
type: Web Page
title: 'Modules: composing your own - DSPy'
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/diving-deeper/modules
timestamp: '2026-07-07T10:31:54.390135+00:00'
---

# Modules: composing your own

## Intent

A `dspy.Module` is the unit of composition: subclass it, define some sub-modules in `__init__`, and write a `forward()` that pipes inputs through them. Modules are how custom logic — control flow, branching, multiple LM calls, post-processing — fits into the DSPy ecosystem in a way that optimizers, save/load, settings overrides, and parallel execution all keep working.

Read this when you’ve outgrown `Predict` / `ChainOfThought` / `ReAct` on their own and want to compose them, or when you want to understand what `__call__` does to your `forward()` behind the scenes.

## Design decisions

### 1. Subclass and override `forward()`

That’s the contract: define sub-modules in `__init__`, do the work in `forward`, let `__call__` wrap it. The split exists because the framework needs predictable places to attach infrastructure. `__init__` is where sub-modules are visible for tree-walking and serialization; `forward` is where the runtime concerns (settings stack, usage tracking, callbacks) wrap your code. Mixing the two muddies what the optimizer sees and what `dump_state` saves.

### 2. Sub-modules are discovered by walking `self.__dict__`, not registered

Assignment is the registration. There’s no `register_module()` call and no inheritance trick. The tree-walk methods (`named_predictors`, `named_parameters`, `named_sub_modules`) inspect `self.__dict__` at call time and recurse into lists and dicts as well as direct attributes. Discovery being lazy means you don’t have to remember to register, but it also means a sub-module hidden inside a closure or a non-walked container is invisible to optimizers and to `dump_state`.

### 3. `Parameter` is the marker that flags an attribute as optimizer-visible

When you assign `self.max_iters = 5`, the tree walk skips it. When you assign `self.predict = dspy.Predict(...)`, it’s a `Parameter` and gets included. This keeps `dump_state` small and the optimizer’s search space focused — only the things that should learn are exposed. `Predict` is a `Parameter`; `Retrieve` is too; plain Python attributes are not.

### 4. `__call__` wraps `forward()` with infrastructure

Going through `__call__` is what pushes `self` onto `settings.caller_modules`, opens a usage-tracking context when `settings.track_usage` is true, and runs the callback decorator chain. Calling `forward()` directly bypasses all of that — usage doesn’t get tallied, callbacks don’t fire, the caller stack doesn’t update. The deprecation warning on direct `forward` calls is there for that reason. Always invoke a module as `module(...)`.

### 5. Async sits alongside sync via `acall()` / `aforward()`

`acall` mirrors `__call__`‘s infrastructure for async; `aforward` is the async counterpart to `forward`. A module that doesn’t define `aforward` falls back to running the sync `forward` in a worker thread via `asyncify`. That gives you a path from “sync only” to “fully async” without rewriting the world — async one sub-module at a time, leave the rest as-is.

### 6. Settings propagate via context, not constructor args

Sub-modules read `dspy.settings.lm` (and `.adapter`, `.callbacks`) at call time. The consequence: `with dspy.context(lm=other_lm): result = my_module(...)` swaps the LM for every sub-module inside, no rewiring needed. The price is implicit state — a sub-module’s behavior depends on whoever calls it, not only on what was assigned in `__init__`. The composition story is worth the tradeoff.

### 7. `_compiled=True` freezes a sub-module’s parameters from optimizers

`named_parameters` checks the `_compiled` flag and skips compiled children. This lets you optimize a sub-module on its own, save the result, drop it into a larger program, and run a fresh optimizer on the outer program without disturbing the inner one. You rarely set this by hand; teleprompts set it when they finish compiling.

### 8. The metaclass `ProgramMeta` enforces `super().__init__()`

`ProgramMeta` installs a fallback `_base_init` that runs before your `__init__`, so the bookkeeping (history, callbacks, the `_compiled` flag) is in place even if your subclass forgets `super().__init__()`. You won’t subclass or instantiate the metaclass; it’s a guardrail. When an error trace lands inside it, the cause is usually a missing `super().__init__()` call.

### 9. `deepcopy()` is hand-written

Modules can hold closures and callbacks that confuse Python’s default `copy.deepcopy`. `Module.deepcopy()` tries `copy.deepcopy` first and falls back to attribute-by-attribute copying on `TypeError`. Optimizers use this to fork candidate programs, and the fallback is what makes that reliable across the variety of things a `Module` subclass might hold.

## API walkthrough

Grouped by what you’re trying to do.

### Defining a module

`dspy.Module(callbacks=None)`

Subclass it. The base stores `callbacks`, `history`, and the `_compiled` flag — nothing more. The interesting behavior is in `__call__` and the tree-walk methods.

`Module.__call__(*args, **kwargs)`

Pushes `self` onto `settings.caller_modules`, opens a usage-tracking context if `settings.track_usage`, runs `forward` (decorated with `@with_callbacks`), and returns the `Prediction`. If `forward` returns something other than a `Prediction`, you’ll get a warning — usage tracking relies on the `Prediction` return type. Direct calls to `module.forward(...)` log a deprecation warning.

`Module.forward(**kwargs)`

What you override. Accept signature inputs as keyword arguments, do your work, return a `Prediction`. Don’t put settings reads in `__init__` — read them inside `forward` so the current LM and adapter are picked up at call time, not at construction time.

`Module.acall(*args, **kwargs)` / `Module.aforward(**kwargs)`

Async variants. `acall` is the public entry; `aforward` is what you override. If you don’t define `aforward`, `acall` falls back to running `forward` inside `asyncify`. Useful when one sub-module is async-only — you can `await` it from `aforward` without making the rest of the program async.

`ProgramMeta`

The metaclass on `Module`. You won’t subclass or instantiate it. It installs `_base_init` as a fallback so `super().__init__()` isn’t strictly required. When an error trace lands inside it, the cause is usually a missing `super().__init__()`.

### Walking the module tree

All of these rely on the same primitive: walk `self.__dict__`, recurse into lists and dicts, yield anything that matches the target type.

`Module.named_predictors()` / `Module.predictors()`

Discovers every `Predict` instance under this module. Used by optimizers to find what to optimize. Order is the order of discovery (depth-first through `__dict__`).

`Module.named_parameters()` / `Module.parameters()`

Discovers every `Parameter`. Honors `_compiled`: a sub-module marked compiled is skipped, and the optimizer leaves its parameters alone.

`Module.named_sub_modules(type_=BaseModule, skip_compiled=False)`

The generator the others build on. Use it directly when you want a custom walk — every `ReAct` instance, every sub-module of a specific user class, anything you can filter by `type_`.

`Module.map_named_predictors(func)`

Applies `func` to each predictor and replaces it in place. The hook teleprompts use when swapping in optimized predictors. You’ll rarely need it directly.

### Setting and getting the LM

`Module.set_lm(lm)`

Walks every predictor in the tree and assigns `lm`. Useful when you’ve built a program with one LM and want to re-evaluate it on another without changing settings globally.

`Module.get_lm()` → `LM`

Returns the LM if every predictor agrees on one. Raises `ValueError` on a mix. A sanity check before fine-tuning or saving, when you want to confirm the whole program is on one model.

### State and copies

The save/load surface. The full lifecycle — JSON vs PKL, full-program vs state-only, `save_program=True` — lives in Saving and loading; the methods here are what that page is about.

`Module.dump_state(json_mode=True)` / `Module.load_state(state, allow_unsafe_lm_state=False)`

Round-trips the optimizer-visible state: per-predictor demos, traces, signature instructions, LM config. Not the Python class itself.

`Module.save(path, save_program=False, modules_to_serialize=None)` / `Module.load(path, allow_pickle=False, allow_unsafe_lm_state=False)`

The user-facing pair. `save_program=True` cloudpickles the whole module to a directory; the default writes state-only JSON or PKL.

`Module.deepcopy()`

Tries `copy.deepcopy`, falls back to attribute-by-attribute copy on `TypeError`. Optimizers use this to fork candidate programs.

`Module.reset_copy()`

`deepcopy()` then `reset()` on every parameter. Use it for a fresh program — same structure, no optimizer state.

### Batch execution

`Module.batch(examples, num_threads=None, max_errors=None, return_failed_examples=False, ...)`

Runs the module against many examples in parallel. Internally wraps `dspy.Parallel`, which snapshots `dspy.settings` and re-applies the snapshot inside each worker thread (so an outer `dspy.context(lm=...)` is honored). Returns predictions in input order. Use it for evaluation, demo collection, and any embarrassingly-parallel inference.

### Debugging

`Module.history` / `Module.inspect_history(n=1, file=None)`

`history` is the list of LM call records attached to this module. `inspect_history` pretty-prints the last `n`. Both are excluded from pickled and JSON state, so a multi-megabyte history doesn’t ride along with your saved program.

`Module.callbacks`

Registered callback handlers. Also excluded from saved state.

### The return type

`dspy.Prediction(**fields)`

A field container. Access with `result.haiku` or `result["haiku"]`. Inherits `dspy.Example`‘s storage but drops the input/output split.

`dspy.Prediction.from_completions(list_or_dict, signature=None)`

Builds a Prediction from multiple LM completions. `BestOfN`, `MultiChainComparison`, and any module that samples more than once uses this.

`Prediction.completions`

A `Completions` object when the prediction came from `n > 1` samples; `pred.completions[i]` returns the i-th completion as its own `Prediction`. `None` for single-sample calls.

`Prediction.get_lm_usage()` / `Prediction.set_lm_usage(tokens_dict)`

Token-usage accounting. Populated when the prediction ran under `dspy.settings(track_usage=True)`.

**Comparison to  dspy.Example.** Same storage layer; different intent. 

`Example` is for training data — it carries `_demos` and `_input_keys`, and `.with_inputs` returns a filtered copy. `Prediction` is for module outputs — no input/output split, plus completions and usage. Use `Example` for what goes in, `Prediction` for what comes out.## Cross-links

- Each built-in module subclass has its own DD page: see the predictor zoo, ReAct, and the optimizer-specific pages.
- Saving and loading covers the persistence story in detail.
- Settings and context() covers how LM / adapter / callback overrides propagate into a module’s sub-modules.

# Citations

1. Source page: https://dspy.ai/diving-deeper/modules
