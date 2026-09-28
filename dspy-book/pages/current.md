---
type: Web Page
title: Overview - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/current
timestamp: '2026-09-28T13:20:17.892585+00:00'
---

DSPy 3.4.0 — PythonInterpreter improvements, faster GEPA, and MCP v2 compatibility · 

 [learn more →](https://github.com/stanfordnlp/dspy/releases)
#  Program, don’t prompt,

your LLMs. 

 DSPy is a Python framework for building AI systems. Express your tasks as structured signatures, not prompts, to produce maintainable, modular, and optimizable programs.

**extract_events.py**

 12345 678910 

 lm = dspy.LM("openai/gpt-5.4-nano")

 class ExtractEvent(dspy.Signature):

  """Extract event details from an email."""

  email: str = dspy.InputField()

  event_name: str = dspy.OutputField()

  date: str = dspy.OutputField()

 extract = dspy.Predict(ExtractEvent)

 extract(email=inbox_message)

 output

 Prediction(

  event_name="Team Offsite",

  date="Thursday, June 5"

 )

 5.2M+

 monthly downloads

 461+

 contributors

 38k

 github stars

 in production at

 ## Compose programs with reusable primitives.

### Signatures

Declare your task.

 Define your task as typed inputs and outputs instead of managing messy prompts. Portable, maintainable, and easy to iterate on.

class Triage(dspy.Signature):

  """Route a support ticket."""

  ticket: str = dspy.InputField()

  urgency: Literal["low", "high"] = dspy.OutputField()

  team: str = dspy.OutputField()

 ### Modules

Same interface, different strategy.

 Modules control how your signature executes. Reason, run ensembles, use tools, add a REPL, and more without rewriting your task.

# Direct completion

 classify = dspy.Predict(Triage)

 # Add step-by-step reasoning

 classify = dspy.ChainOfThought(Triage)

 # Add tools and a reasoning loop

 classify = dspy.ReAct(Triage, tools=[search])

 ### Optimizers

Compile your program against a metric.

 Give DSPy examples and a scoring function. It tunes your prompts automatically until quality converges.

tp = dspy.GEPA(

  metric=semantic_f1,

  auto="medium")

 opt = tp.compile(rag, trainset)

 # Before: 0.41 F1

 # After: 0.63 F1

 opt.save("rag.v2.json")

 ## Define a task. Grow it into a system.

class Extract(dspy.Signature):

  """Extract contact info."""

  message: str = dspy.InputField()

  name: str = dspy.OutputField()

  email: Optional[str] = dspy.OutputField()

  intent: Literal[

  "meeting", "intro", "follow-up"

  ] = dspy.OutputField()

 extract = dspy.Predict(Extract)

 extract(message="I'm Sarah"

  "(sarah@acme.co). Meet Thursday?")

 outputstdout

 Prediction(

  name="Sarah",

  email="sarah@acme.co",

  intent="meeting"

 )

 def search(query: str) -> list[str]:

  """Search a knowledge base."""

  return kb.query(query, k=3)

 def calc(expr: str) -> float:

  """Evaluate a math expression."""

  return dspy.PythonInterpreter({}).execute(expr)

 agent = dspy.ReAct(

  "question -> answer",

  tools=[search, calc])

 agent(question="GDP per capita of France?")

 outputstdout

 # thought 1: I need France's GDP and population.

 # action 1: search("France GDP") → ...

 # thought 2: Now divide GDP by population.

 # action 2: calc("3.13e12 / 68e6") → 46029.4

 Prediction(answer="$46,029")

 class FactCheck(dspy.Module):

  def __init__(self):

  self.find = dspy.ChainOfThought(

  "article -> claims: list[str]")

  self.verify = dspy.ChainOfThought(

  "claim, source -> verdict")

  def forward(self, article):

  found = self.find(article=article)

  return [

  self.verify(claim=c, source=article)

  for c in found.claims]

 outputstdout

 # >>> FactCheck()(article=news_article)

 [Prediction(verdict="supported"),

  Prediction(verdict="unsupported"),

  Prediction(verdict="supported")]

 class AnalyzeChart(dspy.Signature):

  """Describe the trend and key data points in a chart."""

  chart: dspy.Image = dspy.InputField()

  title: str = dspy.OutputField()

  trend: str = dspy.OutputField()

  data_points: list[dict] = dspy.OutputField()

 analyze = dspy.Predict(AnalyzeChart)

 analyze(chart=dspy.Image("quarterly_revenue.png"))

 outputstdout

 Prediction(

  title="Quarterly Revenue (2024)",

  trend="Steady growth, Q3 dip, strong Q4 recovery",

  data_points=[{"q": "Q1", "rev": "$4.2M"}, ...]

 )

 optimizer = dspy.GEPA(

  metric=accuracy, auto="medium")

 optimized = optimizer.compile(

  extract, trainset=labeled_emails)

 optimized.save("extract_v2.json")

 outputstdout

 # Baseline 62% (gpt-5.4-mini, zero-shot)

 # Optimized 89% (gpt-5.4-mini + GEPA compile)

 # Cost $2.18 · 200 examples

 # Saved to → extract_v2.json

 ## Built in the open, since Dec 2022.

DSPy started at Stanford NLP and grew into a research community. New optimizers and module types land here first — then show up in production systems at companies you’ve heard of.

Dec 2025

 Recursive Language Models

  Jul 2025

 GEPA: Reflective Prompt Evolution

  Jul 2024

 BetterTogether: Fine-Tuning + Prompt Opt.

  Jun 2024

 MIPROv2: Optimizing Instructions & Demos

  Feb 2024

 STORM: Writing Wikipedia-like Articles

  Oct 2023

 DSPy: Compiling Declarative LM Calls

  Dec 2022

 Demonstrate-Search-Predict

   ### DSPy in production

Metadata extraction across all shops; ~550× cost reduction

 Optimized Dash relevance judge for ranking and evaluation

 Prompt migration from larger to smaller models on Amazon Nova

 Multiple chatbot use cases on Databricks

 Code repair pipeline using code LLMs to synthesize diffs

 LM judges, RAG, classification, and customer solutions

 Evolutionary self-improvement for the Hermes agent

 See all companies using DSPy in production

# Citations

1. Source page: https://dspy.ai/current
