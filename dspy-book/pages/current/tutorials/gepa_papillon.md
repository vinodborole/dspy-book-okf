---
type: Web Page
title: GEPA for Privacy-Conscious Delegation - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/current/tutorials/gepa_papillon
timestamp: '2026-09-28T13:20:17.892585+00:00'
---

# Tutorial: GEPA for Privacy-Conscious Delegation

 In this tutorial, we optimize the [PAPILLON](https://dspy.ai/tutorials/papillon/) program with `dspy.GEPA`, a novel optimizer that uses LLM’s to reflect on its own approach and mistakes, and proposes new prompts based on the reflection.

PAPILLON is a system for privacy-preserving delegation, a small LM (typically local-hosted) to use a larger “untrusted” external LLM, which is more powerful but may save your private data, to balance high-quality and private chat.

For simplicity, we will use “gpt-4.1-nano” as the small LM, and “gpt-4.1-mini” as the large, “untrusted” LM.

## Recommended: Set up MLflow Autologging to understand what's happening under the hood.

### MLflow DSPy Integration

 [MLflow](https://mlflow.org/) is an LLMOps tool that natively integrates with DSPy and offer explainability and experiment tracking. MLflow’s autologging capability automatically tracks progress of GEPA optimization, as well as visualizes prompts and module executions as traces to understand the DSPy’s behavior better. You can set up MLflow easily by following the four steps below.

**Visualize module executions as traces**

**Automatically track optimization progress and results**

**Setup MLflow**

1. Install MLflow

1.  Start MLflow UI in a separate terminal
2.  Connect the notebook to MLflow
3.  Enabling autologging.

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

### The PAPILLON Program

 ```
Loaded 225 training examples, 225 dev examples, and 214 test examples.
```
### Define metrics for success in this task

 What does it mean for a PAPILLON system to be successful?

1. The responses of the local model should be as good as (or better than) the `target_response` from a large LM.
2. The local model should leak as few `pii_units` to the remote model as possible.

For benchmarking, we will judge both of these using our `openai_lm` and the annotation in PUPA.

With these judges, we can now define the metric for evaluation.

### Evaluate unoptimized PAPILLON

 Let’s now use the PUPA data and the judges above to evaluate the unoptimized version of our PAPILLON pipeline!

```
  0%|                                                                                                                                                                                          | 0/214 [00:00<?, ?it/s]Average Metric: 107.87 / 139 (77.6%):  65%|█████████████████████████████████████████████████████████████████████████████████████████▋                                                | 139/214 [00:13<00:05, 12.75it/s]
2025/08/12 18:31:42 WARNING dspy.clients.lm: LM response was truncated due to exceeding max_tokens=4000. You can inspect the latest LM interactions with `dspy.inspect_history()`. To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. You may also consider increasing the temperature (currently 0.0)  if the reason for truncation is repetition.
Average Metric: 163.71 / 214 (76.5%): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 214/214 [00:22<00:00,  9.57it/s]
2025/08/12 18:31:52 INFO dspy.evaluate.evaluate: Average Metric: 163.70979853479852 / 214 (76.5%)
```
|  | target_response | user_query | pii_str | llm_request | llm_response | response | compute_overall_score | 
|---|---|---|---|---|---|---|---|
| 0 | Subject: Follow-up on Endpoint Management Solution Discussion Dear... | rewrite in professional "as discusseed in last meeting with you an... | effem\|\|balaji\|\|balaji | Rewrite the following message in a professional tone, ensuring cla... | Subject: Follow-up on E-FEM Account Management and Solution Finali... | Subject: Clarification on E-FEM Account Management and Next Steps ... |  | 
| 1 | Кампания "Share a Coke" была запущена Coca-Cola в 2011 году в Авст... | расскажи про PR-кампанию Coca-Cola “Share a Coke" | coca-cola | Please provide a detailed summary of Coca-Cola's "Share a Coke" PR... | Certainly! Here is a detailed summary of Coca-Cola's "Share a Coke... | Кампания Coca-Cola "Share a Coke" — это очень успешная маркетингов... | ✔️ [1.000] | 
| 2 | Creating a Facebook ad campaign within a specific budget and targe... | give me a facebook ads campaign for the next 7 days for canva pro ... | jordan\|\|canva | Create a comprehensive Facebook ads campaign plan for selling Canv... | Certainly! Below is a comprehensive 7-day Facebook Ads campaign pl... | Certainly! Here's a 7-day Facebook ads campaign plan for selling C... | ✔️ [1.000] | 
| 3 | The delivery was scheduled for today, but due to a payment delay t... | The delivery supposed to be today. Seems our payment delay to Lvio... | seagate\|\|lviv\|\|ms. chan | Please rewrite the following message in a professional and clear m... | The delivery was originally scheduled for today; however, due to a... | The delivery was originally scheduled for today; however, due to a... | ✔️ [0.833] | 
| 4 | Dear Ms. Amanda, I hope this email finds you well. I wanted to bri... | write a email to Ms, Amanda ; tell her, we have a way to overcome ... | india\|\|amanda\|\|hermann(germany)\|\|china\|\|vims(france) | Write a professional email to Ms. Amanda explaining that we have a... | Subject: Proposal to Overcome Certification Challenges and Product... | Dear Ms. Amanda, I hope you are well. We have identified a way to ... | ✔️ [0.700] | 

```
EvaluationResult(score=76.5, results=<list of 214 results>)
```
### Optimize PAPILLON with `dspy.GEPA`

 GEPA is a *reflective* prompt optimizer, and it’s strength lies in being able to view textual feedback from the DSPy program’s execution and evaluation pipelines, which provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score. Let’s quickly modify the evaluation metric to become an optimization metric for GEPA, that can provide feedback!

In this case, since the evaluation metric is an aggregate of 2 distinct scores, “quality” score and “leakage” score, the feedback metric can be as simple as showing what the quality and leakage scores are, so GEPA can reflect on what needs to be improved!

Notice how the metric function we had already defined provided all the components we need for this feedback function! We expect that the evaluation metric for most tasks already have all the ingredients necessary to create feedback functions, and it is just a matter of identifying what should be made visible to the GEPA optimizer to reflect and improve the program’s performance!

Let’s use GEPA on PAPILLON. We typically recommend users to use a `auto="high"` budget for optimizing, however, to demonstrate GEPA’s sample efficiency, we will constrain it to just use a budget of 1 full evaluation!

```
2025/08/12 18:38:42 INFO dspy.teleprompt.gepa.gepa: Running GEPA for approx 450 metric calls of the program. This amounts to 1.00 full evals on the train+val set.
2025/08/12 18:38:42 INFO dspy.teleprompt.gepa.gepa: Using 225 examples for tracking Pareto scores. You can consider using a smaller sample of the valset to allow GEPA to explore more diverse solutions within the same budget.
2025/08/12 18:38:58 INFO dspy.evaluate.evaluate: Average Metric: 179.6126262626263 / 225 (79.8%)
2025/08/12 18:38:58 INFO dspy.teleprompt.gepa.gepa: Iteration 0: Base program full valset score: 0.7982783389450057
2025/08/12 18:38:58 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Selected program 0 score: 0.7982783389450057
Average Metric: 2.00 / 3 (66.7%): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 13.20it/s]
2025/08/12 18:38:59 INFO dspy.evaluate.evaluate: Average Metric: 2.0 / 3 (66.7%)
2025/08/12 18:39:13 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Proposed new text for craft_redacted_request.predict: You are given a private user query as input. Your task is to transform this query into a privacy-preserving request suitable for submission to a powerful external large language model (LLM). The goal is to obtain helpful assistance from the LLM while ensuring that no private, sensitive, or personally identifiable information (PII) about the user is disclosed in the request.
Follow these steps:
1. **Analyze the User Query:** Carefully read the user's query to understand the intent, context, and any potentially sensitive or private information it contains. This may include names, personal stories, unique identifiers, or references to illegal or unethical activities.
2. **Reasoning Step:** Briefly explain your reasoning process. Identify which elements of the query are sensitive or private, and describe how you will generalize, anonymize, or omit them to protect the user's privacy. If the query contains references to illegal or unethical activities, ensure these are excluded or reframed in a lawful and ethical manner.
3. **Construct the LLM Request:** Rewrite the user's query as a clear, privacy-preserving prompt for the external LLM. 
    - Remove or generalize any PII or sensitive details (e.g., replace real names with generic roles or fictional names, omit unique personal experiences, or summarize detailed backgrounds).
    - If the query requests creative content (e.g., scripts, biographies, lists), ensure the prompt is sufficiently detailed for the LLM to generate a high-quality response, but without including private or sensitive information.
    - If the query includes illegal or unethical activities, exclude these from the LLM request and focus on lawful, ethical alternatives.
    - Maintain the original intent and utility of the user's request as much as possible.
4. **Output Format:** Your response should include two sections:
    - **reasoning:** A concise explanation of your privacy-preserving strategy for this specific query.
    - **llm_request:** The rewritten, privacy-preserving prompt to be sent to the external LLM.
**Additional Notes:**
- Strive for a balance between preserving user privacy and maintaining the quality and usefulness of the LLM request.
- Avoid over-generalizing to the point where the LLM cannot provide a meaningful response.
- Do not include any information in the LLM request that could be used to identify the user or any other real individual.
- If the user query is already privacy-preserving, you may use it as-is, but still provide a brief reasoning statement confirming this.
This process ensures that user privacy is protected while leveraging the capabilities of external language models for a wide range of tasks, including creative writing, information synthesis, and idea generation.
2025/08/12 18:40:01 INFO dspy.evaluate.evaluate: Average Metric: 3.0 / 3 (100.0%)
2025/08/12 18:45:52 INFO dspy.evaluate.evaluate: Average Metric: 189.65 / 225 (84.3%)
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: New program is on the linear pareto front
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Full valset score for new program: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Full train_val score for new program: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Individual valset scores for new program: [1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.7, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.875, 1.0, 1.0, 1.0, 1.0, 0.0, 0.6, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 0.875, 0.6666666666666667, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.75, 1.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.8333333333333334, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.5, 1.0, 0.8333333333333334]
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: New valset pareto front scores: [1.0, 0.5, 1.0, 1.0, 0.75, 1.0, 0.8333333333333334, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.7, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.75, 1.0, 1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.875, 1.0, 1.0, 1.0, 1.0, 0.5, 0.9, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 0.875, 0.6666666666666667, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.75, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8333333333333334, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.8333333333333334]
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Full valset pareto front score: 0.9004444444444444
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Updated valset pareto front programs: [{1}, {0, 1}, {1}, {1}, {0}, {0, 1}, {0}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0}, {1}, {0}, {0}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {1}, {0, 1}, {1}, {1}, {0, 1}, {1}, {0, 1}, {1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {0, 1}, {1}, {1}, {0, 1}, {0, 1}, {1}, {1}, {1}, {1}, {1}, {0}, {1}, {1}, {0, 1}, {0, 1}, {1}, {0, 1}, {1}, {0, 1}, {0, 1}, {1}, {1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0}, {1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {1}, {1}, {0, 1}, {0}, {1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {0}, {0, 1}, {0, 1}, {0, 1}, {1}, {0, 1}, {0, 1}, {1}, {0}, {0, 1}, {0, 1}, {0}, {0, 1}, {0, 1}]
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best valset aggregate score so far: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best program as per aggregate score on train_val: 1
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best program as per aggregate score on valset: 1
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best score on valset: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best score on train_val: 0.8428888888888889
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Linear pareto front program index: 1
2025/08/12 18:45:52 INFO dspy.teleprompt.gepa.gepa: Iteration 1: New program candidate index: 1
```
### Display the GEPA generated prompt

 Note that since we allowed GEPA the budget to only generate 1 candidate, it has updated the prompt for only one of the predictors

```
Task Description:
You will be given a private user query that may contain sensitive, personal, or identifying information. Your task is to transform this user query into a privacy-preserving prompt or request suitable for a powerful external large language model (LLM). The generated request should enable the external LLM to assist with the user’s original intent effectively, while ensuring that no private, personally identifiable, or sensitive information about the user is exposed or leaked.
Input Format:
- The user query is a free-form text input that may include explicit personal details, real names, descriptive traits, or specific events.
- The input may range from requests for creative writing, role descriptions, lists of hobbies, profiles, or any other user-generated content.
Detailed Task Requirements and Domain-Specific Considerations:
1. Privacy Preservation:
   - Do not include any directly identifying details, real names, or sensitive personal information in the generated request.
   - Replace or anonymize any personal data, including but not limited to names, dates, biographical details, or events tied to individuals.
   - Avoid leaking private user context or information that could be traced back to the user.
2. Maintaining Task Integrity:
   - The LLM request must capture the user’s original intention and task goals as faithfully as possible, phrased in a generalized and context-neutral way.
   - For content requests involving real-world entities (e.g., institutions, events), consider making the scenario fictionalized or anonymized if it involves potentially identifying or sensitive context.
   - For character or profile descriptions, abstract/remove intimate personal details but preserve key personality traits or narrative elements in a generalized manner.
3. Quality of Generated LLM Request:
   - The description should be clear, coherent, and sufficiently detailed to enable a relevant and helpful response from the LLM without needing further clarifications.
   - Avoid introducing new biases or assumptions outside the user’s input.
   - Exclude illegal or ethically sensitive activities where applicable, or transform them into acceptable analogues to maintain ethical standards.
4. Reasoning Component:
   - Alongside the final LLM request, provide a concise reasoning summary that explains how private data was identified and sanitized, the thought process behind generalization or anonymization, and how the final prompt preserves user intent while protecting privacy.
Generalizable Strategy:
- Identify sensitive or private details in the user query.
- Abstract or anonymize such details without losing essential meaning.
- Frame the request in terms of generic or fictional scenarios.
- Explicitly instruct the external LLM to avoid real or private details and to focus on a neutral, generic, or fictional perspective when relevant.
- Validate that no PII or identifiable content remains in the final prompt.
Outputs:
- A reasoning section summarizing privacy considerations and prompt construction approach.
- A final privacy-preserving LLM prompt/request ready to be sent to an external LLM.
This approach ensures strong privacy protection aligned with best practices while maximizing task quality and relevance.
```
```
Average Metric: 184.26 / 214 (86.1%): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 214/214 [00:13<00:00, 15.90it/s]
2025/08/12 18:27:34 INFO dspy.evaluate.evaluate: Average Metric: 184.2602564102564 / 214 (86.1%)
```
|  | target_response | user_query | pii_str | llm_request | llm_response | response | compute_overall_score | 
|---|---|---|---|---|---|---|---|
| 0 | Subject: Follow-up on Endpoint Management Solution Discussion Dear... | rewrite in professional "as discusseed in last meeting with you an... | effem\|\|balaji\|\|balaji | Please draft a professional, formal update regarding a previous di... | Subject: Update on Device Management Solution Discussion Dear Team... | Subject: Follow-up on Device Management Solution Discussion Dear T... | ✔️ [1.000] | 
| 1 | Кампания "Share a Coke" была запущена Coca-Cola в 2011 году в Авст... | расскажи про PR-кампанию Coca-Cola “Share a Coke" | coca-cola | Please provide an overview of a well-known marketing campaign by a... | Certainly! One of the most famous marketing campaigns involving pe... | Конечно! PR-кампания Coca-Cola “Share a Coke” — это одна из самых ... | ✔️ [1.000] | 
| 2 | Creating a Facebook ad campaign within a specific budget and targe... | give me a facebook ads campaign for the next 7 days for canva pro ... | jordan\|\|canva | Create a detailed 7-day social media advertising campaign plan for... | Certainly! Below is a detailed 7-day social media advertising camp... | Certainly! Here's a detailed 7-day Facebook Ads campaign plan for ... | ✔️ [1.000] | 
| 3 | The delivery was scheduled for today, but due to a payment delay t... | The delivery supposed to be today. Seems our payment delay to Lvio... | seagate\|\|lviv\|\|ms. chan | Please help rewrite the following message into a professional, pri... | Subject: Shipment Rescheduling Due to Payment Processing Delay Dea... | Subject: Shipment Rescheduling Notice Dear [Recipient], Please be ... | ✔️ [0.500] | 
| 4 | Dear Ms. Amanda, I hope this email finds you well. I wanted to bri... | write a email to Ms, Amanda ; tell her, we have a way to overcome ... | india\|\|amanda\|\|hermann(germany)\|\|china\|\|vims(france) | Please draft a professional email to a colleague named Ms. Amanda.... | Subject: Strategy for CE Certification and Manufacturing Options f... | Subject: Strategy to Address Certification and Import Challenges f... | ✔️ [0.700] | 

```
EvaluationResult(score=86.1, results=<list of 214 results>)
```
**Here, we see GEPA optimize the PAPILLON program from a score of 77% to 86% after proposing just 1 new candidate!**

# Citations

1. Source page: https://dspy.ai/current/tutorials/gepa_papillon
