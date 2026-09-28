---
type: Web Page
title: Entity Extraction - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/current/tutorials/entity_extraction
timestamp: '2026-09-28T13:20:17.892585+00:00'
---

# Tutorial: Entity Extraction

 This tutorial demonstrates how to perform **entity extraction** using the CoNLL-2003 dataset with DSPy. The focus is on extracting entities referring to people. We will:

- Extract and label entities from the CoNLL-2003 dataset that refer to people
- Define a DSPy program for extracting entities that refer to people
- Optimize and evaluate the program on a subset of the CoNLL-2003 dataset

By the end of this tutorial, you’ll understand how to structure tasks in DSPy using signatures and modules, evaluate your system’s performance, and improve its quality with optimizers.

Install the latest version of DSPy and follow along. If you’re looking instead for a conceptual overview of DSPy, this [recent lecture](https://www.youtube.com/live/JEMYuzrKLUw) is a good place to start.

## Recommended: Set up MLflow Tracing to understand what's happening under the hood.

### MLflow DSPy Integration

 [MLflow](https://mlflow.org/) is an LLMOps tool that natively integrates with DSPy and offer explainability and experiment tracking. In this tutorial, you can use MLflow to visualize prompts and optimization progress as traces to understand the DSPy’s behavior better. You can set up MLflow easily by following the four steps below.

1. Install MLflow

1.  Start MLflow UI in a separate terminal
2.  Connect the notebook to MLflow
3.  Enabling tracing.

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

## Load and Prepare the Dataset

 In this section, we prepare the CoNLL-2003 dataset, which is commonly used for entity extraction tasks. The dataset includes tokens annotated with entity labels such as persons, organizations, and locations.

We will: 1. Load the dataset using the Hugging Face `datasets` library. 2. Define a function to extract tokens referring to people. 3. Slice the dataset to create smaller subsets for training and testing.

DSPy expects examples in a structured format, so we’ll also transform the dataset into DSPy `Examples` for easy integration.

## Configure DSPy and create an Entity Extraction Program

 Here, we define a DSPy program for extracting entities referring to people from tokenized text.

Then, we configure DSPy to use a particular language model (`gpt-4o-mini`) for all invocations of the program.

**Key DSPy Concepts Introduced:** - **Signatures:** Define structured input/output schemas for your program. - **Modules:** Encapsulate program logic in reusable, composable units.

Specifically, we’ll: - Create a `PeopleExtraction` DSPy Signature to specify the input (`tokens`) and output (`extracted_people`) fields. - Define a `people_extractor` program that uses DSPy’s built-in `dspy.ChainOfThought` module to implement the `PeopleExtraction` signature. The program extracts entities referring to people from a list of input tokens using language model (LM) prompting. - Use the `dspy.LM` class and `dspy.configure()` method to configure the language model that DSPy will use when invoking the program.

Here, we tell DSPy to use OpenAI’s `gpt-4o-mini` model in our program. To authenticate, DSPy reads your `OPENAI_API_KEY`. You can easily swap this out for [other providers or local models](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb).

## Define Metric and Evaluation Functions

 In DSPy, evaluating a program’s performance is critical for iterative development. A good evaluation framework allows us to: - Measure the quality of our program’s outputs. - Compare outputs against ground-truth labels. - Identify areas for improvement.

**What We’ll Do:** - Define a custom metric (`extraction_correctness_metric`) to evaluate whether the extracted entities match the ground truth. - Create an evaluation function (`evaluate_correctness`) to apply this metric to a training or test dataset and compute the overall accuracy.

The evaluation function uses DSPy’s `Evaluate` utility to handle parallelism and visualization of results.

## Evaluate Initial Extractor

 Before optimizing our program, we need a baseline evaluation to understand its current performance. This helps us: - Establish a reference point for comparison after optimization. - Identify potential weaknesses in the initial implementation.

In this step, we’ll run our `people_extractor` program on the test set and measure its accuracy using the evaluation framework defined earlier.

```
Average Metric: 172.00 / 200 (86.0%): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:16<00:00, 11.94it/s]
2024/11/18 21:08:04 INFO dspy.evaluate.evaluate: Average Metric: 172 / 200 (86.0%)
```
|  | tokens | expected_extracted_people | rationale | extracted_people | extraction_correctness_metric | 
|---|---|---|---|---|---|
| 0 | [SOCCER, -, JAPAN, GET, LUCKY, WIN, ,, CHINA, IN, SURPRISE, DEFEAT... | [CHINA] | We extracted "JAPAN" and "CHINA" as they refer to specific countri... | [JAPAN, CHINA] |  | 
| 1 | [Nadim, Ladki] | [Nadim, Ladki] | We extracted the tokens "Nadim" and "Ladki" as they refer to speci... | [Nadim, Ladki] | ✔️ [True] | 
| 2 | [AL-AIN, ,, United, Arab, Emirates, 1996-12-06] | [] | There are no tokens referring to specific people in the provided l... | [] | ✔️ [True] | 
| 3 | [Japan, began, the, defence, of, their, Asian, Cup, title, with, a... | [] | We did not find any tokens referring to specific people in the pro... | [] | ✔️ [True] | 
| 4 | [But, China, saw, their, luck, desert, them, in, the, second, matc... | [] | The extracted tokens referring to specific people are "China" and ... | [China, Uzbekistan] |  | 
| ... | ... | ... | ... | ... | ... | 
| 195 | ['The', 'Wallabies', 'have', 'their', 'sights', 'set', 'on', 'a', ... | [David, Campese] | The extracted_people includes "David Campese" as it refers to a sp... | [David, Campese] | ✔️ [True] | 
| 196 | ['The', 'Wallabies', 'currently', 'have', 'no', 'plans', 'to', 'ma... | [] | The extracted_people includes "Wallabies" as it refers to a specif... | [] | ✔️ [True] | 
| 197 | ['Campese', 'will', 'be', 'up', 'against', 'a', 'familiar', 'foe',... | [Campese, Rob, Andrew] | The extracted tokens refer to specific people mentioned in the tex... | [Campese, Rob, Andrew] | ✔️ [True] | 
| 198 | ['"', 'Campo', 'has', 'a', 'massive', 'following', 'in', 'this', '... | [Campo, Andrew] | The extracted tokens referring to specific people include "Campo" ... | [Campo, Andrew] | ✔️ [True] | 
| 199 | ['On', 'tour', ',', 'Australia', 'have', 'won', 'all', 'four', 'te... | [] | We extracted the names of specific people from the tokenized text.... | [] | ✔️ [True] | 

200 rows × 5 columns

```
86.0
```
## Tracking Evaluation Results in MLflow Experiment

To track and visualize the evaluation results over time, you can record the results in MLflow Experiment.

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

## Optimize the Model

 DSPy includes powerful optimizers that can improve the quality of your system.

Here, we use DSPy’s `MIPROv2` optimizer to: - Automatically tune the program’s language model (LM) prompt by 1. using the LM to adjust the prompt’s instructions and 2. building few-shot examples from the training dataset that are augmented with reasoning generated from `dspy.ChainOfThought`. - Maximize correctness on the training set.

This optimization process is automated, saving time and effort while improving accuracy.

## Evaluate Optimized Program

 After optimization, we re-evaluate the program on the test set to measure improvements. Comparing the optimized and initial results allows us to: - Quantify the benefits of optimization. - Validate that the program generalizes well to unseen data.

In this case, we see that accuracy of the program on the test dataset has improved significantly.

```
Average Metric: 186.00 / 200 (93.0%): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:23<00:00,  8.58it/s]
2024/11/18 21:15:00 INFO dspy.evaluate.evaluate: Average Metric: 186 / 200 (93.0%)
```
|  | tokens | expected_extracted_people | rationale | extracted_people | extraction_correctness_metric | 
|---|---|---|---|---|---|
| 0 | [SOCCER, -, JAPAN, GET, LUCKY, WIN, ,, CHINA, IN, SURPRISE, DEFEAT... | [CHINA] | There are no specific people mentioned in the provided tokens. The... | [] |  | 
| 1 | [Nadim, Ladki] | [Nadim, Ladki] | The tokens "Nadim Ladki" refer to a specific individual. Both toke... | [Nadim, Ladki] | ✔️ [True] | 
| 2 | [AL-AIN, ,, United, Arab, Emirates, 1996-12-06] | [] | There are no tokens referring to specific people in the provided l... | [] | ✔️ [True] | 
| 3 | [Japan, began, the, defence, of, their, Asian, Cup, title, with, a... | [] | There are no specific people mentioned in the provided tokens. The... | [] | ✔️ [True] | 
| 4 | [But, China, saw, their, luck, desert, them, in, the, second, matc... | [] | There are no tokens referring to specific people in the provided l... | [] | ✔️ [True] | 
| ... | ... | ... | ... | ... | ... | 
| 195 | ['The', 'Wallabies', 'have', 'their', 'sights', 'set', 'on', 'a', ... | [David, Campese] | The extracted tokens refer to a specific person mentioned in the t... | [David, Campese] | ✔️ [True] | 
| 196 | ['The', 'Wallabies', 'currently', 'have', 'no', 'plans', 'to', 'ma... | [] | There are no specific individuals mentioned in the provided tokens... | [] | ✔️ [True] | 
| 197 | ['Campese', 'will', 'be', 'up', 'against', 'a', 'familiar', 'foe',... | [Campese, Rob, Andrew] | The tokens include the names "Campese" and "Rob Andrew," both of w... | [Campese, Rob, Andrew] | ✔️ [True] | 
| 198 | ['"', 'Campo', 'has', 'a', 'massive', 'following', 'in', 'this', '... | [Campo, Andrew] | The extracted tokens refer to specific people mentioned in the tex... | [Campo, Andrew] | ✔️ [True] | 
| 199 | ['On', 'tour', ',', 'Australia', 'have', 'won', 'all', 'four', 'te... | [] | There are no specific people mentioned in the provided tokens. The... | [] | ✔️ [True] | 

200 rows × 5 columns

```
93.0
```
## Inspect Optimized Program’s Prompt

 After optimizing the program, we can inspect the history of interactions to see how DSPy has augmented the program’s prompt with few-shot examples. This step demonstrates: - The structure of the prompt used by the program. - How few-shot examples are added to guide the model’s behavior.

Use `inspect_history(n=1)` to view the last interaction and analyze the generated prompt.

```
[34m[2024-11-18T21:15:00.584497][0m
[31mSystem message:[0m
Your input fields are:
1. `tokens` (list[str]): tokenized text
Your output fields are:
1. `rationale` (str): ${produce the extracted_people}. We ...
2. `extracted_people` (list[str]): all tokens referring to specific people extracted from the tokenized text
All interactions will be structured in the following way, with the appropriate values filled in.
[[ ## tokens ## ]]
{tokens}
[[ ## rationale ## ]]
{rationale}
[[ ## extracted_people ## ]]
{extracted_people}        # note: the value you produce must be pareseable according to the following JSON schema: {"type": "array", "items": {"type": "string"}}
[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        In a high-stakes situation where accurate identification of individuals is critical for regulatory compliance and public health communication, extract contiguous tokens referring to specific people from the provided list of string tokens. Ensure that you output each identified individual as separate tokens without combining multiple tokens into a single value. This task is essential for ensuring clarity and accountability in communications pertaining to EU regulations and health matters.
[31mUser message:[0m
[[ ## tokens ## ]]
["He", "said", "further", "scientific", "study", "was", "required", "and", "if", "it", "was", "found", "that", "action", "was", "needed", "it", "should", "be", "taken", "by", "the", "European", "Union", "."]
Respond with the corresponding output fields, starting with the field `[[ ## rationale ## ]]`, then `[[ ## extracted_people ## ]]` (must be formatted as a valid Python list[str]), and then ending with the marker for `[[ ## completed ## ]]`.
[31mAssistant message:[0m
[[ ## rationale ## ]]
There are no specific people mentioned in the provided tokens. The only reference is to the "European Union," which is an organization rather than a specific person. We therefore conclude that there are no extracted people.
[[ ## extracted_people ## ]]
[]
[[ ## completed ## ]]
[31mUser message:[0m
[[ ## tokens ## ]]
["BRUSSELS", "1996-08-22"]
Respond with the corresponding output fields, starting with the field `[[ ## rationale ## ]]`, then `[[ ## extracted_people ## ]]` (must be formatted as a valid Python list[str]), and then ending with the marker for `[[ ## completed ## ]]`.
[31mAssistant message:[0m
[[ ## rationale ## ]]
There are no tokens referring to specific people in the provided list. We only have a location and a date.
[[ ## extracted_people ## ]]
[]
[[ ## completed ## ]]
[31mUser message:[0m
[[ ## tokens ## ]]
["Fischler", "proposed", "EU-wide", "measures", "after", "reports", "from", "Britain", "and", "France", "that", "under", "laboratory", "conditions", "sheep", "could", "contract", "Bovine", "Spongiform", "Encephalopathy", "(", "BSE", ")", "--", "mad", "cow", "disease", "."]
Respond with the corresponding output fields, starting with the field `[[ ## rationale ## ]]`, then `[[ ## extracted_people ## ]]` (must be formatted as a valid Python list[str]), and then ending with the marker for `[[ ## completed ## ]]`.
[31mAssistant message:[0m
[[ ## rationale ## ]]
The extracted tokens refer to specific people mentioned in the text. In this case, "Fischler" is the only token that refers to a specific person. We extract this token as it is contiguous and represents an individual involved in the proposed measures.
[[ ## extracted_people ## ]]
["Fischler"]
[[ ## completed ## ]]
[31mUser message:[0m
[[ ## tokens ## ]]
["Germany", "'s", "representative", "to", "the", "European", "Union", "'s", "veterinary", "committee", "Werner", "Zwingmann", "said", "on", "Wednesday", "consumers", "should", "buy", "sheepmeat", "from", "countries", "other", "than", "Britain", "until", "the", "scientific", "advice", "was", "clearer", "."]
Respond with the corresponding output fields, starting with the field `[[ ## rationale ## ]]`, then `[[ ## extracted_people ## ]]` (must be formatted as a valid Python list[str]), and then ending with the marker for `[[ ## completed ## ]]`.
[31mAssistant message:[0m
[[ ## rationale ## ]]
We extracted the name "Werner Zwingmann" from the tokenized text as it refers to a specific person mentioned in the context of a statement regarding sheepmeat consumption.
[[ ## extracted_people ## ]]
["Werner", "Zwingmann"]
[[ ## completed ## ]]
[31mUser message:[0m
[[ ## tokens ## ]]
["LONDON", "1996-12-06"]
Respond with the corresponding output fields, starting with the field `[[ ## rationale ## ]]`, then `[[ ## extracted_people ## ]]` (must be formatted as a valid Python list[str]), and then ending with the marker for `[[ ## completed ## ]]`.
[31mResponse:[0m
[32m[[ ## rationale ## ]]
There are no tokens referring to specific people in the provided list. The tokens only include a location and a date.
[[ ## extracted_people ## ]]
[]
[[ ## completed ## ]][0m
```
## Keeping an eye on cost

 DSPy allows you to track the cost of your programs. The following code demonstrates how to obtain the cost of all LM calls made by the DSPy extractor program so far.

```
0.26362742999999983
```
## Saving and Loading Optimized Programs

 DSPy supports saving and loading programs, enabling you to reuse optimized systems without the need to re-optimize from scratch. This feature is especially useful for deploying your programs in production environments or sharing them with collaborators.

In this step, we’ll save the optimized program to a file and demonstrate how to load it back for future use.

```
['Marcello', 'Cuttitta']
```
## Saving programs in MLflow Experiment

Instead of saving the program to a local file, you can track it in MLflow for better reproducibility and collaboration.

1. **Dependency Management** : MLflow automatically save the frozen environment metadata along with the program to ensure reproducibility.
2. **Experiment Tracking** : With MLflow, you can track the program’s performance and cost along with the program itself.
3. **Collaboration** : You can share the program and results with your team members by sharing the MLflow experiment.

To save the program in MLflow, run the following code:

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

## Conclusion

 In this tutorial, we demonstrated how to: - Use DSPy to build a modular, interpretable system for entity extraction. - Evaluate and optimize the system using DSPy’s built-in tools.

By leveraging structured inputs and outputs, we ensured that the system was easy to understand and improve. The optimization process allowed us to quickly improve performance without manually crafting prompts or tweaking parameters.

**Next Steps:** - Experiment with extraction of other entity types (e.g., locations or organizations). - Explore DSPy’s other builtin modules like `ReAct` for more complex reasoning tasks. - Use the system in larger workflows, such as large scale document processing or summarization.

# Citations

1. Source page: https://dspy.ai/current/tutorials/entity_extraction
