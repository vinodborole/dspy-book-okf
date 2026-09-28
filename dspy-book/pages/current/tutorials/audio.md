---
type: Web Page
title: Audio - DSPy
description: The framework for programming—rather than prompting—language models.
resource: https://dspy.ai/current/tutorials/audio
timestamp: '2026-09-28T13:20:17.892585+00:00'
---

# Tutorial: Using Audio in DSPy Programs

 This tutorial walks through building pipelines for audio-based applications using DSPy.

### Install Dependencies

 Ensure you’re using the latest DSPy version:

To handle audio data, install the following dependencies:

### Load the Spoken-SQuAD Dataset

 We’ll use the Spoken-SQuAD dataset ([Official](https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD) & [HuggingFace version](https://huggingface.co/datasets/AudioLLMs/spoken_squad_test) for tutorial demonstration), which contains spoken audio passages used for question-answering:

### Preprocess Audio Data

 The audio clips in the dataset require some preprocessing into byte arrays with their corresponding sampling rates.

## DSPy program for spoken question answering

 Let’s define a simple DSPy program that uses audio inputs to answer questions directly. This is very similar to the [BasicQA](https://dspy.ai/cheatsheet/?h=basicqa#dspysignature) task, with the only difference being that the passage context is provided as an audio file for the model to listen to and answer the question:

Now let’s configure our LLM which can process input audio.

Note: Using `dspy.Audio` in signatures allows passing in audio directly to the model. 

### Define Evaluation Metric

 We’ll use the Exact Match metric (`dspy.evaluate.answer_exact_match`) to measure answer accuracy compared to the provided reference answers:

### Optimize with DSPy

 You can optimize this audio-based program as you would for any DSPy program using any DSPy optimizer.

Note: Audio tokens can be costly so it is recommended to configure optimizers like `dspy.BootstrapFewShotWithRandomSearch` or `dspy.MIPROv2` conservatively with 0-2 few shot examples and less candidates / trials than the optimizer default parameters.

With this small subset, MIPROv2 led to a ~10% improvement over baseline performance.

Now that we’ve seen how to use an audio-input-capable LLM in DSPy, let’s flip the setup.

In this next task, we’ll use a standard text-based LLM to generate prompts for a text-to-speech model and then evaluate the quality of the produced speech for some downstream task. This approach is generally more cost-effective than asking an LLM like `gpt-4o-mini-audio-preview-2024-12-17` to generate audio directly, while still enabling a pipeline that can be optimized for higher-quality speech output.

### Load the CREMA-D Dataset

 We’ll use the CREMA-D dataset ([Official](https://github.com/CheyneyComputerScience/CREMA-D) & [HuggingFace version](https://huggingface.co/datasets/myleslinder/crema-d) for tutorial demonstration), which includes audio clips of chosen participants speaking the same line with one of six target emotions: neutral, happy, sad, anger, fear, and disgust.

## DSPy pipeline for generating TTS instructions for speaking with a target emotion

 We’ll now build a pipeline that generates emotionally expressive speech by prompting a TTS model with both a line of text and an instruction on how to say it. The goal of this task will be to use DSPy to generate prompts that guide the TTS output to match the emotion and style of reference audio from the dataset.

First let’s set up the TTS generator to produce generate spoken audio with a specified emotion or style. We utilize `gpt-4o-mini-tts` as it supports prompting the model with raw input and speaking and produces an audio response as a `.wav` file processed with `dspy.Audio`. We also set up a cache for the TTS outputs.

Now let’s define the DSPy program for generating TTS instructions. For this program, we can use standard text-based LLMs again since we’re just generating instructions.

### Define Evaluation Metric

 Audio reference comparisons is generally a non-trivial task due to subjective variations of evaluating speech, especially with emotional expression. For the purposes of this tutorial, we use an embedding-based similarity metric for objective evaluation, leveraging Wav2Vec 2.0 to convert audio into embeddings and computing cosine similarity between the reference and generated audio. To evaluate audio quality more accurately, human feedback or perceptual metrics would be more suitable.

We can look at an example to see what instructions the DSPy program generated and the corresponding score:

```
[34m[2025-05-15T22:01:22.667596][0m
[31mSystem message:[0m
Your input fields are:
1. `raw_line` (str)
2. `target_style` (str)
Your output fields are:
1. `reasoning` (str)
2. `openai_instruction` (str)
All interactions will be structured in the following way, with the appropriate values filled in.
[[ ## raw_line ## ]]
{raw_line}
[[ ## target_style ## ]]
{target_style}
[[ ## reasoning ## ]]
{reasoning}
[[ ## openai_instruction ## ]]
{openai_instruction}
[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Generate an OpenAI TTS instruction that makes the TTS model speak the given line with the target emotion or style.
[31mUser message:[0m
[[ ## raw_line ## ]]
It's eleven o'clock
[[ ## target_style ## ]]
disgust
Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## openai_instruction ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
[31mResponse:[0m
[32m[[ ## reasoning ## ]]
To generate the OpenAI TTS instruction, we need to specify the target emotion or style, which in this case is 'disgust'. We will use the OpenAI TTS instruction format, which includes the text to be spoken and the desired emotion or style.
[[ ## openai_instruction ## ]]
"Speak the following line with a tone of disgust: It's eleven o'clock"
[[ ## completed ## ]][0m
```
TTS Instruction:

The instruction specifies the target emotion, but is not too informative beyond that. We can also see that the audio score for this sample is not too high. Let’s see if we can do better by optimizing this pipeline.

### Optimize with DSPy

 We can leverage `dspy.MIPROv2` to refine the downstream task objective and produce higher quality TTS instructions, leading to more accurate and expressive audio generations:

Let’s take a look at how the optimized program performs:

```
[34m[2025-05-15T22:09:40.088592][0m
[31mSystem message:[0m
Your input fields are:
1. `raw_line` (str)
2. `target_style` (str)
Your output fields are:
1. `reasoning` (str)
2. `openai_instruction` (str)
All interactions will be structured in the following way, with the appropriate values filled in.
[[ ## raw_line ## ]]
{raw_line}
[[ ## target_style ## ]]
{target_style}
[[ ## reasoning ## ]]
{reasoning}
[[ ## openai_instruction ## ]]
{openai_instruction}
[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Generate an OpenAI TTS instruction that makes the TTS model speak the given line with the target emotion or style, as if the speaker is a [insert persona relevant to the task, e.g. "irate customer", "angry boss", etc.]. The instruction should specify the tone, pitch, and other characteristics of the speaker's voice to convey the target emotion.
[31mUser message:[0m
[[ ## raw_line ## ]]
It's eleven o'clock
[[ ## target_style ## ]]
disgust
Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## openai_instruction ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
[31mResponse:[0m
[32m[[ ## reasoning ## ]]
To convey disgust, the speaker's voice should be characterized by a high-pitched tone, a slightly nasal quality, and a sense of revulsion. The speaker's words should be delivered with a sense of distaste and aversion, as if the speaker is trying to convey their strong negative emotions.
[[ ## openai_instruction ## ]]
Generate a text-to-speech synthesis of the input text "It's eleven o'clock" with the following characteristics: 
- Tone: Disgusted
- Pitch: High-pitched, slightly nasal
- Emphasis: Emphasize the words to convey a sense of distaste and aversion
- Volume: Moderate to loud, with a sense of rising inflection at the end to convey the speaker's strong negative emotions
- Speaker: A person who is visibly and audibly disgusted, such as a character who has just been served a spoiled meal.
[[ ## completed ## ]][0m
```
MIPROv2 Optimized Program Instruction:

TTS Instruction:

MIPROv2’s instruction tuning added more flavor to the overall task objective, giving more criteria to how the TTS instruction should be defined, and in turn, the generated instruction is much more specific to the various factors of speech prosody and produces a higher similarity score.

# Citations

1. Source page: https://dspy.ai/current/tutorials/audio
