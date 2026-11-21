# Generating Synthetic Data: Step-by-Step Guide

This guide walks you through making your own synthetic dataset, from setup to a checked file you can commit. Every member follows the same steps, so the results can be compared fairly.

**What you produce:** one file, `data/raw/yourname.jsonl`, with about 1,000 examples (the team can change the number in `configs/generation.yaml`).

**Two ways to talk to the model.** You can type prompts into a chat window (a browser or an app), or you can send them from code through an API. Both give you the same kind of data. This guide shows both, so pick the one that fits you and your team.

> The example prompts and example outputs in this guide are illustrations written to show the idea. Adjust the wording for your topic, but keep the structure and the team's shared settings.

## Contents

1. [How your data gets used](#1-how-your-data-gets-used)
2. [Before you start](#2-before-you-start)
3. [Choose how you will talk to the model](#3-choose-how-you-will-talk-to-the-model)
4. [Pick a generation method](#4-pick-a-generation-method)
5. [Method A: Self-Instruct](#5-method-a-self-instruct)
6. [Method B: Evol-Instruct](#6-method-b-evol-instruct)
7. [Method C: Magpie](#7-method-c-magpie)
8. [Light cleanup (and what not to clean)](#8-light-cleanup-and-what-not-to-clean)
9. [Save and check your file](#9-save-and-check-your-file)
10. [Tips, cost and troubleshooting](#10-tips-cost-and-troubleshooting)
11. [Checklist before you commit](#11-checklist-before-you-commit)

---

## 1. How your data gets used

The pipeline is: **generate, filter, fine-tune, evaluate, compare.**

1. You generate synthetic examples (this guide).
2. You apply your filtering method to them, which keeps some examples and drops others.
3. The kept examples are used to **fine-tune** the team's shared model. This is the training step.
4. The model is evaluated on a shared test set.
5. The team compares results across filtering methods.

Each example is an instruction (what the model is asked to do) and an output (what a good answer looks like). During fine-tuning, the model reads the instruction and learns to produce the output.

The three generation methods differ in how they *make* examples, and the two routes differ in how you *talk* to the model, but everything ends in the same place: **instruction/output pairs saved in the same JSONL format.** That shared format is what lets one training script and one set of filters work on everyone's data. The content differs between members, and that is expected. What must stay identical is the structure and the settings.

## 2. Before you start

**Agree on these as a team (they go in `configs/generation.yaml`):**
- the generator: which model everyone uses (see section 3)
- generation method (see section 4)
- temperature (randomness) and random seed, if your route lets you set them
- number of examples per person
- topic or domain, if you are restricting it

**Set up your machine:**
```bash
git clone <repo url>
cd synthetic-data-quality-screening
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Create your branch:**
```bash
git checkout -b yourname/data
```

## 3. Choose how you will talk to the model

| | **Route A: chat window** (browser or app) | **Route B: API** (from code) |
|---|---|---|
| Setup | None. Open the chatbot. | Get access and a key (below) |
| Effort | Copy and paste, about 50 messages for 1,000 examples | Run a script |
| Cost | Whatever your chat plan costs, with usage limits | Pay per use. Set a spending limit |
| Temperature and seed | You can't set them | You can set them |
| Reproducible | Hard | Easier |
| Works with | Self-Instruct, Evol-Instruct | Self-Instruct, Evol-Instruct, Magpie |

**Whichever you choose, everyone should use the same model**, or differences in the data will come from the generator instead of from your filters. If some members use the chat route, the team should still agree on one chatbot and one prompt. Also check the provider's terms of service about using its outputs to train other models. Open-weight models, such as the team's hosted Qwen, are easier on this point, but check the model's license.

### Route A: chat window

**Setup:** none. Open the chatbot the team agreed on.

**Workflow:**
1. Start a **new chat** and paste the prompt from the method section (sections 5 to 7). Ask for JSON lines, as those sections show.
2. Copy the reply into a plain text file, for example `pasted.txt`. Paste every reply into the same file.
3. Start a new chat every few messages. Long conversations start repeating themselves.
4. When you have enough, turn the text into your data file. This adds an `id` and the `meta` fields for you:
```bash
python -m src.prepare_chat_data --input pasted.txt --author yourname \
    --method self_instruct --generator "ChatGPT, model name shown in the app, 2026-11-21" \
    --output data/raw/yourname.jsonl
```
It skips chatter and code fences, keeps every valid example, and lists any lines it couldn't use. To add more later, run it again with `--append`. Numbering continues where it left off.

Write the chatbot's name, the model name shown in the app and the date in `--generator`, because you can't set a temperature or seed.

### Route B: API

You need an endpoint, a key, and a few lines of code. There are two common setups.

**Option 1: the team's hosted model ** Ask the team lead for the endpoint URL and your personal token. Put them in a file called `.env` in the repo root:
```
ENDPOINT_URL=https://...
ENDPOINT_TOKEN=...
```

**Option 2: OpenAI directly.**
The team leads will give you an API Key but PLEASE DO NOT PUSH IT TO THE GITHUB!!!!!
```
OPENAI_API_KEY=...
```

**Never commit `.env` or a key.** Make sure `.env` is in `.gitignore`, and run `git status` before committing.

**Code to send a prompt.** Install `pip install openai python-dotenv`, then:
```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Option 1: the team's endpoint, if it is OpenAI-compatible (servers like vLLM usually are).
# Check how your endpoint expects the token to be sent, and adjust.
client = OpenAI(base_url=os.environ["ENDPOINT_URL"], api_key=os.environ["ENDPOINT_TOKEN"])

# Option 2: OpenAI directly. Use this line instead of the one above.
# client = OpenAI()   # reads OPENAI_API_KEY

def ask(prompt, temperature=0.8):
    reply = client.chat.completions.create(
        model="MODEL_NAME_FROM_CONFIG",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return reply.choices[0].message.content
```
In the method sections, "Route B" means: call `ask(prompt)` in a loop, parse each reply, and save the results. `src/generate.py` is meant to hold that loop. It is still a skeleton until the team picks the model.

## 4. Pick a generation method

| Method | Start from | Best for | Watch out |
|--------|-----------|----------|-----------|
| **Self-Instruct** | A few hand-written seed tasks | Simple, standard, Alpaca-style data | Needs good seed tasks |
| **Evol-Instruct** | Existing instructions | Making instructions harder and more varied | Needs a starting set; more steps |
| **Magpie** | Nothing (just the chat template) | Cheap, large amounts of data | Needs an open instruct model and a raw text-completion endpoint, so **API only** |

If you aren't sure, use Self-Instruct. The team may also assign methods so every method gets covered. Whichever you use, record it in each example's `meta` field.

---

## 5. Method A: Self-Instruct

**Idea:** show the model a few example tasks and ask it for new ones. Then have it write an input and output for each new task. The prompt is a fixed template. The variety comes from which seed examples you put in it each round.

### Step 1: Write your seed tasks

Make a small list of hand-written seed instructions, about 20 to 50 to start. They should be varied: different topics, different kinds of tasks.

```
Explain the difference between a list and a tuple in Python.
Write a polite email asking a professor for an extension.
Summarize the main causes of the French Revolution in three sentences.
Give three tips for staying focused while studying.
```

### Route B (API): the full two-step version

**Step 2: generate new instructions.** Each round, pick 3 seeds at random and fill them into the template from `prompts/self_instruct.txt`.

*Example prompt (filled in):*
```
Come up with 8 new, diverse task instructions.
Vary the topics. Each task must be clear and doable by an LLM.
Do not copy the examples below.

Task 1: Explain the difference between a list and a tuple in Python.
Task 2: Write a polite email asking a professor for an extension.
Task 3: Summarize the main causes of the French Revolution in three sentences.
Task 4:
```

*Example model reply (illustrative):*
```
Convert 72 degrees Fahrenheit to Celsius and show your steps.
Suggest three questions to ask at the end of a job interview.
Describe how photosynthesis works to a ten-year-old.
...
```
Split the reply into one instruction per line and strip any numbering the model added. In later rounds, you can also use some of the model's own accepted instructions as examples, so the pool keeps growing and changing.

**Step 3: write an input and output for each instruction.**

*Example prompt (filled in):*
```
Instruction: Convert 72 degrees Fahrenheit to Celsius and show your steps.
Write an input if needed, then a high-quality output.
Input:
Output:
```

*Example model reply (illustrative):*
```
Input: (none)
Output: Use C = (F - 32) x 5/9. First, 72 - 32 = 40. Then 40 x 5/9 is about 22.2. So 72 degrees F is about 22.2 degrees C.
```

**Step 4: save it as one JSONL line.**
```json
{"id": "yourname-0001", "instruction": "Convert 72 degrees Fahrenheit to Celsius and show your steps.", "input": "", "output": "Use C = (F - 32) x 5/9. First, 72 - 32 = 40. Then 40 x 5/9 is about 22.2. So 72 degrees F is about 22.2 degrees C.", "meta": {"author": "yourname", "method": "self_instruct", "generator": "MODEL_NAME", "temperature": 0.8, "seed": 42}}
```
Repeat steps 2 to 4 until you have enough examples.

### Route A (chat window): the one-prompt version

A chat window is awkward for two separate steps, so you can ask for finished examples in one go. Paste this, with 3 seed tasks filled in:

```
Generate 20 diverse training examples for fine-tuning a language model.
Output exactly 20 lines of JSON (JSON Lines), one object per line, using the keys
"instruction", "input" and "output". If an example needs no input, use "".
No numbering, no code fences and no other text.
Vary the topics and task types. Do not copy these example tasks:
Task 1: Explain the difference between a list and a tuple in Python.
Task 2: Write a polite email asking a professor for an extension.
Task 3: Summarize the main causes of the French Revolution in three sentences.
```

*Example reply (illustrative, shortened):*
```
{"instruction": "Convert 72 degrees Fahrenheit to Celsius and show your steps.", "input": "", "output": "Use C = (F - 32) x 5/9. First, 72 - 32 = 40. Then 40 x 5/9 is about 22.2. So 72 degrees F is about 22.2 degrees C."}
{"instruction": "Suggest three questions to ask at the end of a job interview.", "input": "", "output": "1. What does success look like in this role? ..."}
...
```
Copy each reply into `pasted.txt`, change the 3 seed tasks each time, and repeat until you have enough. Then run `src.prepare_chat_data` (section 3). **This is a simplified version of Self-Instruct** (one step instead of two), so if some members use the API route and some use chat, the team should agree on which version everyone uses.

---

## 6. Method B: Evol-Instruct

**Idea:** take instructions you already have and have the model rewrite them to be harder (depth) or to create something new but related (breadth). Repeat for a few rounds.

### Step 1: Get starting instructions

Use your seed tasks from Self-Instruct, or any set of simple instructions the team agrees on.

### Step 2: Evolve them

*Depth prompt (from `prompts/evol_instruct.txt`, filled in):*
```
Act as a Prompt Rewriter.
Rewrite the prompt below into a more complex version.
Keep it reasonable and answerable by a person.
Do one of: add a constraint, require deeper reasoning,
or replace a general concept with a specific one.
#Given Prompt#: Explain what overfitting is.
#Rewritten Prompt#:
```

*Example replies across rounds (illustrative):*
```
Round 0: Explain what overfitting is.
Round 1: Explain what overfitting is and give one example using polynomial regression.
Round 2: Explain what overfitting is, give one example using polynomial regression,
         and describe two ways to detect it by comparing training and validation loss.
```

*Breadth prompt:*
```
Write a brand-new prompt in the same domain as the one below.
Make it rarer and more unusual, but still answerable.
#Given Prompt#: Explain what overfitting is.
#New Prompt#:
```
Example reply (illustrative): `Describe how data leakage between training and test sets can make a model look better than it is.`

Run 1 to 3 rounds. The team should agree on the number of rounds so everyone matches. Throw away any rewritten prompt that came back broken: empty, a refusal, or the model just talking about the task instead of rewriting it.

### Step 3: Get an answer for each final instruction

Send each final instruction to the model as a normal prompt to get the output.

### Route B (API)

Loop over your instructions and call `ask()` once per instruction for each round, using the prompts above. Then call `ask()` once per final instruction to get its answer, and save each pair as one JSONL line:
```json
{"id": "yourname-0002", "instruction": "Explain what overfitting is, give one example using polynomial regression, and describe two ways to detect it by comparing training and validation loss.", "input": "", "output": "Overfitting happens when a model memorizes noise in its training data...", "meta": {"author": "yourname", "method": "evol_instruct", "generator": "MODEL_NAME", "temperature": 0.8, "seed": 42, "rounds": 2}}
```

### Route A (chat window)

Work on a batch of about 10 instructions at a time instead of one at a time.

*Round prompt:*
```
Act as a Prompt Rewriter. Rewrite each of the 10 prompts below into a more complex version.
Keep each one reasonable and answerable by a person.
For each, do one of: add a constraint, require deeper reasoning,
or replace a general concept with a specific one.
Output exactly 10 lines, one rewritten prompt per line, in the same order,
with no numbering and no other text.

Prompts:
1. Explain what overfitting is.
2. Explain what a loss function is.
...
```
For the next round, paste the replies back in as the new "Prompts". After your last round, ask for the answers:

*Answer prompt:*
```
For each of the following 10 instructions, write a high-quality answer.
Output exactly 10 lines of JSON (JSON Lines), one object per line, using the keys
"instruction", "input" and "output". Use "" for the input.
No numbering, no code fences and no other text.

Instructions:
...
```
Copy the replies into `pasted.txt` and run `src.prepare_chat_data` with `--method evol_instruct` (section 3).

---

## 7. Method C: Magpie

**Idea:** there is no instruction prompt. You send an aligned open model only the *start* of its chat template, as if a user was about to speak. The model writes the user's message itself, so no seed tasks are needed.

> **This works only through the API, and only if the endpoint offers raw text completion.** A chat window or a normal chat API adds the template for you, which defeats the trick. Ask the team lead whether your endpoint supports a text-completion route. Members on the chat route can't use Magpie.

### Step 1: Find your model's chat template

Every model has its own template tokens. As far as I know, Qwen models use ChatML-style tokens, so the start of a user turn looks like this, but confirm in the model's own tokenizer or chat template before you rely on it:
```
<|im_start|>user
```
For a Llama-3-style model it would be:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
```

### Step 2: Send only that start and let the model continue

*Example completion (illustrative):*
```
How can I improve my time management when studying for multiple exams?
```
Stop the generation at the model's end-of-turn token so you capture just one message. Sample many times with a higher temperature to get variety.

### Step 3: Send that message back as a normal prompt

Use the regular chat interface of the API to get the answer.

### Step 4: Save as JSONL

```json
{"id": "yourname-0003", "instruction": "How can I improve my time management when studying for multiple exams?", "input": "", "output": "Start by listing every exam and its date...", "meta": {"author": "yourname", "method": "magpie", "generator": "MODEL_NAME", "temperature": 1.0, "seed": 42}}
```

---

## 8. Light cleanup (and what not to clean)

The project is about which *filtering method* removes bad examples best. If you fix every quality problem yourself, there is nothing left for the filters to find. So only remove **mechanical problems**, and leave quality problems in.

**Remove these (they are formatting problems, not quality judgments):**
- exact duplicates
- examples with an empty instruction or empty output
- replies that couldn't be parsed or are cut off mid-sentence
- obvious refusals, such as "As an AI language model, I cannot..."

**Leave these in (the filters should handle them):**
- answers that are wrong, shallow or low quality
- repetitive or very similar examples that aren't exact duplicates, unless the team agrees on a near-duplicate rule
- awkward wording, bias or off-topic content

The team should agree on exactly which of these everyone applies, so everyone's data is treated the same way. The checker in the next section finds the mechanical problems for you.

## 9. Save and check your file

**Route A (chat window):** `src.prepare_chat_data` already numbered your examples and filled in the `meta` fields (section 3).

**Route B (API):** number examples as `yourname-0001`, `yourname-0002`, and so on, with unique ids. Save one JSON object per line to `data/raw/yourname.jsonl` and put the generation details in `meta` (author, method, generator model, temperature, seed).

**Then check the file, for both routes:**

1. Check the format:
```bash
python -m src.validate data/raw/yourname.jsonl
```
It prints `OK` with a count, or lists the problems it found. Fix them and run it again.

2. Run the full standard check. It applies the cleanup rules from section 8 and prints statistics:
```bash
python -m src.check_data data/raw/yourname.jsonl
```
Every line in the report starts with a level:
- `FAIL`: fix it before you commit (duplicates, refusals, leftover template placeholders, format problems).
- `WARN`: look at it and fix it if it is a real problem (very short or very long examples, outputs that look cut off, near-duplicates, missing `meta` fields).
- `NOTE` and `PASS`: information only.

The report ends with `READY` or `NOT READY`. The checker does **not** judge whether your answers are good. Quality problems are left in on purpose, because the filtering methods are what find them. Use `--expected 1000` to compare against a specific example count, and `--strict` to make warnings count as failures.

## 10. Tips, cost and troubleshooting

- **Test small first.** Generate 20 examples, look at them, and check the cost before running the full amount.
- **A shared endpoint may queue.** Run generation as a script and avoid starting big jobs at the same time as teammates. The first request after a quiet period can be slow.
- **Track your usage.** Note roughly how many requests you made. The team's GPU budget is limited, and chat plans have limits too.
- **Read a sample by hand** before saving. You are looking for broken formatting, not for quality problems.
- **Don't edit the generated text to improve it.** If you hand-fix examples, your data isn't comparable to everyone else's.
- **Never put real people's private information** or copyrighted text into prompts or data.

| Problem | Likely fix |
|---------|-----------|
| The reply isn't in the format you expected | Make the prompt stricter about the format, or fix your parsing code |
| The chatbot writes text around the JSON | That's fine on Route A. `prepare_chat_data` ignores it. Check the "Skipped" list it prints |
| `prepare_chat_data` skips many lines | The chatbot probably broke the JSON, for example with unescaped quotes. Ask it to fix those lines, or re-ask for fewer examples per message |
| Lots of near-identical instructions | Raise the temperature (API), start a new chat and change the seed tasks (chat), and sample different seeds each round |
| Replies refuse or lecture | Reword the prompt, and drop refusals as part of cleanup |
| Magpie returns nothing useful | Check you're using the exact template tokens and a completion endpoint |
| Validation errors | Read the line numbers it prints, and fix those lines |

## 11. Checklist before you commit

- [ ] I used the team's shared settings (model, method, temperature, seed, count) or recorded what I used
- [ ] My file is at `data/raw/yourname.jsonl`
- [ ] `python -m src.validate` reports no problems
- [ ] `python -m src.check_data` has no `FAIL` lines, and I've read every `WARN`
- [ ] Each example has a unique `id` and its details in `meta`
- [ ] I only applied the agreed mechanical cleanup
- [ ] `.env` and any tokens or keys are **not** staged (check `git status`)
- [ ] I'm on my own branch and will open a pull request
