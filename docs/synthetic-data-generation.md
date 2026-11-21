# Generating Synthetic Data: Step-by-Step Guide

This guide walks you through making your own synthetic dataset, from setup to a checked file you can commit. Every member follows the same steps, so the results can be compared fairly.

**What you produce:** one file, `data/raw/yourname.jsonl`, with about 1,000 examples (the number is set in `configs/generation.yaml`).

**Two ways to generate.** The main route runs a notebook that sends your prompt to the OpenAI API using a key the team lead gives you. If you have no key yet, a fallback route lets you use a chat window (browser or app) and paste the results into a helper script. Both produce the same JSONL format.

> The example prompts and outputs in this guide are illustrations. Adjust the wording for your topic, but keep the structure and the team's shared settings.

## Contents

1. [How your data gets used](#1-how-your-data-gets-used)
2. [Before you start](#2-before-you-start)
3. [Choose a route](#3-choose-a-route)
4. [Route A: the notebook (API)](#4-route-a-the-notebook-api)
5. [Write your data instructions](#5-write-your-data-instructions)
6. [Route B: chat window (fallback)](#6-route-b-chat-window-fallback)
7. [Other generation methods](#7-other-generation-methods)
8. [Light cleanup (and what not to clean)](#8-light-cleanup-and-what-not-to-clean)
9. [Save and check your file](#9-save-and-check-your-file)
10. [Submit your data (where it goes)](#10-submit-your-data-where-it-goes)
11. [Tips, cost and troubleshooting](#11-tips-cost-and-troubleshooting)
12. [Checklist before you commit](#12-checklist-before-you-commit)
13. [For the team lead: handing out keys](#13-for-the-team-lead-handing-out-keys)

---

## 1. How your data gets used

The pipeline is: **generate, filter, fine-tune, evaluate, compare.**

1. You generate synthetic examples (this guide).
2. You apply your filtering method, which keeps some examples and drops others.
3. The kept examples are used to **fine-tune** the team's shared model. This is the training step.
4. The model is evaluated on a shared test set.
5. The team compares results across filtering methods.

Each example is an instruction (what the model is asked to do) and an output (what a good answer looks like). During fine-tuning, the model reads the instruction and learns to produce the output.

Everyone's data ends up in the same place: **instruction/output pairs in the same JSONL format.** That shared format is what lets one training script and one set of filters work on everyone's data. The content differs between members, and that is expected. What must stay identical is the structure and the settings.

```json
{"id": "sam-0001", "instruction": "Summarize the text.", "input": "Cats sleep a lot.", "output": "Cats sleep much.", "meta": {"author": "sam", "method": "self_instruct", "generator": "gpt-6-luna"}}
```

`id`, `instruction` and `output` are required. `input` (extra context, or an empty string) and `meta` (how the example was made) are optional but expected by the checker.

## 2. Before you start

**Get these from the team lead:**
- your **API key** (do not create your own)
- the **generator model** name everyone uses
- the **generation method** and the **number of examples** (the defaults are in `configs/generation.yaml`)

**Set up your machine:**
```bash
git clone <repo url>
cd synthetic-data-quality-screening
python -m venv .venv && source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Create your branch:**
```bash
git checkout -b yourname/data
```

**Look after your key.** The key is billed to the team.
- Never paste it into a notebook cell, a chat window, a commit, a screenshot or a shared document.
- The notebook asks for it in a hidden prompt, so it is never saved in the file.
- If you think it has leaked, tell the team lead straight away so it can be replaced.

## 3. Choose a route

| | **Route A: notebook (API)** | **Route B: chat window** |
|---|---|---|
| Use it when | You have the team's API key | You have no key yet |
| Setup | VS Code with the Python and Jupyter extensions | A browser or app |
| Effort | Run a few cells | Copy and paste about 50 times for 1,000 examples |
| Cost | Billed to the team's key | Your own chat plan and its limits |
| Settings | Same model for everyone, recorded automatically | You choose and record the chatbot yourself |
| Reproducible | Mostly | Hard |

**Fairness:** the point of the project is to compare filtering methods, so differences in the data should come from your filters and not from the generator. Everyone should use the same model and the same prompt template. If you use Route B, use the chatbot the team agreed on. Also check the provider's terms of service about using its outputs to train other models.

---

## 4. Route A: the notebook (API)

The notebook is `notebooks/batch_data_generator.ipynb`. It uses OpenAI's **Batch API**: you submit one job with all your requests, it runs on OpenAI's servers (up to 24 hours), and you collect the results afterwards. You do not need a GPU and you can close VS Code while it runs.

### Setup

1. Open the repository folder in VS Code and install the **Python** and **Jupyter** extensions.
2. Open `notebooks/batch_data_generator.ipynb` and select the Python kernel from the `.venv` you created (VS Code may offer to install `ipykernel`).
3. Run the cells **one at a time**. Do not use "Run All", because some cells start paid jobs.

*Optional:* if you would rather not type the key into the prompt, set it as an environment variable in the terminal before opening VS Code from that same terminal. On macOS and Linux: `export OPENAI_API_KEY="..."` then `code .`. On Windows PowerShell: `$env:OPENAI_API_KEY="..."` then `code .`. The notebook uses the variable if it is set.

### The steps

| Section | What you do | Costs money? |
|---|---|---|
| 1. Install | Run the cell once | No |
| 2. Key and settings | Edit `AUTHOR`, `METHOD` and `MODEL`, run the cell, and paste your key into the hidden prompt | No |
| 3. Instructions | Replace the placeholder with your data instructions (see [section 5](#5-write-your-data-instructions)) | No |
| 4. Test 3 examples | Run the cell, open `data/work/test_3_examples.jsonl` and read it. Edit your instructions and repeat until you are happy | Yes, one small request |
| 5A. Batch test | Set `SUBMIT_TEST_BATCH = True` and run it **once** | Yes, one small batch |
| 5B. Collect the test | Paste the batch ID and run it, repeating later if it is not finished | No |
| 6. Full batch | Check `TOTAL_EXAMPLES`, set `SUBMIT_BATCH = True` and run it **once** | **Yes, the main cost** |
| 7. Collect the dataset | Paste the batch ID from Section 6 and run it. Repeat later if it is not finished | No |

**Settings in Section 2:**
- `AUTHOR`: your name in lowercase with no spaces. It becomes your ID prefix, for example `sam-0001`. The cell refuses to run until you change it.
- `METHOD`: `self_instruct`, `evol_instruct` or `magpie`, as the team agreed. It is recorded in each example's `meta`.
- `MODEL`: the generator model from the team lead. It is recorded in each example's `meta`.

**Rules that protect the team's budget:**
- **Submit each batch once.** Running a submission cell again creates another paid job. To check progress, rerun only the retrieval cell (Section 5B or 7).
- Keep `TOTAL_EXAMPLES` equal to `num_examples` in `configs/generation.yaml`.
- Always do Sections 4 and 5A before Section 6, so a mistake costs a few examples and not the full run.
- Retrieve your results soon after the batch finishes. OpenAI's result files are not kept forever.

**Where the files go:**
- `data/raw/yourname.jsonl`: your finished dataset (only written when every request succeeded)
- `data/work/`: test runs, batch IDs and raw request files. Git ignores this folder.
- If some requests failed, the notebook writes `yourname_PARTIAL.jsonl` and `batch_errors.jsonl` into `data/work/` instead, so an incomplete file never reaches `data/raw/`. Ask the team lead before re-running anything, since a rerun costs money.

If you restart VS Code after submitting, run Sections 1 and 2, then Section 6 with `SUBMIT_BATCH = False` (this restores the counts without submitting), then Section 7 with your saved batch ID from `data/work/batch_id.txt`.

Before you commit the notebook, clear its outputs (in VS Code: "Clear All Outputs") so no stray paths or batch IDs end up in Git.

After Section 7, go to [section 9](#9-save-and-check-your-file) to check your file, then [section 10](#10-submit-your-data-where-it-goes) to submit it.

---

## 5. Write your data instructions

Section 3 of the notebook has a string called `DATA_INSTRUCTIONS`. The notebook sends it, plus a request for N examples, in every request. Replace only the line `[REPLACE THIS WITH YOUR OWN SYNTHETIC DATA INSTRUCTIONS]`. **Leave the block below it** (the required `instruction`, `input` and `output` fields) unchanged, since the rest of the notebook depends on it.

### Self-Instruct template

Self-Instruct works by showing the model a few example tasks and asking it for new, different ones. Here the examples and the answers are requested together, in one step. Start from this and adjust the topics:

```
Generate diverse, high-quality instruction-following examples for fine-tuning a language model.

Cover many topics (science, history, everyday life, programming, writing, math,
health, business) and many task types (questions, rewriting, summarizing,
classification, short reasoning problems, drafting emails and so on).
Vary the length, difficulty, wording and style.
Do not repeat or closely paraphrase any other example in the same batch.
Do not copy the seed tasks below. They only show the style.

Seed tasks:
1. Explain the difference between a list and a tuple in Python.
2. Write a polite email asking a professor for an extension.
3. Summarize the main causes of the French Revolution in three sentences.
4. Give three tips for staying focused while studying.
```

Write about 20 to 50 varied seed tasks of your own if you can. The seed tasks, topics and wording are where your dataset's variety comes from.

*Example of one generated example (illustrative):*
```json
{"instruction": "Convert 72 degrees Fahrenheit to Celsius and show your steps.", "input": "", "output": "Use C = (F - 32) x 5/9. First, 72 - 32 = 40. Then 40 x 5/9 is about 22.2. So 72 degrees F is about 22.2 degrees C."}
```

### Keep your data varied

The notebook sends the **same** instructions in every request, so many requests can return similar examples. To reduce repeats:
- give a long list of topics and task types, and ask for variety, as above
- use more seed tasks, with different styles
- keep `EXAMPLES_PER_REQUEST` modest, because a model asked for very many examples at once tends to repeat itself

The checker reports exact duplicates and near-duplicates. Removing exact duplicates is allowed as cleanup (see [section 8](#8-light-cleanup-and-what-not-to-clean)).

### A different kind of data

The team's data is general instruction/output pairs. If your filtering method needs a particular shape of data, agree it with the team first so all the datasets stay comparable.

---

## 6. Route B: chat window (fallback)

Use this if you do not have a key yet. You talk to a chatbot in a browser or app, then a helper script turns what you pasted into the standard file.

**1. Start a new chat** with the chatbot the team agreed on, and paste this prompt, with your own seed tasks filled in:
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
{"instruction": "Convert 72 degrees Fahrenheit to Celsius and show your steps.", "input": "", "output": "Use C = (F - 32) x 5/9. ..."}
{"instruction": "Suggest three questions to ask at the end of a job interview.", "input": "", "output": "1. What does success look like in this role? ..."}
```

**2. Paste each reply into a plain text file**, for example `pasted.txt`. Change the seed tasks each time, and start a new chat every few messages, because long chats start repeating themselves. About 50 messages gives 1,000 examples.

**3. Turn the text into your data file:**
```bash
python -m src.prepare_chat_data --input pasted.txt --author yourname \
    --method self_instruct --generator "ChatGPT, model name shown in the app, 2026-11-21" \
    --output data/raw/yourname.jsonl
```
It ignores chatter and code fences, keeps every valid example, adds ids and `meta`, and lists any lines it could not use. To add more later, run it again with `--append`. Numbering continues where it left off. It will not overwrite an existing file unless you use `--append`.

Write the chatbot's name, the model name shown in the app and the date in `--generator`, since you cannot set a temperature or a seed in a chat window.

**Note:** this is a one-step version of Self-Instruct. The team should agree on which version everyone uses, so the datasets stay comparable.

---

## 7. Other generation methods

The team has discussed three generation methods. The notebook sends one prompt per request, which fits the one-step Self-Instruct approach above. The other two do not fit the notebook as it is.

**Evol-Instruct** starts from simple instructions and has the model rewrite them to be harder (depth) or to branch into new related tasks (breadth), over one to three rounds. This takes several passes, and the notebook does not automate them. If the team wants it, the rewriting prompts are in `prompts/evol_instruct.txt`, and you can work in batches of about 10 in a chat window:

```
Act as a Prompt Rewriter. Rewrite each of the 10 prompts below into a more complex version.
Keep each one reasonable and answerable by a person.
For each, do one of: add a constraint, require deeper reasoning,
or replace a general concept with a specific one.
Output exactly 10 lines, one rewritten prompt per line, in the same order,
with no numbering and no other text.

Prompts:
1. Explain what overfitting is.
...
```
Paste the results back in as the next round's prompts. After the last round, ask for answers as JSON lines, as in [section 6](#6-route-b-chat-window-fallback), and run `src.prepare_chat_data` with `--method evol_instruct`. The team should agree on the number of rounds so everyone matches.

**Magpie** sends an open-weight chat model only the start of its chat template and lets the model write the user's message itself (see `prompts/magpie.md`). It needs raw text completion on an open model, which neither a chat window nor a standard hosted API like the team's OpenAI key provides. Skip it unless the team sets up an open-weight endpoint.

Whichever method you use, make sure `meta.method` says so, and check with the team that everyone's data is comparable.

---

## 8. Light cleanup (and what not to clean)

The project is about which *filtering method* removes bad examples best. If you fix every quality problem yourself, there is nothing left for the filters to find. So only remove **mechanical problems**, and leave quality problems in.

**Remove these (they are formatting problems, not quality judgments):**
- exact duplicates
- examples with an empty instruction or empty output
- replies that could not be parsed or are cut off mid-sentence
- obvious refusals, such as "As an AI language model, I cannot..."

**Leave these in (the filters should handle them):**
- answers that are wrong, shallow or low quality
- repetitive or very similar examples that are not exact duplicates, unless the team agrees on a near-duplicate rule
- awkward wording, bias or off-topic content

The team should agree on exactly which of these everyone applies, so everyone's data is treated the same way. The checker in the next section finds the mechanical problems for you.

## 9. Save and check your file

Your finished file should be at `data/raw/yourname.jsonl`. The notebook (Route A) and `prepare_chat_data` (Route B) already add the ids and `meta` fields.

**1. Check the format:**
```bash
python -m src.validate data/raw/yourname.jsonl
```
It prints `OK` with a count, or lists the problems it found. Fix them and run it again.

**2. Run the full standard check.** It finds the mechanical problems from section 8 and prints statistics:
```bash
python -m src.check_data data/raw/yourname.jsonl
```
Every line in the report starts with a level:
- `FAIL`: fix it before you commit (duplicates, refusals, leftover template placeholders, format problems).
- `WARN`: look at it and fix it if it is a real problem (very short or very long examples, outputs that look cut off, near-duplicates, missing `meta` fields).
- `NOTE` and `PASS`: information only.

The report ends with `READY` or `NOT READY`. The checker does **not** judge whether your answers are good. Quality problems are left in on purpose, because the filtering methods are what find them. Use `--expected 1000` to compare against a specific count, and `--strict` to make warnings count as failures.

## 10. Submit your data (where it goes)

Your dataset goes in **`data/raw/yourname.jsonl`**, pushed to your own branch with a pull request into `main`. Nothing else needs to be committed for this step. Your notebook, `data/work/` files and any `pasted.txt` stay on your machine.

```bash
git checkout -b yourname/data          # skip if you already created it in section 2
git add data/raw/yourname.jsonl
git status                             # only your .jsonl file should be staged
git commit -m "Add yourname synthetic data"
git push -u origin yourname/data
```

Then open a pull request into `main` on GitHub. The team lead reviews it and merges it. Do not push to `main` directly, and do not edit other people's files in `data/raw/`.

## 11. Tips, cost and troubleshooting

- **Test small first.** Always run the 3-example test, read the examples, and only then submit the full batch.
- **Read a sample by hand** before checking the file. You are looking for broken formatting and obviously wrong instructions, not for quality problems.
- **Don't edit the generated text to improve it.** If you hand-fix examples, your data is not comparable to everyone else's.
- **Never put real people's private information** or copyrighted text into prompts or data.
- **Don't share your key, and don't commit `data/work/`.** The `.gitignore` already excludes `.env`, `data/work/` and Python caches. Run `git status` before committing.

| Problem | Likely fix |
|---------|-----------|
| Section 2 says to set `AUTHOR` | Change `AUTHOR` in the settings block from `"yourname"` to your own name, and run the cell again |
| "Replace the placeholder in DATA_INSTRUCTIONS" | Edit Section 3 and run that cell, then run the test cell again |
| The test says the model did not return exactly 3 examples | Run it again. If it keeps failing, make your instructions simpler and clearer |
| Error about a missing model or no access | Check `MODEL` with the team lead, and check that your key has access to it |
| Authentication error | The key may be wrong or revoked. Ask the team lead |
| Batch status is not `completed` | It can take up to 24 hours. Rerun only the retrieval cell later, never the submission cell |
| Output is `_PARTIAL` | Some requests failed. Read `data/work/batch_errors.jsonl` and tell the team lead before re-running |
| Many near-identical instructions | Add more topics and seed tasks to your instructions |
| `prepare_chat_data` skips many lines | The chatbot probably broke the JSON, for example with unescaped quotes. Ask it to fix those lines, or ask for fewer examples per message |
| `check_data` reports duplicates | Remove the exact duplicates, since that is allowed cleanup, then check again |
| Validation errors | Read the line numbers it prints, and fix those lines |

## 12. Checklist before you commit

- [ ] I used the team's shared settings (model, method, count) or recorded what I used
- [ ] My file is at `data/raw/yourname.jsonl`
- [ ] `python -m src.validate` reports no problems
- [ ] `python -m src.check_data` has no `FAIL` lines, and I have read every `WARN`
- [ ] Each example has a unique `id` and its details in `meta`
- [ ] I only applied the agreed mechanical cleanup
- [ ] My API key is not in any file I am committing (check `git status`), and I cleared the notebook's outputs
- [ ] I pushed `data/raw/yourname.jsonl` to my own branch and opened a pull request into `main`

## 13. For the team lead: handing out keys

- Create a separate project for this work in the OpenAI account, set a monthly **spending limit** on it, and create the keys inside that project. Check the current options in OpenAI's own documentation, since the account settings change over time.
- Prefer **one key per member**, so you can revoke a single key if it leaks and see who used what.
- Send keys privately (a password manager share or a direct message), never in a group channel or the repository.
- Tell everyone the model name, the method and the example count, and keep `configs/generation.yaml` and the notebook's `MODEL`, `METHOD` and `TOTAL_EXAMPLES` in step with each other.
- Do a first trial with one member and a small `TOTAL_EXAMPLES` (for example 40) to confirm the model name, the access and the cost before everyone starts.
- When the data is collected, revoke the keys.
