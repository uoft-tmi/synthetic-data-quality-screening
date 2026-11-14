# Magpie template

Magpie has no instruction prompt. Send an aligned open model only the start of its
own chat template, and the model writes the user message by itself.

## 1. Send only this (Llama-3-style example)

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

```

The template tokens are different for every model. Look up your model's own chat template.

## 2. Send the generated instruction back as a normal prompt to get the answer

## 3. Save each pair as one JSONL line

```json
{"id": "yourname-0001", "instruction": "<generated instruction>", "output": "<model response>"}
```
