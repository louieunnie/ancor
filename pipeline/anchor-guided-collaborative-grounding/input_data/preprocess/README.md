# Preprocessing

This folder converts **RiVEG visual entailment outputs** into **JSONL inputs for visual grounding**.

Specifically, it takes the `.txt` files generated after:

1. Running **visual entailment in RiVEG**, and
2. Running `Twitter10000_to_OFA_REC.py` in `data_processing (RiVEG)`

and converts them into JSONL files that include:

* entity predictions,
* CONLL text,
* and entity-level knowledge.

---

## Usage

1. Edit `preprocess.sh`:

   * set the virtual environment name
   * update input/output file paths

2. Run:

```bash
bash preprocess.sh
```

---

## Output

The final output is a JSONL file used as input to our visual grounding model.

---
