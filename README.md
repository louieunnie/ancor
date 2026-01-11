## ANCOR  
**Anchor-guided Collaborative Grounding for Context-aware Grounded Multimodal Named Entity Recognition**

> 🚧 This paper is currently under review.

This repository contains the official implementation of **ANCOR**, a multi-stage framework for grounded multimodal named entity recognition (GMNER).  
The proposed method integrates textual context, visual evidence, and LLM-generated knowledge through a staged pipeline.

---

## Overview

The ANCOR pipeline consists of four sequential stages:

1. **Context-enhanced Multimodal Named Entity Recognition (MNER)**
2. **Entity Visual Knowledge Generation**
3. **Visual Entity Entailment**
4. **Anchor-Guided Collaborative Grounding**

We use the RoBERTa-large tokenizer (HuggingFace Transformers) across all stages.

---

## Data

We use publicly available GMNER benchmarks:

- **Twitter-GMNER** and **Twitter-FMNERG** datasets from:
  - https://github.com/NUSTM/GMNER  
  - https://github.com/NUSTM/FMNERG

- Raw images and VinVL-extracted region features are obtained from:
  - https://github.com/NUSTM/GMNER/tree/main

> Example input files are provided in the corresponding directories for each stage.

---

## LLM-based Data Generation

LLMs are used at **two points** in the pipeline:

1. **Background knowledge generation for textual inputs**, which is incorporated into the MNER input (Stage 1).
2. **Visual knowledge generation for detected entities** after MNER (Stage 2).

- The default LLM is **gpt-4o-mini**.
- For robustness analysis, we additionally evaluate **Qwen-series models** (Qwen2.5-3B, Qwen2.5-7B) using identical prompts.
- Related code is located in:
  - `pipeline/knowledge-generation/`

---

## Stage 1: Context-Enhanced MNER

- Directory: `pipeline/context-enhanced-MNER/`

This stage predicts entity spans and entity types by enriching textual representations with:
- Local textual context,
- Region-level visual features, and
- LLM-generated background knowledge.

Boundary detection is performed using a focal loss, and span-level representations are selectively refined and injected back into token embeddings for final BIO tagging.

---

## Stage 2: Entity Visual Knowledge Generation

- Directory: `pipeline/knowledge-generation/`

For each detected entity, this stage generates visual descriptions and auxiliary knowledge using LLMs, which are later used for grounding and entailment.

---

## Stage 3: Visual Entity Entailment

- We directly use the **Visual Entailment Module** from:
  - https://github.com/JinYuanLi0012/RiVEG

- Fine-grained entity types are obtained by:
  - Generating them in the first step of Stage 2, or
  - Directly using the fine-grained types inferred by the MNER module for **Twitter-FMNERG**.

---

## Stage 4: Anchor-Guided Collaborative Grounding

- Directory: `pipeline/anchor-guided-collaborative-grounding/`

This stage performs final visual grounding by:
- Selecting anchor entities,
- Encouraging collaborative reasoning across entities within the same sample, and
- Enforcing consistency and exclusion constraints among visual regions.

Note that anchor selection is not needed during inference for visual entity grounding.

---

## Usage

Each stage can be executed independently using the provided scripts in its corresponding directory.

- Example input files are already included in each stage directory.
- Please refer to the README or script-level comments within each stage for detailed execution instructions and arguments.

---

## Environment

- Python 3.10.18
- PyTorch
- HuggingFace Transformers
- CUDA-enabled GPU is recommended for training and inference

Additional dependencies are specified within individual stage directories if required.

---

## Acknowledgements

This work builds upon and reuses components from the following projects:

- GMNER / FMNERG (Data Usage): https://github.com/NUSTM/GMNER  /  https://github.com/NUSTM/FMNERG
- RiVEG (Visual Entailment Module): https://github.com/JinYuanLi0012/RiVEG

We thank the authors for making their code and data publicly available.

---

## Notes

- This repository is released for research purposes only.
- The code structure and documentation may be updated after the review process.


