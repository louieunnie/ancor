from __future__ import annotations
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
import unicodedata
import spacy
import json
import numpy as np
import os

def _norm(s: str) -> str:
    if s is None: return ""
    return unicodedata.normalize("NFKC", s).strip()

def _label_to_type(label: str) -> str:
    lab = _norm(label)
    return lab.split("-", 1)[1] if lab != "O" and "-" in lab else ("O" if lab=="O" else lab)

def load_knowledge_map(path):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            kid = str(obj["imgid"]).strip()   
            mapping[kid] = obj.get("knowledge", "")
    return mapping

def load_bio_file(path: str, img_dir: str | None = None, img_ext: str = ".jpg"):
    """
    Parse GMNER BIO files with image paths
    - 3 columns: token, coarse_label, fine_label 
    - 2 columns: token, tag 
    Returns: list of dicts with keys:
      - img_id: str
      - img_path: str
      - tokens: List[str]
      - coarse_labels: List[str]
      - fine_labels: List[str]
    """

    examples = []
    tokens, coarse_labels, fine_labels, img_id = [], [], [], None

    def flush():
        nonlocal tokens, coarse_labels, fine_labels, img_id
        if not tokens:
            return
        img_path = None
        if img_id is not None:
            img_path = os.path.join(img_dir, f"{img_id}{img_ext}")
        examples.append({
            "img_id": img_id,
            "img_path": img_path,
            "tokens": tokens,
            "coarse_labels": coarse_labels,
            "fine_labels": fine_labels
        })
        tokens, coarse_labels, fine_labels = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line == "":
                continue
            if line.startswith("IMGID:"):
                flush()
                img_id = line.split("IMGID:", 1)[1].strip()
                continue

            parts = line.split("\t") if "\t" in line else line.split()

            if len(parts) == 3:
                token, coarse_label, fine_label = parts[0], parts[1], parts[2]
                tokens.append(token)
                coarse_labels.append(_norm(coarse_label))    
                fine_labels.append(_norm(fine_label))        
            elif len(parts) == 2:
                token, tag = parts[0], parts[1]
                tokens.append(token)
                tag = _norm(tag)                             
                coarse_labels.append(tag)
                fine_labels.append(tag)
            elif len(parts) == 1:
                tokens.append(parts[0])
                coarse_labels.append("O")
                fine_labels.append("O")
            else:
                pass
    # print(examples[0])
    flush()
    return examples

def ner_collate_fn(batch, tokenizer):
    input_ids      = [ex["input_ids"] for ex in batch]
    attention_mask = [ex["attention_mask"] for ex in batch]
    labels         = [ex["labels"] for ex in batch]
    start_labels   = [ex["start_labels"] for ex in batch]
    end_labels     = [ex["end_labels"] for ex in batch]
    gold_spans     = [ex["gold_spans"] for ex in batch]

    region_feats = [ex["region_feats"] for ex in batch]
    region_boxes = [ex["region_boxes"] for ex in batch]
    region_mask  = [ex["region_mask"] for ex in batch]

    raw_texts       = [ex.get("raw_text", None) for ex in batch]
    offset_mappings = [ex.get("offset_mapping", None) for ex in batch]
    tokens_list     = [ex.get("tokens", None) for ex in batch]     # ✅ 추가
    word_ids_list   = [ex.get("word_ids", None) for ex in batch]   # ✅ 추가

    knowledge_masks = [ex.get("knowledge_mask", torch.zeros_like(ex["input_ids"], dtype=torch.bool))
                       for ex in batch]

    batch_enc = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt"
    )
    max_len = batch_enc["input_ids"].size(1)

    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    start_padded  = pad_sequence(start_labels, batch_first=True, padding_value=0)
    end_padded    = pad_sequence(end_labels,   batch_first=True, padding_value=0)

    padded_offsets = []
    for off in offset_mappings:
        if off is None:
            off = torch.zeros((0, 2), dtype=torch.long)
        T = off.size(0)
        pad = torch.zeros((max_len, 2), dtype=torch.long)
        if T > 0:
            L = min(T, max_len)
            pad[:L] = off[:L]
        padded_offsets.append(pad)
    offset_mapping_padded = torch.stack(padded_offsets, dim=0)

    padded_kmask = []
    for km in knowledge_masks:
        T = km.size(0)
        pad = torch.zeros(max_len, dtype=torch.bool)
        if T > 0:
            L = min(T, max_len)
            pad[:L] = km[:L]
        padded_kmask.append(pad)
    knowledge_mask_padded = torch.stack(padded_kmask, dim=0)

    # fine type targets flatten (optional)
    if "span_fine_type_targets" in batch[0]:
        fine_list = [b["span_fine_type_targets"] for b in batch]
        span_fine_type_targets = torch.cat(
            [t if t.numel() > 0 else torch.tensor([], dtype=torch.long) for t in fine_list],
            dim=0
        ) if len(fine_list) > 0 else torch.tensor([], dtype=torch.long)
    else:
        span_fine_type_targets = None
    knowledge_texts = [ex.get("knowledge_text", "") for ex in batch]
    return {
        "input_ids": batch_enc["input_ids"],
        "attention_mask": batch_enc["attention_mask"],
        "labels": labels_padded,
        "start_labels": start_padded,
        "end_labels": end_padded,
        "gold_spans": gold_spans,
        "region_feats": torch.stack(region_feats),
        "region_boxes": torch.stack(region_boxes),
        "region_mask": torch.stack(region_mask),
        "raw_texts": raw_texts,
        "offset_mapping": offset_mapping_padded,
        "knowledge_mask": knowledge_mask_padded,  
        "span_fine_type_targets": span_fine_type_targets,
        "knowledge_text": knowledge_texts,
        "tokens": tokens_list,          
        "word_ids": word_ids_list,     
        "img_id": [ex.get("img_id", None) for ex in batch], 
        "caption": [ex.get("caption", None) for ex in batch], 
    }

def bio_to_spans(tags: List[str], label2typeid: Dict[str, int]) -> List[Span]:
    Span = Tuple[int, int, int]  # (start_idx, end_idx, type_id)
    spans: List[Span] = []
    start, end, cur_type = None, None, None

    for i, tag in enumerate(tags):
        # PAD/-100 
        if tag == "O" or tag == -100 or tag is None:
            if start is not None:
                t = _label_to_type(cur_type)           
                spans.append((start, end, label2typeid[t]))
                start, end, cur_type = None, None, None
            continue

        if isinstance(tag, str):
            tag = _norm(tag)                            
            prefix, tlabel = (tag.split("-", 1) if "-" in tag else ("O", None))
        else:
            prefix, tlabel = "O", None

        if prefix == "B":
            if start is not None:
                t = _label_to_type(cur_type)           
                spans.append((start, end, label2typeid[t]))
            start, end, cur_type = i, i, tlabel
        elif prefix == "I":
            if start is not None and cur_type == tlabel:
                end = i
            else:
                if start is not None:
                    t = _label_to_type(cur_type)       
                    spans.append((start, end, label2typeid[t]))
                start, end, cur_type = i, i, tlabel

    if start is not None:
        t = _label_to_type(cur_type)                   
        spans.append((start, end, label2typeid[t])) 
    return spans

def make_boundary_labels_from_aligned(aligned_tags: List[str],
                                      label2typeid: Dict[str, int]):
    """
    Generate start_labels and end_labels from aligned BIO tags.
    - aligned_tags: List of BIO tags aligned with sub-tokens.
    - aligned_tags: ex) ["O","B-PER","I-PER","O","B-LOC",...]
    - Returns:
        start_labels: LongTensor [T] (0/1)
        end_labels:   LongTensor [T] (0/1)
        gold_spans:   List[Tuple[int,int,int]]
    """
    spans = bio_to_spans(aligned_tags, label2typeid)
    T = len(aligned_tags)
    start_labels = torch.zeros(T, dtype=torch.long)
    end_labels = torch.zeros(T, dtype=torch.long)
    for s, e, _ in spans:
        if 0 <= s < T: start_labels[s] = 1
        if 0 <= e < T: end_labels[e] = 1
    # print(f"aligned_tags: {aligned_tags}") 
    # print(f"label2typeid: {label2typeid}") 
    # print(f"start_labels: {start_labels}")
    # print(f"end_labels: {end_labels}")
    # print(f"spans: {spans}") 
    return start_labels, end_labels, spans


def prepare_region_tensors_from_npz(npz_path, max_regions=32, device='cpu'):
    data = np.load(npz_path, allow_pickle=True)
    num_boxes = int(data['num_boxes'])
    img_w = float(data['image_w'])
    img_h = float(data['image_h'])

    boxes = data['bounding_boxes']        # [num_boxes, 4]
    feats = data['box_features']          # [num_boxes, D_v]
    scores = data['scores'] if 'scores' in data else np.ones((num_boxes,), dtype=np.float32)
    if scores.dtype == object:
        scores = np.array([float(s) for s in scores], dtype=np.float32)

    # select top max_regions based on scores
    order = np.argsort(scores)[::-1][:max_regions]
    boxes = boxes[order]
    feats = feats[order]

    feats_list, boxes_list, mask_list = [], [], []

    for box, feat in zip(boxes, feats):
        x1, y1, x2, y2 = box
        # normalize box coordinates
        nx1, ny1 = x1 / img_w, y1 / img_h
        nx2, ny2 = x2 / img_w, y2 / img_h
        w_norm, h_norm = max(nx2 - nx1, 1e-6), max(ny2 - ny1, 1e-6)
        area = w_norm * h_norm
        aspect = w_norm / h_norm

        box_pe = np.array([nx1, ny1, nx2, ny2, area, aspect], dtype=np.float32)
        feats_list.append(feat.astype(np.float32))
        boxes_list.append(box_pe)
        mask_list.append(1.0)

    # padding
    while len(feats_list) < max_regions:
        feats_list.append(np.zeros_like(feats_list[0]))
        boxes_list.append(np.zeros(6, dtype=np.float32))
        mask_list.append(0.0)

    region_feats = torch.tensor(np.stack(feats_list), device=device)  # [max_regions, D_v]
    region_boxes = torch.tensor(np.stack(boxes_list), device=device)  # [max_regions, 6]
    region_mask = torch.tensor(mask_list, dtype=torch.float32, device=device)  # [max_regions]

    return region_feats, region_boxes, region_mask

def tokenize_and_align_labels_with_spans(
    example: Dict,
    tokenizer,
    coarse_label2id: Dict[str, int] = None,
    fine_label2id: Dict[str, int] = None,
    label_set: str = "coarse",
    label2typeid: Dict[str, int] = None,
):
    """
    - example["tokens"] + example["knowledge"] can be included
    - knowledge tokens are excluded from loss/evaluation
    """

    if label_set == "coarse":
        raw_labels = example["coarse_labels"]
        label2id = coarse_label2id
    elif label_set == "fine":
        raw_labels = example["fine_labels"]
        label2id = fine_label2id
    else:
        raise ValueError("label_set must be 'coarse' or 'fine'")

    tokens = example["tokens"]

    if "[KNOWLEDGE]" in tokens:
        knowledge_start_idx = tokens.index("[KNOWLEDGE]")
    else:
        knowledge_start_idx = None

    tok = tokenizer(tokens, is_split_into_words=True, truncation=True,
                    return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = tok["offset_mapping"].squeeze(0)
    word_ids = tok.word_ids(batch_index=0)

    aligned_ids: List[int] = []
    aligned_tags: List[str] = []
    knowledge_mask = []

    prev = None
    for i, widx in enumerate(word_ids):
        is_knowledge = False
        if widx is not None and knowledge_start_idx is not None and widx >= knowledge_start_idx:
            is_knowledge = True
        knowledge_mask.append(is_knowledge)

        if widx is None:
            aligned_ids.append(-100)
            aligned_tags.append("PAD")
        elif widx != prev:
            tag = raw_labels[widx] if widx < len(raw_labels) else "O"
            aligned_ids.append(label2id[tag] if not is_knowledge else -100)
            aligned_tags.append(tag if not is_knowledge else "O")
        else:
            tag = raw_labels[widx] if widx < len(raw_labels) else "O"
            if tag.startswith("B-"):
                tag = tag.replace("B-", "I-")
            aligned_ids.append(label2id[tag] if not is_knowledge else -100)
            aligned_tags.append(tag if not is_knowledge else "O")
        prev = widx

    if label2typeid is None:
        seen, types = set(), []
        for t in raw_labels:
            if t == "O":
                continue
            tt = _label_to_type(t)
            if tt not in seen:
                seen.add(tt)
                types.append(tt)
        label2typeid = {t: i for i, t in enumerate(types)}

    start_labels, end_labels, gold_spans = make_boundary_labels_from_aligned(
        [t if t != "PAD" else "O" for t in aligned_tags],
        label2typeid
    )

    knowledge_mask_tensor = torch.tensor(knowledge_mask, dtype=torch.bool)
    aligned_ids = torch.tensor(aligned_ids, dtype=torch.long)
    aligned_ids[knowledge_mask_tensor] = -100
    start_labels[knowledge_mask_tensor] = 0
    end_labels[knowledge_mask_tensor] = 0

    if knowledge_start_idx is not None:
        raw_text = " ".join(tokens[:knowledge_start_idx])
    else:
        raw_text = " ".join(tokens)

    return {
        "input_ids": tok["input_ids"].squeeze(0),
        "attention_mask": tok["attention_mask"].squeeze(0),
        "labels": aligned_ids,
        "start_labels": start_labels,
        "end_labels": end_labels,
        "gold_spans": gold_spans,
        "raw_text": raw_text,
        "offset_mapping": offset_mapping,
        "knowledge_mask": knowledge_mask_tensor,
        "img_path": example.get("img_path"),
        "img_id": example.get("img_id"),
        "tokens": tokens,          
        "word_ids": word_ids       
    }


def normalize_entity(text: str) -> str:
    if text is None: return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = " ".join(text.split()) 
    return text.strip()


def merge_examples_with_knowledge(examples, knowledge_map, tokenizer, max_knowledge_tokens=50):
    new_examples = []
    for ex in examples:
        imgid = str(ex.get("img_id"))  
        tokens = ex["tokens"]
        coarse_labels = ex["coarse_labels"]
        fine_labels = ex["fine_labels"]

        knowledge_mask = [0] * len(tokens)

        if imgid in knowledge_map:
            knowledge = normalize_entity(knowledge_map[imgid])
            kn_tokens = tokenizer.tokenize(knowledge)[:max_knowledge_tokens]

            tokens = tokens + ["[KNOWLEDGE]"] + kn_tokens
            coarse_labels = coarse_labels + ["O"] * (1 + len(kn_tokens))
            fine_labels   = fine_labels   + ["O"] * (1 + len(kn_tokens))
            knowledge_mask = knowledge_mask + [1] * (1 + len(kn_tokens))

            ex["knowledge"] = knowledge
        else:
            print(f"[WARN] No knowledge for imgid {imgid}")  # Always log when no match is found

        ex["tokens"] = tokens
        ex["coarse_labels"] = coarse_labels
        ex["fine_labels"] = fine_labels
        ex["knowledge_mask"] = knowledge_mask
        new_examples.append(ex)
    return new_examples
