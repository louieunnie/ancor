import torch
import torch.nn.functional as F
import numpy as np
from math import inf
import json
import logging
import os

logger = logging.getLogger(__name__)

def _get_token_logits(outputs):
    if isinstance(outputs, dict):
        if "logits" in outputs:
            return outputs["logits"]
        if "token_logits" in outputs:
            return outputs["token_logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    if hasattr(outputs, "token_logits"):
        return outputs.token_logits
    raise KeyError("No token logits found in model outputs.")

def _bio_valid(prev_tag, curr_tag):
    if curr_tag == "O":
        return True
    if "-" not in curr_tag:
        return False
    c_pref, c_type = curr_tag.split("-", 1)

    if prev_tag is None or prev_tag == "O":
        return c_pref == "B"
    if "-" not in prev_tag:
        return c_pref == "B"
    p_pref, p_type = prev_tag.split("-", 1)

    if c_pref == "B":
        return True  
    # c_pref == "I"
    return (p_pref in ["B", "I"]) and (p_type == c_type)

def _make_maps(id2label):
    if isinstance(id2label, dict):
        id2lab = [None]*len(id2label)
        for k,v in id2label.items():
            try:
                id2lab[int(k)] = v
            except:
                pass
        lab2id = {v:i for i,v in enumerate(id2lab)}
    else:
        id2lab = list(id2label)
        lab2id = {v:i for i,v in enumerate(id2lab)}
    return id2lab, lab2id


def viterbi_redecode_with_threshold(logits_t, conf_t, id2label, k=3,
                                    thr_global=0.95, per_type_threshold=None,
                                    keep_mask=None):
    """
    logits_t: [T, C] 
    conf_t  : [T]     (argmax softmax probability)
    keep_mask: [T] bool, (optional)
    """
    T, C = logits_t.shape
    id2lab, lab2id = _make_maps(id2label)
    logp = F.log_softmax(torch.tensor(logits_t), dim=-1).numpy()  # [T,C]
    probs = np.exp(logp)

    argmax_ids = probs.argmax(-1)  # [T]
    argmax_tags = [id2lab[i] for i in argmax_ids]
    conf = np.take_along_axis(probs, argmax_ids[:,None], axis=-1).squeeze(-1)  # [T]

    cand_ids_list = []
    for t in range(T):

        if keep_mask is not None and keep_mask[t]:
            cand_ids_list.append([argmax_ids[t]])
            continue

        thr = thr_global
        pred_tag = id2lab[argmax_ids[t]]
        if per_type_threshold and pred_tag != "O" and "-" in pred_tag:
            pred_type = pred_tag.split("-", 1)[1]
            if pred_type in per_type_threshold:
                thr = per_type_threshold[pred_type]

        if conf[t] >= thr:
            cand_ids_list.append([argmax_ids[t]])
        else:
            topk = min(k, C)
            top_ids = np.argpartition(-probs[t], topk-1)[:topk]
            top_ids = top_ids[np.argsort(-probs[t][top_ids])]
            if lab2id.get("O", None) is not None and lab2id["O"] not in top_ids:
                top_ids = np.concatenate([top_ids, [lab2id["O"]]])
            cand_ids_list.append(list(dict.fromkeys(top_ids.tolist())))  

    dp = []
    bp = []
    first = []
    for j, cid in enumerate(cand_ids_list[0]):
        tag = id2lab[cid]
        if _bio_valid(None, tag):
            first.append(logp[0, cid])
        else:
            first.append(-inf)
    dp.append(first)
    bp.append([-1]*len(cand_ids_list[0]))

    for t in range(1, T):
        curr = []
        curr_bp = []
        for j, cid in enumerate(cand_ids_list[t]):
            tag = id2lab[cid]
            best_score = -inf
            best_k = -1
            for kidx, pid in enumerate(cand_ids_list[t-1]):
                ptag = id2lab[pid]
                if not _bio_valid(ptag, tag):
                    continue
                sc = dp[t-1][kidx] + logp[t, cid]
                if sc > best_score:
                    best_score = sc
                    best_k = kidx
            curr.append(best_score)
            curr_bp.append(best_k)
        dp.append(curr)
        bp.append(curr_bp)
    last_idx = int(np.argmax(dp[-1]))
    seq_ids = [None]*T
    seq_ids[-1] = cand_ids_list[-1][last_idx]
    for t in range(T-1, 0, -1):
        last_idx = bp[t][last_idx]
        if last_idx < 0:
            seq_ids[t-1] = argmax_ids[t-1]
        else:
            seq_ids[t-1] = cand_ids_list[t-1][last_idx]
    return [id2lab[i] for i in seq_ids], conf  

def fix_bio_sequence_with_logits(tokens, tags, logits, id2label):
    """
    logits: [len(tokens), num_labels]
    id2label: {label_id: "B-ORG", ...}
    """
    fixed = []
    prev_tag, prev_type = "O", None

    for i, tag in enumerate(tags):
        if tag == "O" or tag is None:
            fixed.append("O")
            prev_tag, prev_type = "O", None
            continue

        prefix, ttype = tag.split("-", 1)

        if prefix == "I" and (prev_tag == "O" or prev_type != ttype):
            b_label = f"B-{ttype}"
            i_label = f"I-{prev_type}" if prev_type else None

            b_logit = logits[i, list(id2label.values()).index(b_label)] if b_label in id2label.values() else -np.inf
            i_logit = logits[i, list(id2label.values()).index(i_label)] if i_label and i_label in id2label.values() else -np.inf

            chosen_label = b_label if b_logit >= i_logit else i_label or b_label
            prefix, ttype = chosen_label.split("-", 1)

        elif prefix == "B" and prev_tag in ["B", "I"] and prev_type == ttype:
            b_label = f"B-{ttype}"
            i_label = f"I-{ttype}"
            b_logit = logits[i, list(id2label.values()).index(b_label)]
            i_logit = logits[i, list(id2label.values()).index(i_label)]
            prefix = "B" if b_logit >= i_logit else "I"

        fixed.append(f"{prefix}-{ttype}")
        prev_tag, prev_type = prefix, ttype

    return fixed

def evaluate(
    ner_model, dataloader, tokenizer, id2label, save_path=None,
    redo=True,           
    thr_global=0.95,    
    per_type_threshold=None,  
    topk=3               
):
    ner_model.eval()
    gold_spans_all, pred_spans_all = [], []

    def get_spans_from_tags(tags):
        spans = []
        start, label = None, None
        for i, tag in enumerate(tags):
            if tag == "O":
                if start is not None:
                    spans.append((label, start, i - 1))
                    start, label = None, None
                continue
            ttype, tlabel = tag.split("-", 1)
            if ttype == "B":
                if start is not None:
                    spans.append((label, start, i - 1))
                start, label = i, tlabel
            elif ttype == "I" and label == tlabel:
                continue
            else:
                if start is not None:
                    spans.append((label, start, i - 1))
                start, label = i, tlabel
        if start is not None:
            spans.append((label, start, len(tags) - 1))
        return set(spans)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fout_json = open(save_path + ".jsonl", "w", encoding="utf-8")
        fout_bio  = open(save_path + ".bio", "w", encoding="utf-8")
    else:
        fout_json = fout_bio = None

    total_lowconf = 0
    total_changed = 0
    total_tokens = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for batch in dataloader:
                device = next(ner_model.parameters()).device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = ner_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    region_feats=batch["region_feats"].to(device),
                    region_boxes=batch["region_boxes"].to(device),
                    region_mask=batch["region_mask"].to(device),
                    knowledge_texts=batch["knowledge_text"]
                )

                logits = _get_token_logits(outputs)                        # [B,T,C]
                probs  = torch.softmax(logits, dim=-1)
                conf_all, pred_ids = probs.max(dim=-1)                    # [B,T], [B,T]

                B, T = labels.size()
                for b in range(B):
                    true_seq, pred_seq = [], []
                    for t in range(T):
                        if labels[b, t].item() == -100:
                            true_seq.append(None)
                            pred_seq.append(None)
                        else:
                            true_seq.append(id2label[labels[b, t].item()])
                            pred_seq.append(id2label[pred_ids[b, t].item()])

                    word_ids = batch["word_ids"][b]
                    input_tokens = batch["input_ids"][b]

                    tokens, gold_tags, pred_tags = [], [], []
                    word_confidences, word_logits = [], []
                    wid_to_token_ids = {}
                    keep_mask = []

                    for t, wid in enumerate(word_ids):
                        if wid is None or true_seq[t] is None:
                            continue
                        wid_to_token_ids.setdefault(wid, []).append(t)

                    for wid, token_indices in sorted(wid_to_token_ids.items()):
                        tok_ids = [input_tokens[i].item() for i in token_indices]
                        word_str = tokenizer.decode(tok_ids, skip_special_tokens=True).strip()
                        tokens.append(word_str)

                        first_t = token_indices[0]
                        g_tag = true_seq[first_t]
                        p_tag = pred_seq[first_t]
                        gold_tags.append(g_tag)
                        pred_tags.append(p_tag)

                        word_logits.append(logits[b, first_t].detach().cpu().numpy())
                        word_confidences.append(conf_all[b, first_t].item())
                        keep_mask.append(False)

                    pred_tags = fix_bio_sequence_with_logits(
                        tokens, pred_tags, np.array(word_logits), id2label
                    )

                    if redo:
                        logits_np = np.array(word_logits)
                        conf_np = np.array(word_confidences)

                        low_conf_count = (conf_np < thr_global).sum()
                        total_lowconf += low_conf_count
                        total_tokens += len(conf_np)

                        base_tags = pred_tags[:]
                        pred_tags, _ = viterbi_redecode_with_threshold(
                            logits_np, conf_np, id2label,
                            k=topk,
                            thr_global=thr_global,
                            per_type_threshold=per_type_threshold,
                            keep_mask=np.array(keep_mask, dtype=bool)
                        )

                        changed = sum(bt != pt for bt, pt in zip(base_tags, pred_tags))
                        total_changed += changed

                    gold_spans_all.append(get_spans_from_tags(gold_tags))
                    pred_spans_all.append(get_spans_from_tags(pred_tags))

                    if fout_json:
                        fout_json.write(json.dumps({
                            "tokens": tokens,
                            "gold_tags": gold_tags,
                            "pred_tags": pred_tags,
                            "confidences": [round(c, 4) for c in word_confidences]
                        }, ensure_ascii=False) + "\n")

                    if fout_bio:
                        imgid = batch["img_id"][b]
                        fout_bio.write(f"IMGID:{imgid}\n")
                        for w, g, p, c in zip(tokens, gold_tags, pred_tags, word_confidences):
                            fout_bio.write(f"{w}\t{g}\t{p}\t{c:.4f}\n")
                        fout_bio.write("\n")

    if fout_json: fout_json.close()
    if fout_bio: fout_bio.close()

    # ---- Span-level F1 ----
    TP = FP = FN = 0
    for gold_spans, pred_spans in zip(gold_spans_all, pred_spans_all):
        TP += len(gold_spans & pred_spans)
        FP += len(pred_spans - gold_spans)
        FN += len(gold_spans - pred_spans)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info(f"\n== Span-level F1: {f1:.4f} "
                f"(P={precision:.4f}, R={recall:.4f}, TP={TP}, FP={FP}, FN={FN})")
    # if redo:
    #     logger.info(f"[DEBUG] thr={thr_global:.2f} | low-conf tokens={total_lowconf}/{total_tokens} "
    #                 f"({total_lowconf/total_tokens*100:.2f}%) | changed tags={total_changed}")

    ner_model.train()
    return {"precision": precision, "recall": recall, "f1": f1}
