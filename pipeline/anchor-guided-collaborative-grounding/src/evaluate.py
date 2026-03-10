import torch
import json
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support
from utils import compute_iou
import torch


def _select_regions_greedy_conflict(prob_er: torch.Tensor):
    """
    Joint decoding over entities with greedy conflict resolution.
    - Input: prob_er [E, R]
    - Output: selected region indices per entity [E]
    """
    E, R = prob_er.shape
    selected = torch.full((E,), -1, dtype=torch.long, device=prob_er.device)
    used_regions = torch.zeros((R,), dtype=torch.bool, device=prob_er.device)

    candidates = []
    for e in range(E):
        for r in range(R):
            candidates.append((float(prob_er[e, r].item()), e, r))
    candidates.sort(key=lambda x: x[0], reverse=True)

    for _, e, r in candidates:
        if selected[e] != -1:
            continue
        if used_regions[r]:
            continue
        selected[e] = r
        used_regions[r] = True
        if bool((selected != -1).all()):
            break

    # Fallback for any unassigned entity: best remaining region (or local argmax)
    for e in range(E):
        if selected[e] != -1:
            continue
        avail = (~used_regions).nonzero(as_tuple=True)[0]
        if len(avail) > 0:
            best_local = avail[prob_er[e, avail].argmax()]
        else:
            best_local = prob_er[e].argmax()
        selected[e] = best_local
        used_regions[selected[e]] = True

    return selected

@torch.no_grad()
def evaluate_dev(model, device, loader, iou_thresh=0.5):
    """Evaluate the model on the development set. (do not use anchoring here)"""
    model.eval()
    all_preds, all_labels = [], []
    # Stratified stats by number of entities in each sample.
    # bucket[k] = {"correct": int, "total": int, "samples": int}
    bucket = {}
    joint_mode = os.getenv("JOINT_INFER_MODE", "none").strip().lower()

    for batch in loader:
        tb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        _, probs, _, _, _ = model(tb["input_ids"], tb["attn_masks"], tb["region_feats"])
        B, E, R = probs.shape

        for b in range(B):
            sample = loader.dataset.samples[batch["index"][b].item()]
            num_entities = len(sample["entities"])
            sample_correct = 0
            sample_total = 0
            probs_e = probs[b, :num_entities]  # [E, R]
            if joint_mode == "greedy_conflict":
                selected_regions = _select_regions_greedy_conflict(probs_e)
            else:
                selected_regions = probs_e.argmax(dim=-1)

            for e in range(num_entities):
                pred_r = int(selected_regions[e].item())
                pred_box = tb["region_boxes"][b, pred_r].cpu().numpy()

                gold_boxes = sample["entities"][e].get("bboxes", [])
                
                if not gold_boxes:
                    continue  

                correct = any(compute_iou(pred_box, g) >= iou_thresh for g in gold_boxes)
                sample_total += 1
                if correct:
                    sample_correct += 1

                all_preds.append(1 if correct else 0)
                all_labels.append(1)

            if sample_total > 0:
                if num_entities not in bucket:
                    bucket[num_entities] = {"correct": 0, "total": 0, "samples": 0}
                bucket[num_entities]["correct"] += sample_correct
                bucket[num_entities]["total"] += sample_total
                bucket[num_entities]["samples"] += 1

    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    bucket_acc = {}
    for k in sorted(bucket.keys()):
        total = bucket[k]["total"]
        acc = (bucket[k]["correct"] / total) if total > 0 else 0.0
        bucket_acc[k] = {
            "acc": acc,
            "correct": bucket[k]["correct"],
            "total": total,
            "samples": bucket[k]["samples"],
        }
    return p, r, f1, bucket_acc

@torch.no_grad()
def inference_and_save(model, device, loader, save_path="inference_results.jsonl"):
    model.eval()
    results = []
    joint_mode = os.getenv("JOINT_INFER_MODE", "none").strip().lower()

    with open(save_path, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(loader):
            tb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, probs, _, _, _ = model(tb["input_ids"], tb["attn_masks"], tb["region_feats"])
            B, E, R = probs.shape

            for b in range(B):
                sample = loader.dataset.samples[batch_idx * loader.batch_size + b]
                num_entities = len(sample["entities"])
                probs_e = probs[b, :num_entities]  # [E, R]
                if joint_mode == "greedy_conflict":
                    selected_regions = _select_regions_greedy_conflict(probs_e)
                else:
                    selected_regions = probs_e.argmax(dim=-1)

                entity_results = []
                for e, ent in enumerate(sample["entities"]):
                    # pred_label=0 → box=None
                    if ent.get("pred_label", 1) == 0:
                        pred_box = None
                    else:
                        pred_r = int(selected_regions[e].item())
                        pred_box = tb["region_boxes"][b, pred_r].cpu().numpy().tolist()

                    entity_results.append({
                        "entity": ent["text"],
                        "ent_type": ent["type"],
                        "pred_box": pred_box  
                    })

                result = {
                    "img_id": sample["img"],
                    "entities": entity_results,
                }
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[INFO] Inference results saved to {save_path}")