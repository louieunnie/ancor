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
def inference_and_save(
    model,
    device,
    loader,
    save_path="inference_results.jsonl",
    save_all_entities_path=None,
):
    model.eval()
    results = []
    all_entities_results = []
    joint_mode = os.getenv("JOINT_INFER_MODE", "none").strip().lower()

    with open(save_path, "w", encoding="utf-8") as f:
        fout_all = None
        if save_all_entities_path:
            os.makedirs(os.path.dirname(save_all_entities_path) or ".", exist_ok=True)
            fout_all = open(save_all_entities_path, "w", encoding="utf-8")
        for batch_idx, batch in enumerate(loader):
            tb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            B = tb["input_ids"].shape[0]

            for b in range(B):
                sample_idx = int(batch["index"][b].item())
                sample = loader.dataset.samples[sample_idx]
                num_entities = len(sample["entities"])
                active_indices = [
                    i for i, ent in enumerate(sample["entities"])
                    if int(ent.get("pred_label", 1)) != 0
                ]

                pred_boxes = [None] * num_entities
                if active_indices:
                    input_ids_active = tb["input_ids"][b, active_indices].unsqueeze(0)
                    attn_masks_active = tb["attn_masks"][b, active_indices].unsqueeze(0)
                    region_feats_b = tb["region_feats"][b].unsqueeze(0)

                    _, probs_active, _, _, _ = model(input_ids_active, attn_masks_active, region_feats_b)
                    probs_e = probs_active[0, :len(active_indices)]  # [E_active, R]
                    if joint_mode == "greedy_conflict":
                        selected_regions = _select_regions_greedy_conflict(probs_e)
                    else:
                        selected_regions = probs_e.argmax(dim=-1)

                    for local_e, original_e in enumerate(active_indices):
                        pred_r = int(selected_regions[local_e].item())
                        pred_boxes[original_e] = tb["region_boxes"][b, pred_r].cpu().numpy().tolist()

                entity_results = []
                for e, ent in enumerate(sample["entities"]):
                    entity_results.append({
                        "entity": ent["text"],
                        "ent_type": ent["type"],
                        "pred_box": pred_boxes[e],
                    })

                result = {
                    "img_id": sample["img"],
                    "entities": entity_results,
                }
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

                # Additional output: explicitly include pred_label for every entity.
                # pred_label==0 keeps pred_box=None without region selection.
                if fout_all is not None:
                    all_entity_results = []
                    for src_ent, ent_item in zip(sample["entities"], entity_results):
                        all_entity_results.append({
                            "entity": ent_item.get("entity"),
                            "ent_type": ent_item.get("ent_type"),
                            "pred_label": int(src_ent.get("pred_label", 1)),
                            "pred_box": ent_item.get("pred_box"),
                        })

                    all_result = {
                        "img_id": sample["img"],
                        "entities": all_entity_results,
                    }
                    all_entities_results.append(all_result)
                    fout_all.write(json.dumps(all_result, ensure_ascii=False) + "\n")

        if fout_all is not None:
            fout_all.close()

    print(f"[INFO] Inference results saved to {save_path}")
    if save_all_entities_path:
        print(f"[INFO] Inference all-entities results saved to {save_all_entities_path}")