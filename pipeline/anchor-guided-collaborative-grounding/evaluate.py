import torch
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from utils import compute_iou
import torch

@torch.no_grad()
def evaluate_dev(model, device, loader, iou_thresh=0.5):
    """Evaluate the model on the development set. (do not use anchoring here)"""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        tb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        _, probs, _, _, _ = model(tb["input_ids"], tb["attn_masks"], tb["region_feats"])
        B, E, R = probs.shape

        for b in range(B):
            sample = loader.dataset.samples[batch["index"][b].item()]
            num_entities = len(sample["entities"])

            for e in range(num_entities):
        
                pred_r = probs[b, e].argmax().item()
                pred_box = tb["region_boxes"][b, pred_r].cpu().numpy()

                gold_boxes = sample["entities"][e].get("bboxes", [])
                
                if not gold_boxes:
                    continue  

                correct = any(compute_iou(pred_box, g) >= iou_thresh for g in gold_boxes)

                all_preds.append(1 if correct else 0)
                all_labels.append(1)

    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return p, r, f1

@torch.no_grad()
def inference_and_save(model, device, loader, save_path="inference_results.jsonl"):
    model.eval()
    results = []

    with open(save_path, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(loader):
            tb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, probs, _, _, _ = model(tb["input_ids"], tb["attn_masks"], tb["region_feats"])
            B, E, R = probs.shape

            for b in range(B):
                sample = loader.dataset.samples[batch_idx * loader.batch_size + b]

                entity_results = []
                for e, ent in enumerate(sample["entities"]):
                    # pred_label=0 → box=None
                    if ent.get("pred_label", 1) == 0:
                        pred_box = None
                    else:
                        pred_r = probs[b, e].argmax().item()
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