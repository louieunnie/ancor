import torch

MAX_REGIONS = 40

def collate_fn(batch):
    if len(batch) == 0: return None
    max_R = min(MAX_REGIONS, max(b["region_feats"].shape[0] for b in batch))
    max_E = max(len(b["entities"]) for b in batch)

    def pad_2d(t, max_len):
        R, D = t.shape
        if R >= max_len: return t[:max_len]
        return torch.cat([t, torch.zeros(max_len - R, D)], dim=0)

    out = {"region_feats": [], "region_boxes": [], "input_ids": [],
           "attn_masks": [], "labels": [], "ent_types": [], "img_ids": []}

    for b in batch:
        out["region_feats"].append(pad_2d(b["region_feats"], max_R))
        out["region_boxes"].append(pad_2d(b["region_boxes"], max_R))
        input_ids_e = torch.zeros(max_E, 128, dtype=torch.long)
        attn_mask_e = torch.zeros(max_E, 128, dtype=torch.long)
        labels_e = torch.zeros(max_E, max_R)
        ent_types_e = [""] * max_E
        for i, ent in enumerate(b["entities"]):
            input_ids_e[i] = ent["input_ids"]
            attn_mask_e[i] = ent["attn_mask"]
            lab = b["labels"][i]
            labels_e[i, :len(lab)] = lab[:max_R]
            ent_types_e[i] = ent["ent_type"]
        out["input_ids"].append(input_ids_e)
        out["attn_masks"].append(attn_mask_e)
        out["labels"].append(labels_e)
        out["ent_types"].append(ent_types_e)
        out["img_ids"].append(b["img_id"])

    for k in ["region_feats", "region_boxes", "input_ids", "attn_masks", "labels"]:
        out[k] = torch.stack(out[k])
    return out

def collate_fn_infer(batch):
    if len(batch) == 0: return None
    max_R = min(MAX_REGIONS, max(b["region_feats"].shape[0] for b in batch))
    max_E = max(len(b["entities"]) for b in batch)

    def pad_2d(t, max_len):
        R, D = t.shape
        if R >= max_len: return t[:max_len]
        return torch.cat([t, torch.zeros(max_len - R, D)], dim=0)

    out = {"region_feats": [], "region_boxes": [], "input_ids": [],
           "attn_masks": [], "ent_types": [], "img_ids": [], "index": []}  

    for b in batch:
        out["region_feats"].append(pad_2d(b["region_feats"], max_R))
        out["region_boxes"].append(pad_2d(b["region_boxes"], max_R))
        input_ids_e = torch.zeros(max_E, 128, dtype=torch.long)
        attn_mask_e = torch.zeros(max_E, 128, dtype=torch.long)
        ent_types_e = [""] * max_E
        for i, ent in enumerate(b["entities"]):
            input_ids_e[i] = ent["input_ids"]
            attn_mask_e[i] = ent["attn_mask"]
            ent_types_e[i] = ent["ent_type"]
        out["input_ids"].append(input_ids_e)
        out["attn_masks"].append(attn_mask_e)
        out["ent_types"].append(ent_types_e)
        out["img_ids"].append(b["img_id"])
        out["index"].append(b["index"])   

    for k in ["region_feats", "region_boxes", "input_ids", "attn_masks"]:
        out[k] = torch.stack(out[k])
    out["index"] = torch.tensor(out["index"], dtype=torch.long)  
    return out