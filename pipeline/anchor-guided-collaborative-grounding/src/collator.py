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
           "attn_masks": [], "labels": [], "ent_types": [], "img_ids": [],
           "ent_input_ids": [], "ent_attn_masks": [],
           "type_input_ids": [], "type_attn_masks": [],
           "ctx_input_ids": [], "ctx_attn_masks": [],
           "know_input_ids": [], "know_attn_masks": []}

    for b in batch:
        out["region_feats"].append(pad_2d(b["region_feats"], max_R))
        out["region_boxes"].append(pad_2d(b["region_boxes"], max_R))
        input_ids_e = torch.zeros(max_E, 128, dtype=torch.long)
        attn_mask_e = torch.zeros(max_E, 128, dtype=torch.long)
        ent_input_ids_e = torch.zeros(max_E, 32, dtype=torch.long)
        ent_attn_mask_e = torch.zeros(max_E, 32, dtype=torch.long)
        type_input_ids_e = torch.zeros(max_E, 16, dtype=torch.long)
        type_attn_mask_e = torch.zeros(max_E, 16, dtype=torch.long)
        ctx_input_ids_e = torch.zeros(max_E, 128, dtype=torch.long)
        ctx_attn_mask_e = torch.zeros(max_E, 128, dtype=torch.long)
        know_input_ids_e = torch.zeros(max_E, 64, dtype=torch.long)
        know_attn_mask_e = torch.zeros(max_E, 64, dtype=torch.long)
        labels_e = torch.zeros(max_E, max_R)
        ent_types_e = [""] * max_E
        for i, ent in enumerate(b["entities"]):
            input_ids_e[i] = ent["input_ids"]
            attn_mask_e[i] = ent["attn_mask"]
            ent_input_ids_e[i] = ent["ent_input_ids"]
            ent_attn_mask_e[i] = ent["ent_attn_mask"]
            type_input_ids_e[i] = ent["type_input_ids"]
            type_attn_mask_e[i] = ent["type_attn_mask"]
            ctx_input_ids_e[i] = ent["ctx_input_ids"]
            ctx_attn_mask_e[i] = ent["ctx_attn_mask"]
            know_input_ids_e[i] = ent["know_input_ids"]
            know_attn_mask_e[i] = ent["know_attn_mask"]
            lab = b["labels"][i]
            labels_e[i, :len(lab)] = lab[:max_R]
            ent_types_e[i] = ent["ent_type"]
        out["input_ids"].append(input_ids_e)
        out["attn_masks"].append(attn_mask_e)
        out["ent_input_ids"].append(ent_input_ids_e)
        out["ent_attn_masks"].append(ent_attn_mask_e)
        out["type_input_ids"].append(type_input_ids_e)
        out["type_attn_masks"].append(type_attn_mask_e)
        out["ctx_input_ids"].append(ctx_input_ids_e)
        out["ctx_attn_masks"].append(ctx_attn_mask_e)
        out["know_input_ids"].append(know_input_ids_e)
        out["know_attn_masks"].append(know_attn_mask_e)
        out["labels"].append(labels_e)
        out["ent_types"].append(ent_types_e)
        out["img_ids"].append(b["img_id"])

    for k in ["region_feats", "region_boxes", "input_ids", "attn_masks", "labels",
              "ent_input_ids", "ent_attn_masks", "type_input_ids", "type_attn_masks",
              "ctx_input_ids", "ctx_attn_masks", "know_input_ids", "know_attn_masks"]:
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
           "attn_masks": [], "ent_types": [], "img_ids": [], "index": [],
           "ent_input_ids": [], "ent_attn_masks": [],
           "type_input_ids": [], "type_attn_masks": [],
           "ctx_input_ids": [], "ctx_attn_masks": [],
           "know_input_ids": [], "know_attn_masks": []}

    for b in batch:
        out["region_feats"].append(pad_2d(b["region_feats"], max_R))
        out["region_boxes"].append(pad_2d(b["region_boxes"], max_R))
        input_ids_e = torch.zeros(max_E, 128, dtype=torch.long)
        attn_mask_e = torch.zeros(max_E, 128, dtype=torch.long)
        ent_input_ids_e = torch.zeros(max_E, 32, dtype=torch.long)
        ent_attn_mask_e = torch.zeros(max_E, 32, dtype=torch.long)
        type_input_ids_e = torch.zeros(max_E, 16, dtype=torch.long)
        type_attn_mask_e = torch.zeros(max_E, 16, dtype=torch.long)
        ctx_input_ids_e = torch.zeros(max_E, 128, dtype=torch.long)
        ctx_attn_mask_e = torch.zeros(max_E, 128, dtype=torch.long)
        know_input_ids_e = torch.zeros(max_E, 64, dtype=torch.long)
        know_attn_mask_e = torch.zeros(max_E, 64, dtype=torch.long)
        ent_types_e = [""] * max_E
        for i, ent in enumerate(b["entities"]):
            input_ids_e[i] = ent["input_ids"]
            attn_mask_e[i] = ent["attn_mask"]
            ent_input_ids_e[i] = ent["ent_input_ids"]
            ent_attn_mask_e[i] = ent["ent_attn_mask"]
            type_input_ids_e[i] = ent["type_input_ids"]
            type_attn_mask_e[i] = ent["type_attn_mask"]
            ctx_input_ids_e[i] = ent["ctx_input_ids"]
            ctx_attn_mask_e[i] = ent["ctx_attn_mask"]
            know_input_ids_e[i] = ent["know_input_ids"]
            know_attn_mask_e[i] = ent["know_attn_mask"]
            ent_types_e[i] = ent["ent_type"]
        out["input_ids"].append(input_ids_e)
        out["attn_masks"].append(attn_mask_e)
        out["ent_input_ids"].append(ent_input_ids_e)
        out["ent_attn_masks"].append(ent_attn_mask_e)
        out["type_input_ids"].append(type_input_ids_e)
        out["type_attn_masks"].append(type_attn_mask_e)
        out["ctx_input_ids"].append(ctx_input_ids_e)
        out["ctx_attn_masks"].append(ctx_attn_mask_e)
        out["know_input_ids"].append(know_input_ids_e)
        out["know_attn_masks"].append(know_attn_mask_e)
        out["ent_types"].append(ent_types_e)
        out["img_ids"].append(b["img_id"])
        out["index"].append(b["index"])   

    for k in ["region_feats", "region_boxes", "input_ids", "attn_masks",
              "ent_input_ids", "ent_attn_masks", "type_input_ids", "type_attn_masks",
              "ctx_input_ids", "ctx_attn_masks", "know_input_ids", "know_attn_masks"]:
        out[k] = torch.stack(out[k])
    out["index"] = torch.tensor(out["index"], dtype=torch.long)  
    return out