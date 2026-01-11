from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from PIL import Image
import json
from torchvision import transforms
from transformers import RobertaTokenizer

TOKENIZER = RobertaTokenizer.from_pretrained("roberta-large")
IMG_TFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
REGION_IOU_POS_THRESH = 0.5

class GroundingDataset(Dataset):
    def __init__(self, jsonl_path, npz_dir, img_dir,
                 knowledge_key="visual_knowledge", transform=IMG_TFORM):
        self.samples = []
        self.npz_dir = npz_dir
        self.img_dir = img_dir
        self.transform = transform
        self.knowledge_key = knowledge_key

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ents = []
                for ent in d["entities"]:
                    if ent.get("bboxes") is None or len(ent["bboxes"]) == 0:
                        continue
                    ents.append({
                        "text": ent["text"],
                        "type": ent["type"],
                        "know": ent.get(self.knowledge_key, ""),
                        "bboxes": ent["bboxes"]
                    })
                if ents:
                    self.samples.append({"img": d["image"], "text": d["text"], "entities": ents})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, item["img"])).convert("RGB")
        img_tensor = self.transform(img)

        npz = np.load(os.path.join(self.npz_dir, item["img"] + ".npz"))
        all_boxes = npz["bounding_boxes"].astype(np.float32)
        all_feats = torch.tensor(npz["box_features"], dtype=torch.float32)

        ents_out, labels_out = [], []
        for ent in item["entities"]:
            prompt = f"Entity: {ent['text']} ({ent['type']}). Context: {item['text']}"
            # prompt = f"Entity: {ent['text']} ({ent['type']})."
            if ent["know"]:
                prompt += f" Knowledge: {ent['know']}"
            toks = TOKENIZER(prompt, padding="max_length", truncation=True,
                             max_length=128, return_tensors="pt")
            ents_out.append({
                "input_ids": toks["input_ids"].squeeze(0),
                "attn_mask": toks["attention_mask"].squeeze(0),
                "ent_type": ent["type"]
            })
            labels = np.zeros(len(all_boxes), dtype=np.float32)
            for (x1g, y1g, x2g, y2g) in ent["bboxes"]:
                for i, (x1, y1, x2, y2) in enumerate(all_boxes):
                    xx1, yy1 = max(x1, x1g), max(y1, y1g)
                    xx2, yy2 = min(x2, x2g), min(y2, y2g)
                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    union = (x2-x1)*(y2-y1) + (x2g-x1g)*(y2g-y1g) - inter
                    if union > 0 and inter/union >= REGION_IOU_POS_THRESH:
                        labels[i] = 1.0
            labels_out.append(torch.tensor(labels, dtype=torch.float32))

        return {
            "img_id": item["img"],
            "region_feats": all_feats,
            "region_boxes": torch.tensor(all_boxes, dtype=torch.float32),
            "entities": ents_out,
            "labels": labels_out,
            "index": idx 
        }
class InferenceDataset(Dataset):
    def __init__(self, jsonl_path, npz_dir, img_dir,
                 knowledge_key="visual_knowledge", transform=IMG_TFORM):
        self.samples = []
        self.npz_dir = npz_dir
        self.img_dir = img_dir
        self.transform = transform
        self.knowledge_key = knowledge_key

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ents = []
                for ent in d["entities"]:
                    ents.append({
                        "text": ent["text"],
                        "type": ent["type"],
                        "know": ent.get(self.knowledge_key, ""),
                        "bboxes": ent.get("bboxes", []),
                        "pred_label": ent.get("pred_label", 1)
                    })
                if ents:
                    self.samples.append({"img": d["image"], "text": d["text"], "entities": ents})


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, item["img"])).convert("RGB")
        img_tensor = self.transform(img)

        npz = np.load(os.path.join(self.npz_dir, item["img"] + ".npz"))
        all_boxes = npz["bounding_boxes"].astype(np.float32)
        all_feats = torch.tensor(npz["box_features"], dtype=torch.float32)

        ents_out = []
        for ent in item["entities"]:
            prompt = f"Entity: {ent['text']} ({ent['type']}). Context: {item['text']}"
            if ent["know"]:
                prompt += f" Knowledge: {ent['know']}"
            toks = TOKENIZER(prompt, padding="max_length", truncation=True,
                             max_length=128, return_tensors="pt")
            ents_out.append({
                "input_ids": toks["input_ids"].squeeze(0),
                "attn_mask": toks["attention_mask"].squeeze(0),
                "ent_type": ent["type"]
            })

        return {
            "img_id": item["img"],
            "region_feats": all_feats,
            "region_boxes": torch.tensor(all_boxes, dtype=torch.float32),
            "entities": ents_out,
            "index": idx  
        }