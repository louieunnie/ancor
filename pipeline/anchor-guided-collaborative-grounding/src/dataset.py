from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from PIL import Image
import json
from torchvision import transforms
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("MODEL_NAME", "roberta-large"))
IMG_TFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
REGION_IOU_POS_THRESH = 0.5


def _tok(text, max_length):
    return TOKENIZER(
        text if text is not None else "",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _env_flag(name, default="1"):
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _build_query_prompt(ent_text, ent_type, ctx_text, know_text, use_entity, use_type, use_context, use_knowledge):
    chunks = []
    if use_entity:
        if use_type:
            chunks.append(f"Entity: {ent_text} ({ent_type}).")
        else:
            chunks.append(f"Entity: {ent_text}.")
    elif use_type:
        chunks.append(f"Type: {ent_type}.")
    if use_context:
        chunks.append(f"Context: {ctx_text}")
    if use_knowledge and know_text:
        chunks.append(f"Knowledge: {know_text}")
    if not chunks:
        # Keep a safe non-empty prompt for tokenizer/model stability.
        chunks.append(f"Entity: {ent_text}.")
    return " ".join(chunks)


class GroundingDataset(Dataset):
    def __init__(self, jsonl_path, npz_dir, img_dir,
                 knowledge_key="visual_knowledge", transform=IMG_TFORM,
                 use_xml_clip_regions=False, xml_dir=None,
                 use_clip_region_encoder=False,
                 clip_model_name="openai/clip-vit-base-patch32",
                 clip_device="cpu"):
        self.samples = []
        self.npz_dir = npz_dir
        self.img_dir = img_dir
        self.transform = transform
        self.knowledge_key = knowledge_key
        self.use_xml_clip_regions = bool(use_xml_clip_regions)
        self.use_clip_region_encoder = bool(use_clip_region_encoder)
        self.xml_dir = xml_dir
        self.clip_model_name = clip_model_name
        self.clip_device = clip_device
        self._region_cache = {}
        self.prompt_use_entity = _env_flag("PROMPT_USE_ENTITY", "1")
        self.prompt_use_type = _env_flag("PROMPT_USE_TYPE", "1")
        self.prompt_use_context = _env_flag("PROMPT_USE_CONTEXT", "1")
        self.prompt_use_knowledge = _env_flag("PROMPT_USE_KNOWLEDGE", "1")
        # Force NPZ-only region loading regardless of caller options.
        self.use_xml_clip_regions = False
        self.use_clip_region_encoder = False

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

    def _load_xml_boxes(self, img_id):
        xml_path = os.path.join(self.xml_dir, img_id.replace(".jpg", "").replace(".png", "") + ".xml")
        if not os.path.exists(xml_path):
            return []
        root = ET.parse(xml_path).getroot()
        boxes = []
        for obj in root.findall(".//object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                xmin = int(float(bnd.find("xmin").text))
                ymin = int(float(bnd.find("ymin").text))
                xmax = int(float(bnd.find("xmax").text))
                ymax = int(float(bnd.find("ymax").text))
            except Exception:
                continue
            boxes.append((xmin, ymin, xmax, ymax))
        return boxes

    def _clip_encode_boxes(self, pil_img, boxes):
        if not boxes:
            return torch.zeros((0, 768), dtype=torch.float32), np.zeros((0, 4), dtype=np.float32)

        w, h = pil_img.size
        crops = []
        valid_boxes = []
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(pil_img.crop((x1, y1, x2, y2)))
            valid_boxes.append((x1, y1, x2, y2))

        if not crops:
            return torch.zeros((0, 768), dtype=torch.float32), np.zeros((0, 4), dtype=np.float32)

        with torch.no_grad():
            proc = self.clip_processor(images=crops, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(self.clip_device)
            out = self.clip_model(pixel_values=pixel_values)
            feats = out.pooler_output.detach().cpu().float()

        return feats, np.asarray(valid_boxes, dtype=np.float32)

    def _load_regions(self, item, pil_img):
        del pil_img
        img_id = item["img"]
        if img_id in self._region_cache:
            cached_feats, cached_boxes = self._region_cache[img_id]
            return cached_feats.clone(), cached_boxes.copy()
        npz = np.load(os.path.join(self.npz_dir, item["img"] + ".npz"))
        all_boxes = npz["bounding_boxes"].astype(np.float32)
        feats = torch.tensor(npz["box_features"], dtype=torch.float32)

        self._region_cache[img_id] = (feats, all_boxes)
        return feats.clone(), all_boxes.copy()

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, item["img"])).convert("RGB")
        all_feats, all_boxes = self._load_regions(item, img)

        ents_out, labels_out = [], []
        for ent in item["entities"]:
            prompt = _build_query_prompt(
                ent_text=ent["text"],
                ent_type=ent["type"],
                ctx_text=item["text"],
                know_text=ent["know"],
                use_entity=self.prompt_use_entity,
                use_type=self.prompt_use_type,
                use_context=self.prompt_use_context,
                use_knowledge=self.prompt_use_knowledge,
            )
            toks = _tok(prompt, 128)
            tok_ent = _tok(ent["text"], 32)
            tok_type = _tok(ent["type"], 16)
            tok_ctx = _tok(item["text"], 128)
            tok_know = _tok(ent["know"], 64)
            ents_out.append({
                "input_ids": toks["input_ids"].squeeze(0),
                "attn_mask": toks["attention_mask"].squeeze(0),
                "ent_input_ids": tok_ent["input_ids"].squeeze(0),
                "ent_attn_mask": tok_ent["attention_mask"].squeeze(0),
                "type_input_ids": tok_type["input_ids"].squeeze(0),
                "type_attn_mask": tok_type["attention_mask"].squeeze(0),
                "ctx_input_ids": tok_ctx["input_ids"].squeeze(0),
                "ctx_attn_mask": tok_ctx["attention_mask"].squeeze(0),
                "know_input_ids": tok_know["input_ids"].squeeze(0),
                "know_attn_mask": tok_know["attention_mask"].squeeze(0),
                "ent_type": ent["type"],
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
                 knowledge_key="visual_knowledge", transform=IMG_TFORM,
                 use_xml_clip_regions=False, xml_dir=None,
                 use_clip_region_encoder=False,
                 clip_model_name="openai/clip-vit-base-patch32",
                 clip_device="cpu"):
        self.samples = []
        self.npz_dir = npz_dir
        self.img_dir = img_dir
        self.transform = transform
        self.knowledge_key = knowledge_key
        self.use_xml_clip_regions = bool(use_xml_clip_regions)
        self.use_clip_region_encoder = bool(use_clip_region_encoder)
        self.xml_dir = xml_dir
        self.clip_model_name = clip_model_name
        self.clip_device = clip_device
        self._region_cache = {}
        self.prompt_use_entity = _env_flag("PROMPT_USE_ENTITY", "1")
        self.prompt_use_type = _env_flag("PROMPT_USE_TYPE", "1")
        self.prompt_use_context = _env_flag("PROMPT_USE_CONTEXT", "1")
        self.prompt_use_knowledge = _env_flag("PROMPT_USE_KNOWLEDGE", "1")
        # Force NPZ-only region loading regardless of caller options.
        self.use_xml_clip_regions = False
        self.use_clip_region_encoder = False

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

    def _load_xml_boxes(self, img_id):
        xml_path = os.path.join(self.xml_dir, img_id.replace(".jpg", "").replace(".png", "") + ".xml")
        if not os.path.exists(xml_path):
            return []
        root = ET.parse(xml_path).getroot()
        boxes = []
        for obj in root.findall(".//object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                xmin = int(float(bnd.find("xmin").text))
                ymin = int(float(bnd.find("ymin").text))
                xmax = int(float(bnd.find("xmax").text))
                ymax = int(float(bnd.find("ymax").text))
            except Exception:
                continue
            boxes.append((xmin, ymin, xmax, ymax))
        return boxes

    def _clip_encode_boxes(self, pil_img, boxes):
        if not boxes:
            return torch.zeros((0, 768), dtype=torch.float32), np.zeros((0, 4), dtype=np.float32)
        w, h = pil_img.size
        crops = []
        valid_boxes = []
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(pil_img.crop((x1, y1, x2, y2)))
            valid_boxes.append((x1, y1, x2, y2))
        if not crops:
            return torch.zeros((0, 768), dtype=torch.float32), np.zeros((0, 4), dtype=np.float32)
        with torch.no_grad():
            proc = self.clip_processor(images=crops, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(self.clip_device)
            out = self.clip_model(pixel_values=pixel_values)
            feats = out.pooler_output.detach().cpu().float()
        return feats, np.asarray(valid_boxes, dtype=np.float32)

    def _load_regions(self, item, pil_img):
        del pil_img
        img_id = item["img"]
        if img_id in self._region_cache:
            cached_feats, cached_boxes = self._region_cache[img_id]
            return cached_feats.clone(), cached_boxes.copy()
        npz = np.load(os.path.join(self.npz_dir, item["img"] + ".npz"))
        all_boxes = npz["bounding_boxes"].astype(np.float32)
        feats = torch.tensor(npz["box_features"], dtype=torch.float32)

        self._region_cache[img_id] = (feats, all_boxes)
        return feats.clone(), all_boxes.copy()

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, item["img"])).convert("RGB")
        all_feats, all_boxes = self._load_regions(item, img)

        ents_out = []
        for ent in item["entities"]:
            prompt = _build_query_prompt(
                ent_text=ent["text"],
                ent_type=ent["type"],
                ctx_text=item["text"],
                know_text=ent["know"],
                use_entity=self.prompt_use_entity,
                use_type=self.prompt_use_type,
                use_context=self.prompt_use_context,
                use_knowledge=self.prompt_use_knowledge,
            )
            toks = _tok(prompt, 128)
            tok_ent = _tok(ent["text"], 32)
            tok_type = _tok(ent["type"], 16)
            tok_ctx = _tok(item["text"], 128)
            tok_know = _tok(ent["know"], 64)
            ents_out.append({
                "input_ids": toks["input_ids"].squeeze(0),
                "attn_mask": toks["attention_mask"].squeeze(0),
                "ent_input_ids": tok_ent["input_ids"].squeeze(0),
                "ent_attn_mask": tok_ent["attention_mask"].squeeze(0),
                "type_input_ids": tok_type["input_ids"].squeeze(0),
                "type_attn_mask": tok_type["attention_mask"].squeeze(0),
                "ctx_input_ids": tok_ctx["input_ids"].squeeze(0),
                "ctx_attn_mask": tok_ctx["attention_mask"].squeeze(0),
                "know_input_ids": tok_know["input_ids"].squeeze(0),
                "know_attn_mask": tok_know["attention_mask"].squeeze(0),
                "ent_type": ent["type"],
            })

        return {
            "img_id": item["img"],
            "region_feats": all_feats,
            "region_boxes": torch.tensor(all_boxes, dtype=torch.float32),
            "entities": ents_out,
            "index": idx  
        }