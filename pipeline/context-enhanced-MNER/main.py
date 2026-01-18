# -*- coding: utf-8 -*-
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, get_scheduler, set_seed
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime

from labels import gmner_label_list, coarse_label_list, fine_label_list, build_label_mappings, build_maps, coarse_fine_tree
from model import Context_EnhancedMNER
from data_utils import (
    load_bio_file,
    tokenize_and_align_labels_with_spans,
    ner_collate_fn,
    prepare_region_tensors_from_npz, 
    load_knowledge_map,
    merge_examples_with_knowledge, 
)
from evaluate import evaluate
from config import MODEL_NAME, BATCH_SIZE, EPOCHS, SEED, DEVICE, VINVL_DIR, IMG_DIR, TRAIN_FILE_PATH, DEV_FILE_PATH, TEST_FILE_PATH

set_seed(SEED)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/train_{timestamp}.log"

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)                    
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

class NERDataset(Dataset):
    def __init__(self, examples, tokenizer, coarse_label2id, fine_label2id, coarselabel2typeid, finelabel2typeid, label_set="coarse"):
        self.examples = examples
        self.features_coarse = [
            tokenize_and_align_labels_with_spans(
                ex, tokenizer, coarse_label2id, fine_label2id, label_set="coarse", label2typeid=coarselabel2typeid, 
            )
            for ex in examples
        ]
        self.features_coarse = [feat for feat in self.features_coarse if feat is not None]
        self.features_fine = [
            tokenize_and_align_labels_with_spans(
                ex, tokenizer, fine_label2id, fine_label2id, label_set="fine", label2typeid=finelabel2typeid,
            )
            for ex in examples
        ]
        self.features_fine = [feat for feat in self.features_fine if feat is not None]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.features_coarse)

    def __getitem__(self, idx):
        example = self.examples[idx]
        feat_c = self.features_coarse[idx]
        feat_f = self.features_fine[idx]
        img_path = self.examples[idx]["img_path"]

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        npz_path = os.path.join(VINVL_DIR, f"{base_name}.jpg.npz")
        region_feats, region_boxes, region_mask = prepare_region_tensors_from_npz(
            npz_path, max_regions=20, device='cpu'
        )

        fine_t_idx = [t for (_s, _e, t) in feat_f["gold_spans"]]
        span_fine_type_targets = torch.tensor(fine_t_idx, dtype=torch.long)
        item = item = {
            "input_ids": feat_c["input_ids"],
            "attention_mask": feat_c["attention_mask"],
            "labels": feat_c["labels"],
            "start_labels": feat_c["start_labels"],
            "end_labels": feat_c["end_labels"],
            "gold_spans": feat_c["gold_spans"],
            "region_feats": region_feats,
            "region_boxes": region_boxes,
            "region_mask": region_mask,
            "img_path": img_path,
            "span_fine_type_targets": torch.tensor(fine_t_idx, dtype=torch.long),
            "tokens": feat_c["tokens"],        
            "word_ids": feat_c["word_ids"],    
            "img_id": example["img_id"]        
        }
        if "knowledge_mask" in example:
            item["knowledge_mask"] = torch.tensor(example["knowledge_mask"], dtype=torch.bool)
        if "knowledge" in example:
            item["knowledge_text"] = example["knowledge"]
        else:
            item["knowledge_text"] = ""
        if "raw_text" in feat_c:
            item["raw_texts"] = feat_c["raw_text"]
        if "offset_mapping" in feat_c:
            item["offset_mapping"] = feat_c["offset_mapping"]

        return item

def main():
    scaler = torch.cuda.amp.GradScaler()
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)

    train_data = load_bio_file(TRAIN_FILE_PATH, img_dir=IMG_DIR)
    val_data   = load_bio_file(DEV_FILE_PATH, img_dir=IMG_DIR)
    test_data  = load_bio_file(TEST_FILE_PATH, img_dir=IMG_DIR)

    train_knowledge = load_knowledge_map("./input_data/example_train.jsonl")
    val_knowledge   = load_knowledge_map("./input_data/example_dev.jsonl")
    test_knowledge  = load_knowledge_map("./input_data/example_test.jsonl")

    train_data = merge_examples_with_knowledge(train_data, train_knowledge, tokenizer)
    val_data   = merge_examples_with_knowledge(val_data, val_knowledge, tokenizer)
    test_data  = merge_examples_with_knowledge(test_data, test_knowledge, tokenizer)

    if "fmnerg" in TRAIN_FILE_PATH: 
        coarse_label2id, coarseid2label, coarselabel2typeid, coarsetypeid2label = build_label_mappings(coarse_label_list)
        fine_label2id, fineid2label, finelabel2typeid, finetypeid2label = build_label_mappings(fine_label_list)
        coarse_to_fine, fine_to_coarse = build_maps(coarselabel2typeid, finelabel2typeid, coarse_fine_tree)
    elif "gmner" in TRAIN_FILE_PATH:
        coarse_label2id, coarseid2label, coarselabel2typeid, coarsetypeid2label = build_label_mappings(gmner_label_list)
        fine_label2id, fineid2label, finelabel2typeid, finetypeid2label = build_label_mappings(gmner_label_list)
    else:
        print("dataset name error: ", TRAIN_FILE_PATH)

    train_dataset = NERDataset(train_data, tokenizer, coarse_label2id, fine_label2id, coarselabel2typeid, finelabel2typeid, label_set="coarse")
    val_dataset   = NERDataset(val_data,   tokenizer, coarse_label2id, fine_label2id, coarselabel2typeid, finelabel2typeid, label_set="coarse")
    test_dataset  = NERDataset(test_data,  tokenizer, coarse_label2id, fine_label2id, coarselabel2typeid, finelabel2typeid, label_set="coarse")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: ner_collate_fn(b, tokenizer))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=lambda b: ner_collate_fn(b, tokenizer))
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=lambda b: ner_collate_fn(b, tokenizer))

    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("Caption sample:", batch["caption"][:2])
    print("Knowledge sample:", batch["knowledge_text"][:2])
    logger.info("gmner")
    model = Context_EnhancedMNER(
        num_labels=len(coarse_label2id),
        vis_dim=2048,
        tokenizer=tokenizer,
        model_name="roberta-large",
        use_knowledge=True,
        use_context=False,
        use_span_cls=False,
        use_region_attn=True,
        use_span_injection=False,
        use_span_refine = False,
        # id2label = coarseid2label
    )

    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)
    patience = 7
    best_val_score = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            gold_spans    = batch["gold_spans"]
            S = sum(len(spans) for spans in gold_spans)
            p_off = 0.3
            span_vision_use_labels = torch.zeros(S, dtype=torch.float32, device=DEVICE)
            k = int((1.0 - p_off) * S)
            span_vision_use_labels[:k] = 1.0
            perm = torch.randperm(S, device=DEVICE)
            span_vision_use_labels = span_vision_use_labels[perm]

            with torch.cuda.amp.autocast():   
                out = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    start_labels = batch["start_labels"].to(DEVICE),
                    end_labels = batch["end_labels"].to(DEVICE),
                    region_feats=batch["region_feats"].to(DEVICE),
                    region_boxes=batch["region_boxes"].to(DEVICE),
                    region_mask=batch["region_mask"].to(DEVICE),
                    knowledge_texts=batch["knowledge_text"],
                    token_labels=batch["labels"].to(DEVICE)
                )
                loss = out["loss"]
                if loss.dim() > 0:
                    loss = loss.mean()

            scaler.scale(loss).backward()     # ✅ FP16 safe backward
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        logger.info(f"Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f}")
        torch.cuda.empty_cache()

        val_metrics = evaluate(
            model,
            val_loader,
            tokenizer,
            coarseid2label,
            save_path=f"runs/eval_redecoded_val_{timestamp}",
            redo=True,
            thr_global=0.96,
            per_type_threshold=None,
            topk=3
        )
        val_f1 = val_metrics.get("f1", 0.0)

        if val_f1 > best_val_score:
            best_val_score = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"runs/saved_model/best_model{timestamp}.pt")
            logger.info(f"Validation improved: F1={val_f1:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement. Count={epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break        
        logger.info("===== Epoch End =====")
    best_model_path = f"runs/saved_model/best_model{timestamp}.pt"
    model.load_state_dict(torch.load(best_model_path))
    logger.info("===== Final Evaluation on TEST set =====")
    test_metrics = evaluate(
        model,
        test_loader,
        tokenizer,
        coarseid2label,
        save_path=f"runs/eval_redecoded_test_{timestamp}",
        redo=True,
        thr_global=0.96,
        per_type_threshold=None,
        topk=3
    )
    if isinstance(test_metrics, dict):
        logger.info(f"TEST metrics: {test_metrics}")
        save_dir = "runs/saved_model"
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"model{timestamp}.pt")

if __name__ == "__main__":
    main()
