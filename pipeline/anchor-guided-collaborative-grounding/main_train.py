from src.dataset import GroundingDataset, InferenceDataset
from src.collator import collate_fn, collate_fn_infer
from src.model import GroundingModel, train_one_epoch
from src.evaluate import evaluate_dev, inference_and_save
from config import train_json, dev_json, test_json, npz_dir, img_dir
import torch
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_model.pt"

def main():

    train_ds = GroundingDataset(train_json, npz_dir, img_dir)
    dev_ds   = GroundingDataset(dev_json, npz_dir, img_dir)   
    test_ds  = GroundingDataset(test_json, npz_dir, img_dir) 

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dev_loader   = DataLoader(dev_ds, batch_size=4, shuffle=False, collate_fn=collate_fn_infer, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn_infer, num_workers=0)

    model = GroundingModel(temperature=0.8).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    best_f1, best_state = 0.0, None
    

    for epoch in range(6):
        loss = train_one_epoch(model, DEVICE, train_loader, opt)

        p, r, f1 = evaluate_dev(model, DEVICE, dev_loader, iou_thresh=0.5)
        print(f"[Epoch {epoch}] Loss={loss:.4f} | Dev P={p:.3f} R={r:.3f} F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            torch.save(best_state, SAVE_PATH)
            print(f"  ✅ Best model saved at {SAVE_PATH}")

    if best_state:
        model.load_state_dict(best_state)

    p, r, f1 = evaluate_dev(model, test_loader, iou_thresh=0.5)
    print(f"[TEST] P={p:.3f} R={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()

