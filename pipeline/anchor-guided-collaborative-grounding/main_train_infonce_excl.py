from src.dataset import GroundingDataset, InferenceDataset
from src.collator import collate_fn, collate_fn_infer
from src.model_infonce_excl import GroundingModel, train_one_epoch
from src.evaluate import evaluate_dev, inference_and_save
from config import train_json, dev_json, test_json, npz_dir, img_dir
import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_model.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    def format_bucket_acc(bucket_acc):
        if not bucket_acc:
            return "none"
        parts = []
        for k in sorted(bucket_acc.keys()):
            s = bucket_acc[k]
            parts.append(f"E={k}:acc={s['acc']:.3f}({s['correct']}/{s['total']},n={s['samples']})")
        return " | ".join(parts)

    # Enforce NPZ region features only (no XML/CLIP region pipeline).
    region_dim = int(os.getenv("REGION_DIM", "2048"))

    train_ds = GroundingDataset(
        train_json,
        npz_dir,
        img_dir,
    )
    dev_ds = GroundingDataset(
        dev_json,
        npz_dir,
        img_dir,
    )
    test_ds = GroundingDataset(
        test_json,
        npz_dir,
        img_dir,
    )

    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    epochs = int(os.getenv("EPOCHS", "6"))
    lr = float(os.getenv("LR", "5e-6"))
    temperature = float(os.getenv("TEMPERATURE", "0.8"))
    seed = int(os.getenv("SEED", "45"))
    save_best = os.getenv("SAVE_BEST", "0").lower() in {"1", "true", "yes", "y"}
    save_path = os.getenv("SAVE_PATH", SAVE_PATH)
    test_save_path = os.getenv("TEST_SAVE_PATH", "").strip()

    excl_w = float(os.getenv("LOSS_EXCL_W", "0.6"))
    info_nce_w = float(os.getenv("LOSS_INFO_NCE_W", "1.0"))
    info_nce_tau = float(os.getenv("INFO_NCE_TAU", "0.07"))
    info_nce_topk_neg = int(os.getenv("INFO_NCE_TOPK_NEG", "0"))
    excl_margin = float(os.getenv("EXCL_MARGIN", "0.3"))
    excl_mode = os.getenv("EXCL_MODE", "sim_prob").strip().lower()
    excl_variant = os.getenv("EXCL_VARIANT", "pairwise").strip().lower()
    excl_pair_alpha = float(os.getenv("EXCL_PAIR_ALPHA", "0.5"))
    excl_pair_beta = float(os.getenv("EXCL_PAIR_BETA", "0.3"))
    _g_raw = os.getenv("EXCL_PAIR_GAMMA", "").strip()
    excl_pair_gamma = float(_g_raw) if _g_raw else None
    # Region similarity is fixed to IoU.
    excl_pair_region_sim = "iou"
    excl_pair_type_mode = os.getenv("EXCL_PAIR_TYPE_MODE", "neutral").strip().lower()
    excl_pair_type_factor = float(os.getenv("EXCL_PAIR_TYPE_FACTOR", "1.5"))
    excl_pair_name_thr = float(os.getenv("EXCL_PAIR_NAME_THR", "0.5"))
    excl_pair_region_thr = float(os.getenv("EXCL_PAIR_REGION_THR", "0.5"))
    excl_pair_easy_scale = float(os.getenv("EXCL_PAIR_EASY_SCALE", "0.5"))
    excl_pair_hard_scale = float(os.getenv("EXCL_PAIR_HARD_SCALE", "1.5"))

    set_seed(seed)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, generator=dl_generator
    )
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer, num_workers=0)

    try:
        from config import model_name
    except Exception:
        model_name = "roberta-large"

    model = GroundingModel(temperature=temperature, model_name=model_name, region_dim=region_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(
        f"[HP][InfoNCE+Excl] batch={batch_size} epochs={epochs} lr={lr} temp={temperature} "
        f"| seed={seed} save_best={int(save_best)} "
        f"| w(excl={excl_w},info_nce={info_nce_w}) "
        f"| info_nce_tau={info_nce_tau} info_nce_topk_neg={info_nce_topk_neg} "
        f"| excl_margin={excl_margin} excl_mode={excl_mode} excl_variant={excl_variant} "
        f"| excl_pair(alpha={excl_pair_alpha},beta={excl_pair_beta},"
        f"gamma={'explicit:' + str(excl_pair_gamma) if excl_pair_gamma is not None else 'implicit:1-a-b'},"
        f"region_sim={excl_pair_region_sim},"
        f"type_mode={excl_pair_type_mode},type_factor={excl_pair_type_factor},"
        f"name_thr={excl_pair_name_thr},region_thr={excl_pair_region_thr},"
        f"easy_scale={excl_pair_easy_scale},hard_scale={excl_pair_hard_scale})"
    )

    best_f1, best_state = 0.0, None
    for epoch in range(epochs):
        loss_dict = train_one_epoch(
            model,
            DEVICE,
            train_loader,
            opt,
            excl_w=excl_w,
            info_nce_w=info_nce_w,
            info_nce_tau=info_nce_tau,
            info_nce_topk_neg=info_nce_topk_neg,
            excl_margin=excl_margin,
            excl_mode=excl_mode,
            excl_variant=excl_variant,
            excl_pair_alpha=excl_pair_alpha,
            excl_pair_beta=excl_pair_beta,
            excl_pair_gamma=excl_pair_gamma,
            excl_pair_region_sim=excl_pair_region_sim,
            excl_pair_type_mode=excl_pair_type_mode,
            excl_pair_type_factor=excl_pair_type_factor,
            excl_pair_name_thr=excl_pair_name_thr,
            excl_pair_region_thr=excl_pair_region_thr,
            excl_pair_easy_scale=excl_pair_easy_scale,
            excl_pair_hard_scale=excl_pair_hard_scale,
        )
        p, r, f1, dev_bucket_acc = evaluate_dev(model, DEVICE, dev_loader, iou_thresh=0.5)
        print(
            f"[Epoch {epoch}] Loss={loss_dict['total']:.4f} "
            f"(excl={loss_dict['excl']:.4f}, info_nce={loss_dict['info_nce']:.4f}) "
            f"| Dev P={p:.3f} R={r:.3f} F1={f1:.3f}"
        )
        print(f"  [Dev-by-EntCount] {format_bucket_acc(dev_bucket_acc)}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            if save_best:
                torch.save(best_state, save_path)
                print(f"  ✅ Best model saved at {save_path}")
            else:
                print("  ℹ️ Best model updated in-memory (disk save is OFF)")

    if best_state:
        model.load_state_dict(best_state)

    p, r, f1, test_bucket_acc = evaluate_dev(model, DEVICE, test_loader, iou_thresh=0.5)
    print(f"[TEST] P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print(f"[TEST-by-EntCount] {format_bucket_acc(test_bucket_acc)}")
    if test_save_path:
        inference_and_save(model, DEVICE, test_loader, save_path=test_save_path)
        print(f"[TEST] Inference jsonl saved: {test_save_path}")


if __name__ == "__main__":
    main()

