import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.collator import collate_fn_infer
from src.dataset import InferenceDataset
from src.model_infonce_excl import GroundingModel
from src.evaluate import inference_and_save


def _load_checkpoint(model: torch.nn.Module, model_path: str, device: torch.device) -> None:
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state_dict)}")

    # Handle DataParallel checkpoints.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading checkpoint: {len(unexpected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference entrypoint for InfoNCE+Excl model.")
    parser.add_argument("--infer-json", required=True, help="Input jsonl path for inference.")
    parser.add_argument("--npz-dir", required=True, help="Directory containing region npz files.")
    parser.add_argument("--img-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--model-path", required=True, help="Trained model checkpoint path.")
    parser.add_argument("--save-path", required=True, help="Output jsonl path.")
    parser.add_argument(
        "--save-all-entities-path",
        default="",
        help="Optional additional output jsonl. Includes pred_label for all entities.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--model-name", default="roberta-large", help="Backbone model name.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Model temperature.")
    parser.add_argument("--region-dim", type=int, default=2048, help="Region feature dimension.")
    parser.add_argument(
        "--joint-infer-mode",
        default="none",
        choices=["none", "greedy_conflict"],
        help="Joint decoding mode used in inference_and_save.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.infer_json):
        raise FileNotFoundError(f"infer json not found: {args.infer_json}")
    if not os.path.isdir(args.npz_dir):
        raise FileNotFoundError(f"npz dir not found: {args.npz_dir}")
    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"img dir not found: {args.img_dir}")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"model checkpoint not found: {args.model_path}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.environ["JOINT_INFER_MODE"] = args.joint_infer_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = InferenceDataset(args.infer_json, args.npz_dir, args.img_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_infer,
        num_workers=0,
    )

    model = GroundingModel(
        temperature=args.temperature,
        model_name=args.model_name,
        region_dim=args.region_dim,
    ).to(device)
    _load_checkpoint(model, args.model_path, device)
    model.eval()

    print(f"[INFO] Loaded model: {args.model_path}")
    print(f"[INFO] Inference json: {args.infer_json}")
    print(f"[INFO] Output path: {args.save_path}")
    if args.save_all_entities_path:
        print(f"[INFO] Output all-entities path: {args.save_all_entities_path}")
    print(f"[INFO] joint_infer_mode={args.joint_infer_mode}")

    save_all_path = args.save_all_entities_path.strip() or None
    if save_all_path is None and args.save_path.endswith(".jsonl"):
        save_all_path = args.save_path[:-6] + "_all_entities.jsonl"
    elif save_all_path is None:
        save_all_path = args.save_path + "_all_entities.jsonl"

    inference_and_save(
        model,
        device,
        loader,
        save_path=args.save_path,
        save_all_entities_path=save_all_path,
    )
    print("[INFO] Inference done.")


if __name__ == "__main__":
    main()
