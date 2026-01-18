import torch
from torch.utils.data import DataLoader
from src.collator import collate_fn_infer
from src.dataset import InferenceDataset
from src.model import GroundingModel
from src.evaluate import inference_and_save
from config import npz_dir, img_dir, model_path, save_path
# from grounding7 import InferenceDataset, collate_fn_infer, GroundingModel, inference_and_save

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference():
    test_json = "path_to_test_input_with_visual_knowledge.jsonl"

    test_ds = InferenceDataset(test_json, npz_dir, img_dir)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             collate_fn=collate_fn_infer, num_workers=0)

    model = GroundingModel(temperature=0.8).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded model from {model_path}")

    inference_and_save(model, DEVICE, test_loader, save_path=save_path)
    print(f"[INFO] Inference complete. Results saved to {save_path}")


if __name__ == "__main__":
    run_inference()
