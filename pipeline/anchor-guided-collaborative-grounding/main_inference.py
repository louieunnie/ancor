import torch
from torch.utils.data import DataLoader
from collator import collate_fn_infer
from dataset import InferenceDataset
from model import GroundingModel
from evaluate import inference_and_save
from config import npz_dir, img_dir, model_path, save_path
# from grounding7 import InferenceDataset, collate_fn_infer, GroundingModel, inference_and_save

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference():

    # test_json = "/home/minjik9/noname3/grounding/entailment/entailment0917/ver4/test10_2_preds_4-4-4.jsonl"
    # test_json = "/home/minjik9/RiVEG/data_processing/VG_processing/merged_thisthis!!!.jsonl"
    # test_json = "/home/minjik9/RiVEG/data_processing/VG_processing/my_pred_0930_/withoutA!.jsonl"
    # test_json= "/home/minjik9/RiVEG/data_processing/VG_processing/OFAVE_to_OFAREC_fmnerg_1_test_2025-11-10_13-02-09_with_knowledge.jsonl"
    test_json = "/home/minjik9/RiVEG/data_processing/VG_processing/OFAVE_to_OFAREC_fmnerg_qwen257b_pred_with_text.jsonl"
    # test_json = "/home/minjik9/noname3/grounding/results/2_visual_entity_knowledge/0922/output_test_pred_label.jsonl"


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
