import json
from collections import defaultdict
import argaparse

def parse_file(input_path, output_path):
    data = defaultdict(list)
    cnt = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                image_part = line.split("###")[0]
                image = image_part.split(":")[1].split("#")[0]
                entity_type_part = line.split("###")[1]

                if "(" in entity_type_part and ")" in entity_type_part:
                    text = entity_type_part[:entity_type_part.rfind("(")]
                    type_ = entity_type_part[entity_type_part.rfind("(")+1 : entity_type_part.rfind(")")]
                else:
                    text, type_ = entity_type_part, ""

                fine_type = line.split("###")[-2]
                if " A(n) " in fine_type:
                    fine_type = fine_type.split(" A(n) ")[1]

                entail_pred = line.split("###")[-1]
                if entail_pred == "2":
                    pred_label = "1"
                else:
                    pred_label = entail_pred 

                data[image].append({
                    "text": text,
                    "type": type_,
                    "inferred_fine_type": fine_type,
                    "pred_label": pred_label
                })
                cnt += 1

            except Exception as e:
                print(f"!!! Parse failed: {line}, error: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for image, entities in data.items():
            obj = {"image": image, "entities": entities}
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

    print(f"✅ Conversion completed! Results saved to {output_path}.")
    print(f"Total {cnt} entities saved.")

if __name__ == "__main__":
# parse_file("OFAVE_to_OFAREC_fmnerg_qwen257b_pred7265.txt", 
#            "OFAVE_to_OFAREC_fmnerg_qwen257b_pred.jsonl")
    parser = argparse.ArgumentParser(description="Parse RiVEG-VG preprocessed TXT file to JSONL")
    parser.add_argument("--input_txt", type=str, required=True, help="Input file path")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output file path")
    args = parser.parse_args()

    parse_file(args.input_txt, args.output_jsonl)