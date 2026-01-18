import json
import argparse

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def merge_entity_knowledge(input_file, jsonl_with_knowledge, output_file):
    pred_data = load_jsonl(input_file)  # predictions
    knowledge_data = load_jsonl(jsonl_with_knowledge)  # knowledge-augmented data

    # Build lookup table from jsonl_with_knowledge
    knowledge_entities_map = {}
    for sample in knowledge_data:
        image = sample.get("image")
        for ent in sample.get("entities", []):
            key = (image, ent.get("text"), ent.get("type"))
            knowledge_entities_map[key] = {
                "entity_knowledge": ent.get("entity_knowledge", ""),
                "visual_knowledge": ent.get("visual_knowledge", "")
            }

    # Update entities in input_file
    for sample in pred_data:
        image = sample.get("image").split(".jpg")[0]
        for ent in sample.get("entities", []):
            # Normalize pred_label
            if "pred_label" in ent and isinstance(ent["pred_label"], str):
                try:
                    ent["pred_label"] = int(ent["pred_label"])
                except ValueError:
                    pass

            key = (image, ent.get("text"), ent.get("type"))
            if key in knowledge_entities_map:
                ent.update(knowledge_entities_map[key])
            else:
                print(f"⚠️ no matching entity found: {key}")

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in pred_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ Updated file saved to: {output_file}")



def main():
    parser = argparse.ArgumentParser(
        description="Merge entity-level knowledge from file B into predictions from file A"
    )
    parser.add_argument("--input_file", required=True, help="JSONL file with entity predictions")
    parser.add_argument("--jsonl_with_knowledge", required=True, help="JSONL file with entity knowledge")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")

    args = parser.parse_args()

    merge_entity_knowledge(
        input_file=args.input_file,
        jsonl_with_knowledge=args.jsonl_with_knowledge,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
