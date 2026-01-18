import json

def bio_to_jsonl(bio_path, jsonl_path):
    with open(bio_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []
    image_id = None
    text_tokens = []
    entities = []
    offset = 0
    current_entity = None

    def flush_example(image_id, text_tokens, entities, offset, current_entity):
        if image_id is None:
            return image_id, text_tokens, entities, offset, current_entity
        if current_entity:
            entities.append(current_entity)
            current_entity = None
        text = " ".join(text_tokens)
        data.append({
            "image": image_id.replace("IMGID:", "").strip(),
            "text": text,
            "entities": entities
        })
        return None, [], [], 0, None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("IMGID:"):
            image_id, text_tokens, entities, offset, current_entity = flush_example(
                image_id, text_tokens, entities, offset, current_entity
            )
            image_id = line
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        token = parts[0]
        pred_tag = parts[2]  # predicted BIO tag
        text_tokens.append(token)

        if pred_tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            start = offset
            end = offset + len(token)
            current_entity = {"start": start, "end": end, "type": pred_tag[2:], "text": token}
            # current_entity = {"start": start, "end": end, "type": pred_tag[2:], "inferred_fine_type": pred_tag[2:],"text": token}
        elif pred_tag.startswith("I-") and current_entity:
            current_entity["end"] = offset + len(token)
            current_entity["text"] += " " + token
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

        offset += len(token) + 1  # 공백 포함

    image_id, text_tokens, entities, offset, current_entity = flush_example(
        image_id, text_tokens, entities, offset, current_entity
    )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


bio_to_jsonl("mner_output.bio", "mner_input.jsonl")
