import os
import json
import xml.etree.ElementTree as ET

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_gold_boxes(xml_path, entity_text):
    """load bounding boxes for a specific entity from an XML file"""
    if not os.path.exists(xml_path):
        return None
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall(".//object"):
        name_node = obj.find("name")
        if name_node is None:
            continue
        if name_node.text.strip() == entity_text: 
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                boxes.append((xmin, ymin, xmax, ymax))
    return boxes if boxes else None

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1 = max(x1, box2[0]), max(y1, box2[1])
    xx2, yy2 = min(x2, box2[2]), min(y2, box2[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def evaluate(pred_file, gold_file, xml_dir, iou_thresh=0.5):
    preds = load_jsonl(pred_file)
    golds = load_jsonl(gold_file)

    pred_dict = {}
    for p in preds:
        img_id = p.get("image") or p.get("img_id")
        pred_dict[img_id] = p["entities"]

    metrics = {
        "GMNER": {"TP": 0, "FP": 0, "FN": 0},
        "MNER": {"TP": 0, "FP": 0, "FN": 0},
        "SpanRegion": {"TP": 0, "FP": 0, "FN": 0},
    }
    error_types = {"Null2Box": 0, "Box2Null": 0, "WrongBox": 0, "TypeError": 0}

    gold_count = 0

    # ---------- GOLD LOOP ----------
    for g in golds:
        img_id = g["image"]
        gold_entities = g["entities"]
        pred_entities = pred_dict.get(img_id, [])

        gold_count += len(gold_entities)

        xml_path = os.path.join(xml_dir, img_id.replace(".jpg", "") + ".xml")
        has_xml = os.path.exists(xml_path)

        for gold_ent in gold_entities:
            g_text = gold_ent["text"]
            g_type = gold_ent["type"]

            # load gold boxes from xml
            g_boxes = load_gold_boxes(xml_path, g_text) if has_xml else None

            # match pred by entity+type
            matched_pred = None
            for p in pred_entities:
                if p.get("entity") == g_text and p.get("ent_type") == g_type:
                    matched_pred = p
                    break

            # ----------------- MNER -----------------
            if matched_pred:
                metrics["MNER"]["TP"] += 1
            else:
                metrics["MNER"]["FN"] += 1
                error_types["TypeError"] += 1

            # ----------------- SpanRegion -----------------
            span_match = [p for p in pred_entities if p.get("entity") == g_text]
            if g_boxes is None:  # NONE
                if span_match and span_match[0].get("pred_box") is None:
                    metrics["SpanRegion"]["TP"] += 1
                elif span_match and span_match[0].get("pred_box") is not None:
                    metrics["SpanRegion"]["FP"] += 1
                    error_types["Null2Box"] += 1
                else:
                    metrics["SpanRegion"]["FN"] += 1
            else:  # gold_box
                if not span_match:
                    metrics["SpanRegion"]["FN"] += 1
                    error_types["Box2Null"] += 1
                else:
                    pred = span_match[0]
                    if pred.get("pred_box") is None:
                        metrics["SpanRegion"]["FN"] += 1
                        error_types["Box2Null"] += 1
                    else:
                        ious = [compute_iou(pred["pred_box"], gb) for gb in g_boxes]
                        if max(ious) >= iou_thresh:
                            metrics["SpanRegion"]["TP"] += 1
                        else:
                            metrics["SpanRegion"]["FP"] += 1
                            error_types["WrongBox"] += 1

            # ----------------- GMNER -----------------
            if g_boxes is None:  # gold region = NONE
                if matched_pred and matched_pred.get("pred_box") is None:
                    metrics["GMNER"]["TP"] += 1
                elif matched_pred and matched_pred.get("pred_box") is not None:
                    metrics["GMNER"]["FP"] += 1
                    error_types["Null2Box"] += 1
                else:
                    metrics["GMNER"]["FN"] += 1
            else: # gold box exists
                if matched_pred and matched_pred.get("pred_box") is not None:
                    ious = [compute_iou(matched_pred["pred_box"], gb) for gb in g_boxes]
                    if max(ious) >= iou_thresh:
                        metrics["GMNER"]["TP"] += 1
                    else:
                        metrics["GMNER"]["FP"] += 1
                        metrics["GMNER"]["FN"] += 1
                        error_types["WrongBox"] += 1
                else:
                    metrics["GMNER"]["FN"] += 1
                    if matched_pred:  # entity/type match but box is missing
                        error_types["Box2Null"] += 1
                    else:  # no matching entity+type at all
                        error_types["TypeError"] += 1

    # ---------- PRED LOOP: False Positive ----------
    for img_id, pred_entities in pred_dict.items():
        gold_entities = {(g["text"], g["type"]) for g in next((gg["entities"] for gg in golds if gg["image"] == img_id), [])}
        for p in pred_entities:
            key = (p.get("entity"), p.get("ent_type"))
            if key not in gold_entities:
                metrics["MNER"]["FP"] += 1
                metrics["SpanRegion"]["FP"] += 1
                metrics["GMNER"]["FP"] += 1
                error_types["TypeError"] += 1

    # ---------- Score Calculation ----------
    def calc_scores(res):
        TP, FP, FN = res["TP"], res["FP"], res["FN"]
        prec = TP / (TP + FP) if TP + FP > 0 else 0
        rec = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        return {"precision": prec, "recall": rec, "f1": f1, **res}

    results = {k: calc_scores(v) for k, v in metrics.items()}
    results["ErrorBreakdown"] = error_types
    results["GoldEntities"] = gold_count
    return results

if __name__ == "__main__":
    result = evaluate(
        "runs/inference_infonce_excl_2026-03-14_12-28-13_8416_result.jsonl",
        "input_data/vk_v2_gpt41mini/test.jsonl",
        "path_to_xml_directory",
    )
    print(json.dumps(result, indent=2))
