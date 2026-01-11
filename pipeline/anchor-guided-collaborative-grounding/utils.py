def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1 = max(x1, box2[0]), max(y1, box2[1])
    xx2, yy2 = min(x2, box2[2]), min(y2, box2[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area1 = max(0, x2-x1) * max(0, y2-y1)
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0