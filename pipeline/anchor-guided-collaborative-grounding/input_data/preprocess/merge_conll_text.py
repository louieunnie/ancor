import json
import argparse


# -------------------------------
# Task-specific CONLL parsers
# -------------------------------

def parse_conll_fmnerg(conll_file):
    """
    fmnerg:
    IMGID:<id>
    token \t coarse \t fine
    """
    id2text = {}

    with open(conll_file, "r", encoding="utf-8") as f:
        current_id = None
        tokens = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("IMGID:"):
                if current_id and tokens:
                    id2text[current_id] = " ".join(tokens)
                current_id = line.split("IMGID:")[1].strip()
                tokens = []
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                print(f"⚠️ malformed fmnerg line skipped: {line}")
                continue

            tokens.append(parts[0])

        if current_id and tokens:
            id2text[current_id] = " ".join(tokens)

    return id2text


def parse_conll_gmner(conll_file):
    """
    gmner:
    IMGID:<id>
    token \t tag
    """
    id2text = {}

    with open(conll_file, "r", encoding="utf-8") as f:
        current_id = None
        tokens = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("IMGID:"):
                if current_id and tokens:
                    id2text[current_id] = " ".join(tokens)
                current_id = line.split(":")[1].strip()
                tokens = []
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                print(f"⚠️ malformed gmner line skipped: {line}")
                continue

            tokens.append(parts[0])

        if current_id and tokens:
            id2text[current_id] = " ".join(tokens)

    return id2text


# -------------------------------
# Common merge logic
# -------------------------------

def merge_text(json_file, id2text, output_file):
    with open(json_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    for sample in samples:
        imgid = sample["image"].split(".jpg")[0]
        if imgid in id2text:
            sample["text"] = id2text[imgid]
        else:
            sample["text"] = ""
            print(f"❌ text not found for image: {sample['image']}")

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# -------------------------------
# Entry point
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fmnerg", "gmner"], required=True)
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--conll_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    if args.task == "fmnerg":
        id2text = parse_conll_fmnerg(args.conll_file)
    elif args.task == "gmner":
        id2text = parse_conll_gmner(args.conll_file)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    merge_text(args.json_file, id2text, args.output_file)
    print(f"✅ Done. Saved to {args.output_file}")


if __name__ == "__main__":
    main()
