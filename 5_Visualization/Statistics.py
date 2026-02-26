import csv
import json
import os
from collections import Counter

# Known dataset prefixes
KNOWN_PREFIXES = [
    "WildChat",
    "LMSYSChat",
    "ClimateQ&A",
    "ClimaQA_Gold",
    "ClimaQA_Silver",
    "Reddit",
    "IPCC_AR6",
    "SciDCC",
    "Climate_FEVER",
    "Climata_FEVER",
    "Environmental_Claims",
    "ClimSight",
]

# Display-name mapping rules
DISPLAY_NAME_MAP = {
    "LMSYSChat": "LMSYS-Chat-1M",
    "ClimaQA_Gold": "ClimaQA-Gold",
    "ClimaQA_Silver": "ClimaQA-Silver",
    "Climate_FEVER": "Climate-FEVER",
    "Climata_FEVER": "Climata-FEVER",
    "Environmental_Claims": "Environmental Claims",
}

def normalize_dataset_name(prefix: str) -> str:
    return DISPLAY_NAME_MAP.get(prefix, prefix)

def find_dataset_prefix(sample_id: str) -> str:
    for pref in sorted(KNOWN_PREFIXES, key=len, reverse=True):
        if sample_id.startswith(pref):
            return pref
    for sep in ["_", "-", " "]:
        if sep in sample_id:
            return sample_id.split(sep)[0]
    return sample_id

def row_has_f1_others(row):
    topics = row.get("Final_Topics", [])
    if isinstance(topics, list):
        return any(str(t).strip() == "F1. Others" for t in topics)
    if isinstance(topics, str):
        return topics.strip() == "F1. Others"
    return False

def ensure_outdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON on line {i}: {e}")
    return rows

def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def analyze(input_path: str, outdir: str):
    rows = read_jsonl(input_path)
    if not rows:
        print(f"No data found in {input_path}")
        return

    totals = Counter()
    with_others = Counter()
    without_others = Counter()

    unique_topics = Counter()
    unique_intents = Counter()
    unique_forms = Counter()

    kept_rows = []

    for row in rows:
        sample_id = str(row.get("id", ""))
        prefix = find_dataset_prefix(sample_id)
        display_name = normalize_dataset_name(prefix)

        has_others = row_has_f1_others(row)
        totals[display_name] += 1
        if has_others:
            with_others[display_name] += 1
        else:
            without_others[display_name] += 1
            kept_rows.append(row)

        topics = row.get("Final_Topics", [])
        if isinstance(topics, list):
            for t in topics:
                if t:
                    unique_topics[str(t).strip()] += 1
        elif isinstance(topics, str) and topics.strip():
            unique_topics[topics.strip()] += 1

        fq = row.get("Final_Question_Types", {})
        if isinstance(fq, dict):
            intents = fq.get("Intent", [])
            forms = fq.get("Form", [])
            if isinstance(intents, list):
                for i in intents:
                    if i:
                        unique_intents[str(i).strip()] += 1
            elif isinstance(intents, str) and intents.strip():
                unique_intents[intents.strip()] += 1

            if isinstance(forms, list):
                for fm in forms:
                    if fm:
                        unique_forms[str(fm).strip()] += 1
            elif isinstance(forms, str) and forms.strip():
                unique_forms[forms.strip()] += 1

    base = os.path.splitext(os.path.basename(input_path))[0]
    ensure_outdir(outdir)

    counts_csv = os.path.join(outdir, f"{base}__dataset_counts.csv")
    filtered_jsonl = os.path.join(outdir, f"{base}__no_F1_Others.jsonl")
    topics_txt = os.path.join(outdir, f"{base}__unique_topics.txt")
    intents_txt = os.path.join(outdir, f"{base}__unique_intents.txt")
    forms_txt = os.path.join(outdir, f"{base}__unique_forms.txt")

    datasets = sorted(totals.keys())
    with open(counts_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Total", "With_F1.Others", "Without_F1.Others"])
        for ds in datasets:
            writer.writerow([ds, totals[ds], with_others.get(ds, 0), without_others.get(ds, 0)])
        writer.writerow([])
        writer.writerow(["ALL", sum(totals.values()), sum(with_others.values()), sum(without_others.values())])

    write_jsonl(filtered_jsonl, kept_rows)

    def write_counter_values(path, counter):
        with open(path, "w", encoding="utf-8") as f:
            for k in sorted(counter.keys()):
                f.write(f"{k}\n")

    write_counter_values(topics_txt, unique_topics)
    write_counter_values(intents_txt, unique_intents)
    write_counter_values(forms_txt, unique_forms)

    print("Outputs written to", outdir)

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(here, "All_Data_with_Reassigned_Topic_with_QuestionType.jsonl")
    outdir = os.path.join(here, "utputs")
    analyze(input_path, outdir)
