import os
import json
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "All_Data_with_Reassigned_Topic_with_QuestionType.jsonl")
OUTDIR = os.path.join(HERE, "Normalize")
OUTPUT = os.path.join(OUTDIR, "All_Data_with_Reassigned_Topic_with_QuestionType__normalized_question_types.jsonl")
SUMMARY = os.path.join(OUTDIR, "Normalize_Summary.txt")

def ensure_outdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# Exact and pattern-based mappings for Intent values
INTENT_MAP = {
    "INTENT_1a. Fact Lookup — Request for specific facts/data.": "INTENT_1a. Fact Lookup",
    "INTENT_2b. Data Analysis / Calculation — Numerical computation, statistical analysis, dataset interpretation.": "INTENT_2b. Data Analysis / Calculation",
    "INTENT_4b. Rewrite — Change style, tone, or rephrase while keeping language": "INTENT_4b. Rewrite",
    "INTENT_4z. Others (Other transformation/processing)": "INTENT_4z. Others",
    "INTENT_8c. Emotional Support": "INTENT_8c. Emotional Support / Empathy",
    "INTENT_8z. Others (Other social/engagement)": "INTENT_8z. Others",
    "INTENT_8z. Others (Social/Engagement request for interview/collaboration)": "INTENT_8z. Others",
    "INTENT_8z. Others (Social/Engagement)": "INTENT_8z. Others",
    "INTENT_8z. Others (Social/Engagement: persuasive/social post)": "INTENT_8z. Others",
    "INTENT_8z. Others (Social/Engagement: persuasive/social post)的改为": "INTENT_8z. Others",
}

# Fallback regexes (robust to stray descriptions after an em dash or parentheses)
INTENT_PATTERNS = [
    (re.compile(r'^INTENT_1a\. Fact Lookup\s*—.*$'), "INTENT_1a. Fact Lookup"),
    (re.compile(r'^INTENT_2b\. Data Analysis / Calculation\s*—.*$'), "INTENT_2b. Data Analysis / Calculation"),
    (re.compile(r'^INTENT_4b\. Rewrite\s*—.*$'), "INTENT_4b. Rewrite"),
    (re.compile(r'^INTENT_4z\. Others\s*\(.*\)$'), "INTENT_4z. Others"),
    (re.compile(r'^INTENT_8c\. Emotional Support$'), "INTENT_8c. Emotional Support / Empathy"),
    (re.compile(r'^INTENT_8z\. Others\s*\(.*\)$'), "INTENT_8z. Others"),
]

# Exact and pattern-based mappings for Form values
FORM_MAP = {
    "FORM_1a. Concise Value(s) / Entity(ies) — Direct factual output, minimal context": "FORM_1a. Concise Value(s) / Entity(ies)",
    "FORM_1a. Concise Value(s)": "FORM_1a. Concise Value(s) / Entity(ies)",
    "FORM_2b. Detailed Multi-paragraph — Multiple paragraphs with depth/examples.": "FORM_2b. Detailed Multi-paragraph",
    "FORM_3a Item List": "FORM_3a. Item List",
    "FORM_3d Procedural Steps": "FORM_3d. Procedural Steps",
    "FORM_4b. JSON — Key-value structured output.": "FORM_4b. JSON",
    "FORM_6a. Semantic Document Markup — HTML/Markdown document bodies, headings, sections.": "FORM_6a. Semantic Document Markup",
    "FORM_7b. Yes/No / True/False — Binary choice": "FORM_7b. Yes/No / True/False",
}

FORM_PATTERNS = [
    (re.compile(r'^FORM_1a\. Concise Value\(s\) / Entity\(ies\)\s*—.*$'), "FORM_1a. Concise Value(s) / Entity(ies)"),
    (re.compile(r'^FORM_2b\. Detailed Multi-paragraph\s*—.*$'), "FORM_2b. Detailed Multi-paragraph"),
    (re.compile(r'^FORM_3a\s+Item List$'), "FORM_3a. Item List"),
    (re.compile(r'^FORM_3d\s+Procedural Steps$'), "FORM_3d. Procedural Steps"),
    (re.compile(r'^FORM_4b\. JSON\s*—.*$'), "FORM_4b. JSON"),
    (re.compile(r'^FORM_6a\. Semantic Document Markup\s*—.*$'), "FORM_6a. Semantic Document Markup"),
    (re.compile(r'^FORM_7b\. Yes/No / True/False\s*—.*$'), "FORM_7b. Yes/No / True/False"),
]

def normalize_value(val, exact_map, patterns, counter):
    if not isinstance(val, str):
        return val
    if val in exact_map:
        newv = exact_map[val]
        if newv != val:
            counter[(val, newv)] += 1
        return newv
    for regex, replacement in patterns:
        if regex.match(val):
            if replacement != val:
                counter[(val, replacement)] += 1
            return replacement
    return val

def normalize_row(row, intent_changes, form_changes):
    fqt = row.get("Final_Question_Types")
    if not isinstance(fqt, dict):
        return row

    intents = fqt.get("Intent")
    if isinstance(intents, list):
        fqt["Intent"] = [normalize_value(v, INTENT_MAP, INTENT_PATTERNS, intent_changes) for v in intents]
    elif isinstance(intents, str):
        fqt["Intent"] = normalize_value(intents, INTENT_MAP, INTENT_PATTERNS, intent_changes)

    forms = fqt.get("Form")
    if isinstance(forms, list):
        fqt["Form"] = [normalize_value(v, FORM_MAP, FORM_PATTERNS, form_changes) for v in forms]
    elif isinstance(forms, str):
        fqt["Form"] = normalize_value(forms, FORM_MAP, FORM_PATTERNS, form_changes)

    row["Final_Question_Types"] = fqt
    return row

def main():
    if not os.path.isfile(INPUT):
        raise SystemExit(f"Input file not found: {INPUT}")

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR, exist_ok=True)

    from collections import Counter
    intent_changes = Counter()
    form_changes = Counter()

    total = 0
    kept = []

    with open(INPUT, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON on line {line_no}: {e}")
                continue
            total += 1
            obj = normalize_row(obj, intent_changes, form_changes)
            kept.append(obj)

    with open(OUTPUT, "w", encoding="utf-8") as fout:
        for obj in kept:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(SUMMARY, "w", encoding="utf-8") as fsum:
        fsum.write(f"Total rows processed: {total}\n")
        fsum.write("\nIntent replacements (original -> new : count):\n")
        if intent_changes:
            for (orig, newv), cnt in sorted(intent_changes.items(), key=lambda kv: (-kv[1], kv[0][0])):
                fsum.write(f"- {orig} -> {newv} : {cnt}\n")
        else:
            fsum.write("(none)\n")

        fsum.write("\nForm replacements (original -> new : count):\n")
        if form_changes:
            for (orig, newv), cnt in sorted(form_changes.items(), key=lambda kv: (-kv[1], kv[0][0])):
                fsum.write(f"- {orig} -> {newv} : {cnt}\n")
        else:
            fsum.write("(none)\n")

    print("Normalization complete.")
    print("Output JSONL:", OUTPUT)
    print("Summary:", SUMMARY)

if __name__ == "__main__":
    main()
