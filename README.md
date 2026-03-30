# LLM Benchmark-User Need Misalignment for Climate Change 

This repository contains the official code for the paper:

> **LLM Benchmark-User Need Misalignment for Climate Change**

> 📄 Paper link: https://arxiv.org/abs/2603.26106

> 🤗 HuggingFace Dataset & Resources: https://huggingface.co/datasets/Westing/LLM-Misalign-Climate-Change

---

## 📌 Overview

This project investigates the Misalignment between Benchmarks and User Needs for Climate Change.

The repository provides:

* Data extraction and preprocessing pipelines
* Multi-source dataset unification
* Topic modeling (LLM-driven + iterative merging)
* Question type (user intent + expected answer form) modeling
* Simple statistical analysis
* Interactive web-based visualization tool

---

# 📂 Repository Structure


## 0️⃣ `0_Firstturn_Deduplicated_Conversations_Extraction`
Basic preprocessing of human–LLM conversation datasets.
* **`Extract_Firsturn_Dedup.py`**
  * Extracts the **first-turn conversations** from:
    * WildChat
    * LMSYS-Chat-1M
  * Removes:
    * Conversations with empty user queries
    * Duplicate user queries (only one retained)
  * Detailed dataset statistics available in `Data_Info.txt`

## 1️⃣ `1_RealWorld_Conversations_Extraction`
Extraction of real-world conversations by subject.
* **`Extraction_by_LLM.py`**
  * Uses LLM semantic understanding to extract queries related to any subject in a predefined **subject list**. Subjects include:
    * Climate Change (what we studied in this work)
    * Health
    * Education
    * Finance
    * Math
    * Programming
  * Detailed extracted datasets are available on [HuggingFace](https://huggingface.co/datasets/Westing/LLM-Misalign-Climate-Change).
* **`Extraction_by_Keywords.py`**
  * Uses LLM-generated keywords for topic extraction. Supports multilingual extraction (14 languages including English, Chinese, Russian, etc.)
* **`Reddit_Extraction.py`**
  * Extracts question-style posts from specific Reddit forums.

## 2️⃣ `2_Data_Formats_Unification`
Unifies multiple datasets into a standardized format.
* **`Climate_Change.py`**
Combines climate change related data from:
* WildChat
* LMSYS-Chat-1M
* ClimateQ&A
* ClimaQA_Gold
* ClimaQA_Silver
* Reddit
* IPCC_AR6
* SciDCC
* Climate_FEVER
* Environmental_Claims
* ClimSight
After merging: **50,991 total samples**. Detailed dataset statistics available in `Data_Info.txt`.

## 3️⃣ `3_Topic_Modelling`
Implements Section 4.2 of the paper: **Topic Identification**
* **`Preliminary_Topic_Modelling.py`**
  * Implements *Initial Topic Generation*
  * Uses LLM to generate 1–3 free-form topics with expalnation per instance
* **`Topic_Merge.py`**
  * Implements *Iterative Topic Merging*
  * Bottom-up interative merging of free-form topics. LLM-assisted merging.
* **`Transitional_Final_Topics.py`**
  * Allows direct usage of merged topics as final taxonomy, maps initial topics to merged topics
* **`ReAssignment.py`**
  * Reassigns all samples 1–3 topics based on manually revised topic taxonomy
  * Statistics:
    * 50,991 samples total
    * 4,645 samples deleted due to they were labeled only as "Irrelevant Data" under the S5 topic merging setting
    * 46,346 retained (Four special samples that contain the "Irrelevant Data" label but also include other labels were also retained.)
    * 42,261 core-topic samples (remove 4085 samples labeled as "F. Others")
* **`Topic_Tree_Visualizer.html`**
  * Interactive tree visualization of topic merging process
  * Supports:
    * Keyword filtering/searching
    * Explanation/Count/Samples for each topic
  * Can be used with data in `Logs/`
* **`Statistical_Analysis.py`**
  * Topic distribution/Similarity analysis/Simple visualization

## 4️⃣ `4_Type_Modelling`
Implements Section 4.3 of the paper: **Question Type Classification**
* **`Type_Classification_ClimateChange.py`**
  * Uses LLM to annotate:
    * User intent (1-3 labels)
    * Expected answer form (1-3 labels)
  * If datasets already contain relevant dimensions, original labels are used
  * Irrelevant dimensions are ignored
* **`Statistical_Analysis.py`**
  * Question type (user intent/expected answer form) distribution/Similarity analysis/Simple visualization

## 5️⃣ `5_Visualization`
Final data organization and comprehensive visualization.
* **`Visualization_Web.py`**
After placing dataset files (see HuggingFace) in a unified folder:
```bash
python -m streamlit run Visualization_Web.py
```
The web interface provides:
* Tables
* Heatmaps
* Bar charts
* Differential bar charts

Supports:
* Analysis by individual dataset or analysis by grouped datasets (e.g., datasets under specific knowledge behavior)
* Category-level analysis or Fine-grained Topic/Type-level analysis
* Three different data-weighting methods
* Optional inclusion of “Others” category

---

# 🧠 Prompts
All LLM prompts used in the study are included:
* `1_Get_Keywords.txt` → Used by `Extraction_by_Keywords.py`
* `1_Subject_Match.txt` → Used by `Extraction_by_LLM.py`
* `1_Translate_Keywords.txt` → Used by `Extraction_by_Keywords.py`
* `3_Generate_Topics_ClimateChange.txt` → Used by `Preliminary_Topic_Modelling.py`
* `3_Merge_Topics_ClimateChange.txt` → Used by `Topic_Merge.py`
* `3_Reassign_Topics_ClimateChange.txt` → Used by `ReAssignment.py`
* `4_Generate_Types_ClimateChange.txt` → Used by `Type_Classification_ClimateChange.py`

# 📊 Logs
The `Logs/` directory contains topic merging information under **6 different experimental settings**. Refer to the paper appendix for detailed explanations.

# ⚙️ Configuration
All configurable settings are stored in:
```
Config.yaml
```

Including:
* File paths
* Adjustable parameters
* API keys
* Experimental settings

---

# 📄 Citation

```bibtex
@article{xxx202Xgaps,
  title={LLM Benchmark-User Need Misalignment for Climate Change},
  author={Anonymous},
  journal={To be added},
  year={202X}
}
```
