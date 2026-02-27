# Gaps in LLM Development on Socio-scientific Issues

## Insights from a Climate Change Case Study — Code Repository

This repository contains the official code for the paper:

> **Gaps in LLM Development on Socio-scientific Issues: Insights from a Climate Change Case Study**
> 📄 Paper link: #TODO
> 🤗 HuggingFace Dataset & Resources: #TODO

---

## 📌 Overview

This project investigates the Misalignment between Benchmarks and User Needs, using  Climate Change as a Case Study.

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
    * Finance
    * Math
    * Programming
    * Health
    * Education
  * Detailed extracted datasets are available on HuggingFace #TODO.
* **`Extraction_by_Keywords.py`**
  * Uses LLM-generated keywords for topic extraction. Supports multilingual extraction (14 languages including English, Chinese, Russian, etc.)
* **`Reddit_Extraction.py`**
  * Extracts question-style posts from specific Reddit forums.

## 2️⃣ `2_Data_Formats_Unification`

Unifies multiple datasets into a standardized format.

### Script

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

After merging:

* **50,991 total samples**

Detailed dataset statistics available in `Data_Info.txt`.

---

## 3️⃣ `3_Topic_Modelling`

Implements Section 4.2 of the paper: **Topic Identification**

### Scripts

* **`Preliminary_Topic_Modelling.py`**

  * Implements *Initial Topic Generation*
  * Uses LLM to generate **1–3 free-form topics** per instance

* **`Topic_Merge.py`**

  * Implements *Iterative Topic Merging*
  * Bottom-up recursive merging of free-form topics
  * LLM-assisted clustering & merging

* **`Transitional_Final_Topics.py`**

  * Maps merged topics to final topics
  * Allows direct usage of merged topics as final taxonomy

* **`ReAssignment.py`**

  * Reassigns all samples based on manually revised topic taxonomy
  * Statistics:

    * 50,991 total
    * 4,649 deleted as irrelevant
    * 46,342 retained
    * 42,261 core-topic samples

* **`Topic_Tree_Visualizer.html`**

  * Interactive tree visualization of topic merging process
  * Supports:

    * Keyword filtering
    * Searching
    * Node locating
  * Can be used with data in `Logs/`

* **`Statistical_Analysis.py`**

  * Topic distribution
  * Similarity analysis
  * Visualization

---

## 4️⃣ `4_Type_Modelling`

Implements Section 4.3 of the paper: **Question Type Classification**

### Scripts

* **`Type_Classification_ClimateChange.py`**

  * Uses LLM to annotate:

    * User intent
    * Expected answer form
  * If datasets already contain relevant dimensions, original labels are used
  * Irrelevant dimensions are ignored

* **`Statistical_Analysis.py`**

  * Distribution analysis
  * Similarity analysis
  * Visualization of question types

---

## 5️⃣ `5_Visualization`

Final data organization and comprehensive visualization.

### Script

* **`Visualization_Web.py`**

After placing dataset files (see HuggingFace) in a unified folder:

```bash
python -m streamlit run Visualization_Web.py
```

### Features

The web interface provides:

* Tables
* Heatmaps
* Bar charts
* Differential bar charts

Supports:

* Analysis by individual dataset
* Analysis by grouped datasets (e.g., datasets under specific knowledge behavior)
* Optional inclusion of “Others” category
* Three different data-weighting methods
* Category-level analysis
* Fine-grained Topic/Type-level analysis

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

---
# Gaps in LLM Development on Socio-scientific Issues

## Insights from a Climate Change Case Study — Code Repository

This repository contains the official code for the paper:

> **Gaps in LLM Development on Socio-scientific Issues: Insights from a Climate Change Case Study**
> 📄 Paper link: 【To be added】
> 🤗 HuggingFace Dataset & Resources: 【To be added】

---

## 📌 Overview

This project investigates how Large Language Models (LLMs) perform on socio-scientific issues, using **Climate Change** as a case study.

The repository provides:

* Data extraction and preprocessing pipelines
* Multi-source dataset unification
* Topic modeling (LLM-driven + iterative merging)
* Question type modeling
* Statistical analysis & visualization
* Interactive web-based visualization tool

The full climate change dataset used in the paper contains:

* **50,991 total instances**
* **4,649 irrelevant samples removed**
* **46,342 retained samples**
* **42,261 core-topic samples**

Detailed statistics can be found in `Data_Info.txt` inside relevant folders.

---

# 📂 Repository Structure

---

## 0️⃣ `0_Firstturn_Deduplicated_Conversations_Extraction`

Basic preprocessing of human–LLM conversation datasets.

### Scripts

* **`Extract_Firsturn_Dedup.py`**

  * Extracts the **first-turn conversations** from:

    * WildChat
    * LMSYS-Chat-1M
  * Removes:

    * Conversations with empty user queries
    * Duplicate user queries (only one retained)
  * Detailed dataset statistics available in `Data_Info.txt`

---

## 1️⃣ `1_RealWorld_Conversations_Extraction`

Extraction of real-world conversations by subject.

### Scripts

* **`Extraction_by_LLM.py`**

  * Uses LLM semantic understanding
  * Extracts queries related to any subject in a predefined **subject list**
  * Subjects include:

    * Climate Change (main focus)
    * Finance
    * Math
    * Programming
    * Health
    * Education
  * Detailed extracted datasets are available on HuggingFace.

* **`Extraction_by_Keywords.py`**

  * Uses LLM-generated keywords for topic extraction
  * Supports multilingual extraction (14 languages including English, Chinese, Russian, etc.)
  * Pipeline:

    1. Generate keywords via LLM
    2. Translate keywords
    3. Perform keyword matching

* **`Reddit_Extraction.py`**

  * Extracts question-style posts from specific Reddit forums.

---

## 2️⃣ `2_Data_Formats_Unification`

Unifies multiple datasets into a standardized format.

### Script

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

After merging:

* **50,991 total samples**

Detailed dataset statistics available in `Data_Info.txt`.

---

## 3️⃣ `3_Topic_Modelling`

Implements Section 4.2 of the paper: **Topic Identification**

### Scripts

* **`Preliminary_Topic_Modelling.py`**

  * Implements *Initial Topic Generation*
  * Uses LLM to generate **1–3 free-form topics** per instance

* **`Topic_Merge.py`**

  * Implements *Iterative Topic Merging*
  * Bottom-up recursive merging of free-form topics
  * LLM-assisted clustering & merging

* **`Transitional_Final_Topics.py`**

  * Maps merged topics to final topics
  * Allows direct usage of merged topics as final taxonomy

* **`ReAssignment.py`**

  * Reassigns all samples based on manually revised topic taxonomy
  * Statistics:

    * 50,991 total
    * 4,649 deleted as irrelevant
    * 46,342 retained
    * 42,261 core-topic samples

* **`Topic_Tree_Visualizer.html`**

  * Interactive tree visualization of topic merging process
  * Supports:

    * Keyword filtering
    * Searching
    * Node locating
  * Can be used with data in `Logs/`

* **`Statistical_Analysis.py`**

  * Topic distribution
  * Similarity analysis
  * Visualization

---

## 4️⃣ `4_Type_Modelling`

Implements Section 4.3 of the paper: **Question Type Classification**

### Scripts

* **`Type_Classification_ClimateChange.py`**

  * Uses LLM to annotate:

    * User intent
    * Expected answer form
  * If datasets already contain relevant dimensions, original labels are used
  * Irrelevant dimensions are ignored
# Gaps in LLM Development on Socio-scientific Issues

## Insights from a Climate Change Case Study — Code Repository

This repository contains the official code for the paper:

> **Gaps in LLM Development on Socio-scientific Issues: Insights from a Climate Change Case Study**
> 📄 Paper link: 【To be added】
> 🤗 HuggingFace Dataset & Resources: 【To be added】

---

## 📌 Overview

This project investigates how Large Language Models (LLMs) perform on socio-scientific issues, using **Climate Change** as a case study.

The repository provides:

* Data extraction and preprocessing pipelines
* Multi-source dataset unification
* Topic modeling (LLM-driven + iterative merging)
* Question type modeling
* Statistical analysis & visualization
* Interactive web-based visualization tool

The full climate change dataset used in the paper contains:

* **50,991 total instances**
* **4,649 irrelevant samples removed**
* **46,342 retained samples**
* **42,261 core-topic samples**

Detailed statistics can be found in `Data_Info.txt` inside relevant folders.

---

# 📂 Repository Structure

---

## 0️⃣ `0_Firstturn_Deduplicated_Conversations_Extraction`

Basic preprocessing of human–LLM conversation datasets.

### Scripts

* **`Extract_Firsturn_Dedup.py`**

  * Extracts the **first-turn conversations** from:

    * WildChat
    * LMSYS-Chat-1M
  * Removes:

    * Conversations with empty user queries
    * Duplicate user queries (only one retained)
  * Detailed dataset statistics available in `Data_Info.txt`

---

## 1️⃣ `1_RealWorld_Conversations_Extraction`

Extraction of real-world conversations by subject.

### Scripts

* **`Extraction_by_LLM.py`**

  * Uses LLM semantic understanding
  * Extracts queries related to any subject in a predefined **subject list**
  * Subjects include:

    * Climate Change (main focus)
    * Finance
    * Math
    * Programming
    * Health
    * Education
  * Detailed extracted datasets are available on HuggingFace.

* **`Extraction_by_Keywords.py`**

  * Uses LLM-generated keywords for topic extraction
  * Supports multilingual extraction (14 languages including English, Chinese, Russian, etc.)
  * Pipeline:

    1. Generate keywords via LLM
    2. Translate keywords
    3. Perform keyword matching

* **`Reddit_Extraction.py`**

  * Extracts question-style posts from specific Reddit forums.

---

## 2️⃣ `2_Data_Formats_Unification`

Unifies multiple datasets into a standardized format.

### Script

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

After merging:

* **50,991 total samples**

Detailed dataset statistics available in `Data_Info.txt`.

---

## 3️⃣ `3_Topic_Modelling`

Implements Section 4.2 of the paper: **Topic Identification**

### Scripts

* **`Preliminary_Topic_Modelling.py`**

  * Implements *Initial Topic Generation*
  * Uses LLM to generate **1–3 free-form topics** per instance

* **`Topic_Merge.py`**

  * Implements *Iterative Topic Merging*
  * Bottom-up recursive merging of free-form topics
  * LLM-assisted clustering & merging

* **`Transitional_Final_Topics.py`**

  * Maps merged topics to final topics
  * Allows direct usage of merged topics as final taxonomy

* **`ReAssignment.py`**

  * Reassigns all samples based on manually revised topic taxonomy
  * Statistics:

    * 50,991 total
    * 4,649 deleted as irrelevant
    * 46,342 retained
    * 42,261 core-topic samples

* **`Topic_Tree_Visualizer.html`**

  * Interactive tree visualization of topic merging process
  * Supports:

    * Keyword filtering
    * Searching
    * Node locating
  * Can be used with data in `Logs/`

* **`Statistical_Analysis.py`**

  * Topic distribution
  * Similarity analysis
  * Visualization

---

## 4️⃣ `4_Type_Modelling`

Implements Section 4.3 of the paper: **Question Type Classification**

### Scripts

* **`Type_Classification_ClimateChange.py`**

  * Uses LLM to annotate:

    * User intent
    * Expected answer form
  * If datasets already contain relevant dimensions, original labels are used
  * Irrelevant dimensions are ignored

* **`Statistical_Analysis.py`**

  * Distribution analysis
  * Similarity analysis
  * Visualization of question types

---

## 5️⃣ `5_Visualization`

Final data organization and comprehensive visualization.

### Script

* **`Visualization_Web.py`**

After placing dataset files (see HuggingFace) in a unified folder:

```bash
python -m streamlit run Visualization_Web.py
```

### Features

The web interface provides:

* Tables
* Heatmaps
* Bar charts
* Differential bar charts

Supports:

* Analysis by individual dataset
* Analysis by grouped datasets (e.g., datasets under specific knowledge behavior)
* Optional inclusion of “Others” category
* Three different data-weighting methods
* Category-level analysis
* Fine-grained Topic/Type-level analysis

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

---

# 📊 Logs

The `Logs/` directory contains topic merging information under **6 different experimental settings**.

Refer to the paper appendix for detailed explanations.

---

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

# 🚀 Reproducibility Pipeline

Recommended execution order:

1. First-turn extraction
2. Subject-based real-world extraction
3. Data format unification
4. Topic modeling
5. Type modeling
6. Statistical analysis
7. Visualization

---

# 📌 Notes

* Climate Change is the primary case study.
* Other domains (Finance, Math, Programming, Health, Education) were also extracted for broader comparison.
* The full processed datasets and statistics are available on HuggingFace (link to be added).

---

# 📄 Citation

```bibtex
@article{xxx202Xgaps,
  title={Gaps in LLM Development on Socio-scientific Issues: Insights from a Climate Change Case Study},
  author={Anonymous},
  journal={To be added},
  year={202X}
}
```

---

# 📬 Contact

For questions or collaborations, please open an issue in this repository.

---

If you would like, I can also:

* Make this more concise (conference-style)
* Or create a more polished academic-version README
* Or generate a Chinese version README
* Or add a project pipeline diagram version 🚀

* **`Statistical_Analysis.py`**

  * Distribution analysis
  * Similarity analysis
  * Visualization of question types

---

## 5️⃣ `5_Visualization`

Final data organization and comprehensive visualization.

### Script

* **`Visualization_Web.py`**

After placing dataset files (see HuggingFace) in a unified folder:

```bash
python -m streamlit run Visualization_Web.py
```

### Features

The web interface provides:

* Tables
* Heatmaps
* Bar charts
* Differential bar charts

Supports:

* Analysis by individual dataset
* Analysis by grouped datasets (e.g., datasets under specific knowledge behavior)
* Optional inclusion of “Others” category
* Three different data-weighting methods
* Category-level analysis
* Fine-grained Topic/Type-level analysis

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

---

# 📊 Logs

The `Logs/` directory contains topic merging information under **6 different experimental settings**.

Refer to the paper appendix for detailed explanations.

---

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

# 🚀 Reproducibility Pipeline

Recommended execution order:

1. First-turn extraction
2. Subject-based real-world extraction
3. Data format unification
4. Topic modeling
5. Type modeling
6. Statistical analysis
7. Visualization

---

# 📌 Notes

* Climate Change is the primary case study.
* Other domains (Finance, Math, Programming, Health, Education) were also extracted for broader comparison.
* The full processed datasets and statistics are available on HuggingFace (link to be added).

---

# 📄 Citation

```bibtex
@article{xxx202Xgaps,
  title={Gaps in LLM Development on Socio-scientific Issues: Insights from a Climate Change Case Study},
  author={Anonymous},
  journal={To be added},
  year={202X}
}
```

---

# 📬 Contact

For questions or collaborations, please open an issue in this repository.

---

If you would like, I can also:

* Make this more concise (conference-style)
* Or create a more polished academic-version README
* Or generate a Chinese version README
* Or add a project pipeline diagram version 🚀

# 📊 Logs

The `Logs/` directory contains topic merging information under **6 different experimental settings**.

Refer to the paper appendix for detailed explanations.

---

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

# 🚀 Reproducibility Pipeline

Recommended execution order:

1. First-turn extraction
2. Subject-based real-world extraction
3. Data format unification
4. Topic modeling
5. Type modeling
6. Statistical analysis
7. Visualization

---

# 📌 Notes

* Climate Change is the primary case study.
* Other domains (Finance, Math, Programming, Health, Education) were also extracted for broader comparison.
* The full processed datasets and statistics are available on HuggingFace (link to be added).

---

# 📄 Citation

```bibtex
@article{xxx202Xgaps,
  title={Gaps in LLM Development on Socio-scientific Issues: Insights from a Climate Change Case Study},
  author={Anonymous},
  journal={To be added},
  year={202X}
}
```

---
