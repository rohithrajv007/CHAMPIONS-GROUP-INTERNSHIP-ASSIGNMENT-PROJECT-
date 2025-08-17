
# 🚀 LinkedIn AWS Certification Likelihood Predictor

A machine learning–powered app that predicts the likelihood of a LinkedIn user holding an  
**AWS Certified Machine Learning – Specialty** credential based on their **skills** and **work experience**.

It provides:
- **Scenario 1:** Instant detection if a certification is explicitly listed.  
- **Scenario 2:** Similarity-based scoring for uncertified profiles using pretrained embeddings **with relevant skill filtering**.  
- A **ranked JSON output** sorted by confidence score.  
- A clean **Streamlit web interface** for CSV uploads & downloads.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Scenarios & Logic](#scenarios--logic)
- [Model & Approach](#model--approach)
- [Synthetic Data Preparation](#synthetic-data-preparation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Example Output](#example-output)
- [Future Improvements](#future-improvements)

---

## 📖 Project Overview
This project addresses a scenario where we want to **score and rank LinkedIn profiles** according to how likely it is that the user holds an AWS Machine Learning certification.

LinkedIn profiles often contain **skills** and **work experience**. In many cases:
- Certifications are explicitly listed.
- In other cases, relevant experience is present without formal certification.

Our tool evaluates both situations:
- **If the certificate is listed** → we output 100% certainty.
- **If the certificate is not listed** → we compute a similarity measure between the user’s skill/work profile and those of known certified users, filtering to only relevant AWS ML skills to improve scoring accuracy.

---

## 🧠 Scenarios & Logic

### **Scenario 1 — Certified Match**
- Uses **exact and fuzzy text matching** (via `rapidfuzz`) to detect known AWS ML certification names in the `certifications` column.
- If found:
  - `certified_status = True`
  - `confidence_score = 100%`

### **Scenario 2 — Similarity-Based Scoring**
- For users **without** explicit certifications:
  - **NEW:** Filter the skills text to retain only **relevant AWS ML skills** from a whitelist, reducing noise from unrelated skills.
  - Generate embeddings of filtered `skills` and full `work_experience` using a **Sentence-BERT** model (`all-MiniLM-L6-v2`).
  - Compare each embedding to **reference embeddings** computed from certified profiles (train set only).
  - Score = `(skills_sim * 0.5 + work_sim * 0.5) * 100`
  - Cap maximum score at **95%** to reflect uncertainty without certification.

**Reference Embeddings** = Average embeddings of all certified profiles from the training dataset.

---

## 📊 Model & Approach
We deliberately use a **semantic similarity model** instead of a classifier because:
- Text data from LinkedIn is unstructured and sparse (skills vary widely in naming).
- Semantic embeddings let us **capture meaning** (e.g., “AWS SageMaker” ~ “AWS Machine Learning Service”) even if wording differs.
- **Enhancement:** By filtering only relevant AWS ML skills before embedding, we reduce score drops caused by unrelated skills in a candidate's profile while keeping high similarity for truly relevant skill sets.
- This design works perfectly for **low-data environments** — as long as we have a few certified examples, we can score new inputs.

**Model:**  
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — lightweight & accurate for semantic embeddings (~384 dims).
- Cosine similarity from `scikit-learn` used for comparison.

**Why not an ML classifier?**
- Would require labeled negatives/positives in large quantity.
- Embedding-based approach is faster to adapt and easier to explain.

---

## 🏗 Synthetic Data Preparation
We built synthetic datasets to simulate LinkedIn profiles, ensuring no real personal data is used.

### 1. **train_profiles.csv**
- Contains **only certified profiles** to build reference embeddings.
- Includes realistic skills (AWS SageMaker, Lambda, Amazon Rekognition, etc.) and relevant work experience text.

### 2. **test_profiles.csv**
- Mixed profiles:
  - Certified (Scenario 1)
  - High similarity but uncertified
  - Medium similarity uncertified
  - Low similarity, unrelated roles
- Used **only** for scoring/evaluation (no overlap with training).

### 3. **test_profiles_100.csv**
- Larger stress test dataset with 100 entries generated programmatically.
- Same realistic distribution of scenarios.

---

## 💻 Tech Stack
- **Python 3.10+**
- **Core ML/NLP**:
  - `sentence-transformers`
  - `scikit-learn`
  - `numpy`, `pandas`
- **Text Matching**:
  - `rapidfuzz`
- **Persistence**:
  - `joblib`
- **Web App**:
  - `streamlit`
- **Other**:
  - `json` for export
  - `re` for preprocessing

---

## 📂 Project Structure
```

linkedin_cert_predictor/
│
├── app/
│   └── app_streamlit.py        \# Streamlit web UI
├── src/
│   ├── config.py               \# Constants/config (includes RELEVANT_SKILLS whitelist)
│   ├── preprocess.py           \# Text cleaning
│   ├── matcher.py              \# Scenario 1 matching
│   ├── embeddings.py           \# Embedding generation
│   ├── reference_builder.py    \# Build \& save reference embeddings
│   ├── scorer.py               \# Scenario 2 similarity scoring with skill filtering
│   └── pipeline.py             \# End-to-end CSV → JSON pipeline
├── data/
│   ├── train_profiles.csv
│   ├── test_profiles.csv
│   └── test_profiles_100.csv
├── artifacts/
│   ├── reference_embeddings.joblib
│   └── results.json
├── requirements.txt
├── README.md
└── .gitignore

```

---

## ⚙️ Installation & Setup
```

git clone https://github.com/YOUR_USERNAME/linkedin_cert_predictor.git
cd linkedin_cert_predictor
pip install -r requirements.txt

```

Build reference embeddings from training data:
```

python -m src.reference_builder

```

---

## 🚀 Usage
**Run Pipeline on Test CSV:**
```

python -m src.pipeline data/test_profiles.csv artifacts/results_test.json

```

**Run Streamlit UI:**
```

streamlit run app/app_streamlit.py

```
- Upload your CSV in the correct format  
- Get an interactive table sorted by score  
- Download results as JSON

---

## 📄 Example Output
```

[
{
"name": "Alice Johnson",
"certified_status": true,
"confidence_score": 100.0
},
{
"name": "Daniel Lee",
"certified_status": false,
"confidence_score": 93.45
}
]

```

---

## 🔮 Future Improvements
- Add color-coded visual rankings in Streamlit table.
- Fine-tune weight between skill similarity and experience similarity.
- Expand the RELEVANT_SKILLS whitelist for broader coverage.
- Add more realistic certification name variants for better recall.
- Optionally turn into a REST API with Flask.

---

## Example Images
![Screenshot 1](https://raw.githubusercontent.com/rohithrajv007/CHAMPIONS-GROUP-INTERNSHIP-ASSIGNMENT-PROJECT-/main/Screenshot%202025-08-15%20020423.png)
![Screenshot 2](https://raw.githubusercontent.com/rohithrajv007/CHAMPIONS-GROUP-INTERNSHIP-ASSIGNMENT-PROJECT-/main/Screenshot%202025-08-15%20020521.png)
![Screenshot 3](https://raw.githubusercontent.com/rohithrajv007/CHAMPIONS-GROUP-INTERNSHIP-ASSIGNMENT-PROJECT-/main/Screenshot%202025-08-15%20020632.png)
![Screenshot 4](https://raw.githubusercontent.com/rohithrajv007/CHAMPIONS-GROUP-INTERNSHIP-ASSIGNMENT-PROJECT-/main/Screenshot%202025-08-15%20020641.png)
```




