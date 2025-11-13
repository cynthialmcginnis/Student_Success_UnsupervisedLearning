
<img width="1536" height="1024" alt="ChatGPT Image Nov 13, 2025, 06_11_24 PM" src="https://github.com/user-attachments/assets/98ad66c7-f485-44c1-af83-f9a4a57a24d1" />

# Student Success — Unsupervised Learning (Personas via NMF + K-Means)

Early, **non-stigmatizing** insight into student patterns using **unsupervised learning only**.  
We discover **personas** from academic, engagement, and behavioral features using **NMF (matrix factorization)** and **K-Means**. Labels (e.g., `Dropout_Risk`) are used **post-hoc for context only**—never for training.

---

##  Problem
Identify a small number of **interpretable student personas** to support advising conversations:
- What patterns exist across GPA, credits, attendance, study/online hours, and late submissions?
- How do personas differ (means/medians, key factor themes)?
- Optional: “students like me” view via cosine neighbors in factor space.

---

##  Methods (course-aligned)
- **Preprocessing:** `ColumnTransformer` = MinMax scale numerics + One-Hot encode categoricals (non-negative for NMF).
- **Matrix Factorization:** **NMF(r=12)** → student factor scores `W` and feature loadings `H`.
- **Clustering:** **K-Means** on `W`, model selection by **silhouette** → **k=2** (moderate separation).
- **Similarity (optional):** Cosine neighbors in `W` (recommender-style “students like me”).

---

##  Data
- ~1,000 students × 23 columns (no missing in this snapshot).
- Excluded from training: identifiers and protected attributes (Gender, Ethnicity, SES, Disability, Parental Education).  
- Labels, if present (e.g., `Dropout_Risk`), are **context only**.

---

## Results

### What we did (1-line)
Unsupervised pipeline: **MinMax + OneHot → NMF (rank=12) → K-Means (k=2)** to discover student **personas** from academics + engagement features (labels used only post-hoc for context).

### Model selection
- **NMF rank sweep (r=2…12):** steadily decreasing reconstruction error, elbow ~10–12 → chose **r=12** (good trade-off: lower error + interpretable factors).
- **K-Means (k=2…10) on NMF scores:** best **silhouette ≈ 0.23 at k=2**, indicating **moderately separated** but overlapping cohorts (soft segments).

### Personas (plain-English)
- **Persona A — Further-along / higher GPA & credits**
  - Tends to have **higher GPA**, **more credits**, slightly **fewer late submissions**, and **similar study/online hours**.
- **Persona B — Deadline-friction / lower-engagement cohort**
  - Slightly **lower GPA/credits**, **more late submissions**, somewhat **lower attendance**; study and online hours are comparable but less consistent.

> Note: Differences are **modest** (silhouette < 0.3), so treat them as **personas**, not hard classes.

### Visual highlights
- **NMF rank sweep**: shows error decreasing and stabilizing near r≈12.
- **Cluster quality curve (silhouette vs k)**: peak at **k=2**.
- **2D/3D factor plots**: overlapping clouds with separated centroids.
- **Persona radar chart**: relative-to-cohort (1.0 = cohort mean) for GPA, Credits, Attendance, Study hrs, Online hrs, Late submissions (↓ desirable).

### “Students like me” (Recommender-style view)
Cosine similarity in NMF space surfaces top-K neighbors per student. This supports **case-based** interpretation (how similar students look across features) without training a predictor.

### Post-hoc context (labels not used for training)
- `Dropout_Risk` is **highly imbalanced (~1.8% positives)**. Cluster rates differ slightly (e.g., ~0% vs ~3–4%), but counts are tiny → **descriptive only**, not evidence of predictive power.

### Fairness & stability checks
- Protected attributes (Gender, Ethnicity, SES, Disability, Parental Education) **excluded from training**.
- Post-hoc cluster composition tables show **no glaring skews**; we recommend monitoring Cramér’s V and silhouette-by-group in future cohorts.

### Takeaways
- **Structure exists**: two interpretable personas emerge consistently.
- **Separation is moderate**: useful for **early, non-stigmatizing guidance**, not for classification.
- **Actionability**: persona profiles + “neighbors” help advisors frame supports (study planning, pacing, deadline management).

### Limitations
- Unsupervised clusters can drift across cohorts; re-fit periodically.
- Low silhouette → **soft boundaries**; use aggregate patterns, not per-student judgments.
- Imbalanced labels make any post-hoc rate comparisons fragile.

### Reproduce
1. Run the notebook/script: `StudentSuccessUnsupervisedLearning.py` (or the Colab link).
2. Data: `data/student_management_dataset.csv` (or your path).
3. Outputs:
   - `reports/` figures: NMF sweep, silhouette curve, factor plots, persona radar.
   - Tables: persona profiles and optional fairness tables.

### Next steps
- Try **HDBSCAN** for non-spherical clusters + noise handling.
- Add **SVD** comparison and stability checks (bootstrap clustering).
- (Optional) Build a **Streamlit** mini-app: upload a CSV → fit → persona dashboard + “students like me” view.


---
GitHub-based Colab badge 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/cynthialmcginnis/Student_Success_UnsupervisedLearning/blob/main/notebooks/student_personas.ipynb)


