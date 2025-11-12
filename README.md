# Student Success — Unsupervised Learning (Personas via NMF + K-Means)

Early, **non-stigmatizing** insight into student patterns using **unsupervised learning only**.  
We discover **personas** from academics, engagement, and behavior features with **NMF (matrix factorization)** and **K-Means**. Labels (e.g., `Dropout_Risk`) are used **post-hoc for context only**—never for training.

---

## 📌 Problem
Identify a small number of **interpretable student personas** to support advising conversations:
- What patterns exist across GPA, credits, attendance, study/online hours, late submissions?
- How do personas differ (means/medians, key factor themes)?
- Optional: “students like me” view via cosine neighbors in factor space.

---

## 🔧 Methods (course-aligned)
- **Preprocessing:** `ColumnTransformer` = MinMax scale numerics + One-Hot encode categoricals (non-negative for NMF).
- **Matrix Factorization:** **NMF(r=12)** → student factor scores `W` and feature loadings `H`.
- **Clustering:** **K-Means** on `W`, model selection by **silhouette** → **k=2** (moderate separation).
- **Similarity (optional):** Cosine neighbors in `W` (recommender-style “students like me”).

---

## 🧪 Data
- ~1,000 students × 23 columns (no missing in this snapshot).
- Excluded from training: identifiers and protected attributes (Gender, Ethnicity, SES, Disability, Parental Education).  
- Labels, if present (e.g., `Dropout_Risk`), are **context only**.

---

## 📊 Key Result — Persona Spotlight

> Radial chart shows **persona means relative to cohort average (1.0)** on select metrics  
> (higher = above cohort mean, lower = below; Late subs is inverted so **lower is better**).

<img width="996" height="658" alt="Screenshot 2025-11-12 at 3 21 11 PM" src="https://github.com/user-attachments/assets/dfb37fa1-5e33-4b9c-a3e0-a570bc3d73b0" />


**Interpretation (example):**
- **Persona 1 (RED)**: Below-avg GPA, avg credits, **lower attendance**, **more late submissions**, slightly lower online/study hours → candidate for time-management and deadline coaching.
- **Persona 2 (not shown here)**: The complement—higher attendance/credits, fewer late subs.

> Silhouette ≈ **0.23** → **soft segments** with overlap; treat personas as **support prompts**, not labels.

---

## 🗂️ Repository Structure
