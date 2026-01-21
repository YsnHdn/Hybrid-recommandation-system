# 🚀 Hybrid Recommendation System

This project implements a **hybrid recommendation system** that combines **collaborative filtering** and **content-based filtering** to provide more accurate and personalized recommendations.

## 📌 Features

* Combines **user–item interactions** with **item metadata**
* Reduces cold-start issues
* Modular and scalable architecture
* API-ready deployment

---

## 🧠 Recommendation Approaches

### 1. Collaborative Filtering

* Singular Value Decomposition (**SVD**)
* Alternating Least Squares (**ALS**)
* Learns latent factors from user–item interactions

### 2. Content-Based Filtering

* Similarity computation based on item metadata
* Cosine similarity using feature vectors

### 3. Hybrid Strategy

* Weighted combination of collaborative and content-based scores
* Flexible tuning for performance optimization

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**, **NumPy**
* **Scikit-learn**
* **Surprise**
* **FastAPI** (REST API)
* **MLflow** (experiment tracking & model versioning)

---

## 📂 Project Structure

```text
.
├── data/          # Datasets
├── notebooks/     # Exploratory analysis & experiments
├── src/           # Core source code
├── models/        # Trained and saved models
├── scripts/       # Training and evaluation scripts
└── requirements.txt
```

---

## ⚙️ Setup & Installation

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Running the Project

```bash
python scripts/train.py
```

To launch the API:

```bash
uvicorn src.api.main:app --reload
```

---

## 📈 Experiment Tracking

* All experiments are logged using **MLflow**
* Metrics, parameters, and models are versioned for reproducibility

---

## 🎯 Use Cases

* E-commerce product recommendations
* Movie or music recommendation platforms
* Personalized content delivery systems

---
