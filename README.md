# 🌱 Root & Rise — DSN x BCT LLM Agent Challenge

**Team:** Root & Rise | **Solo Participant:** Gabriel Effangha  
**Hackathon:** Data & AI Summit Hackathon 3.0 — May 4–24, 2026

---

## Overview

Two LLM-powered agents built for the DSN x Bluechip Technologies LLM Agent Challenge.

| Task | Description | Deliverable |
|------|-------------|-------------|
| **Task A** | User Modeling — simulate user reviews & ratings for unseen items | FastAPI containerized app |
| **Task B** | Recommendation — personalized, agentic item recommendations | FastAPI containerized app |

**LLM Backbone:** Groq API (Llama 3.3 70B)  
**Datasets:** Yelp Academic Dataset, Amazon Reviews, Goodreads  
**Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2)  
**Vector Search:** FAISS  

---

## Repository Structure

```
root-and-rise/
├── shared/                  # Shared modules used by both tasks
│   ├── persona.py           # UserPersona data model
│   ├── llm_client.py        # Groq LLM wrapper
│   └── embeddings.py        # Sentence-transformer wrapper
├── task_a/                  # Task A: User Modeling
├── task_b/                  # Task B: Recommendation Agent
├── data/                    # Datasets (raw/processed/sample)
├── notebooks/               # EDA and analysis notebooks
├── scripts/                 # Preprocessing utilities
├── solution_paper/          # 4–8 page solution write-up
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quickstart

```bash
git clone https://github.com/Geff115/root-and-rise.git
cd root-and-rise
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY
```

**Run Task A:** `cd task_a && uvicorn app:app --reload --port 8000`  
**Run Task B:** `cd task_b && uvicorn app:app --reload --port 8001`  
**Docker:** `docker-compose up --build`

---

## Datasets

Download from https://www.yelp.com/dataset and place in `data/raw/yelp/`:
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_user.json`

---

*Built with 🔥 by Gabriel for DSN x BCT LLM Agent Challenge 2026*