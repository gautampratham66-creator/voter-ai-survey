# 🗳️ AI-Powered Voter ID Survey System

> **Agentic AI + RAG + Machine Learning** for State Census Survey

A modern AI system to survey Indian citizens for Voter ID possession across Uttar Pradesh. Built with multi-agent AI, Retrieval Augmented Generation (RAG), and Machine Learning.

---

## 🚀 Live Demo
Run locally with: `streamlit run app.py`

---

## ✨ Features

| Feature | Technology |
|---|---|
| 📄 Document Agent | Claude Vision API reads Aadhaar/ID cards |
| 📋 Survey Agent | Auto-stores family data (CrewAI) |
| 💬 RAG Chatbot | ChromaDB + Claude answers questions |
| 📊 Dashboard & Graphs | Plotly interactive charts |
| 📈 ML Predictions | Random Forest + Anomaly Detection |
| 📄 Official Reports | AI-generated district reports |
| 📰 Articles & Insights | AI-written research articles |

---

## 🤖 AI Architecture

```
MASTER ORCHESTRATOR (CrewAI)
├── 📄 Document Agent  → Claude Vision reads ID docs
├── 📋 Survey Agent    → Stores data in PostgreSQL
├── 💬 RAG Agent       → Queries ChromaDB vector store
└── 📊 Report Agent    → Generates official reports
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Agentic AI**: CrewAI + LangChain
- **AI Brain**: Claude API (claude-sonnet-4-6)
- **RAG**: LangChain + ChromaDB
- **ML**: Scikit-learn (Random Forest + Isolation Forest)
- **Graphs**: Plotly
- **Database**: PostgreSQL / SQLite

---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/gautampratham66/voter-ai-survey.git
cd voter-ai-survey

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export ANTHROPIC_API_KEY="your_key_here"

# 4. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
voter-ai-survey/
├── app.py                    ← Main Streamlit UI
├── requirements.txt
├── agents/
│   └── crew_agents.py        ← Multi-agent system (CrewAI)
├── rag/
│   └── rag_system.py         ← RAG with ChromaDB
├── ml_model/
│   └── predictor.py          ← ML models
└── database/
    └── schema.py             ← DB schema
```

---

## 👨‍💻 Author
**Gautam Pratham** — [@gautampratham66](https://github.com/gautampratham66)

---

## 📄 License
MIT License
