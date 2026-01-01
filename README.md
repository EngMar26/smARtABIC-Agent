# smARtABIC-Agent

An Arabic Retrieval-Augmented Generation (RAG) system for Arabic Question Answering.

This project retrieves relevant Arabic knowledge using semantic search (FAISS + E5 embeddings) and generates grounded answers for user queries.

## Features
- Arabic semantic retrieval using E5 embeddings
- FAISS vector search
- Robust handling of unknown questions
- Offline deployment
- Interactive Web Interface

## Dataset
We use a cleaned Arabic QA dataset split into training and evaluation subsets.

## How to Run

### Requirements
- Python 3.9 or higher
- pip

### Installation
1. Clone the repository or download it as a ZIP:
```bash
git clone https://github.com/EngMar26/smARtABIC-Agent.git
cd smARtABIC-Agent

2. Install required libraries:
 pip install -r requirements.txt

### Run the System
From the project folder, run:
  python app/server.py
Then open your browser at:
  http://127.0.0.1:8000


You can now ask questions in Arabic.

---

## Offline Usage

This system works fully offline after installation.

- All models and indices are stored locally.
- No internet connection is needed for answering questions.



