# 🧮 Math-Optimized RAG Pipeline (JEE Mains)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Framework-orange)
![LLM](https://img.shields.io/badge/Model-Qwen--2.5--32B-purple)
![RAG](https://img.shields.io/badge/RAG-Ensemble%20Retrieval-green)

> **Solving the "Math Gap" in LLMs.** A specialized RAG pipeline designed to handle complex LaTeX equations, preserve logical continuity, and deliver 90% accuracy on JEE Mains problems.

---

## 🚀 Overview

General-purpose RAG systems often fail with mathematics because standard chunking breaks equations and loses the logical link between a problem and its solution. 

This project engineers a **Structure-Aware RAG pipeline** specifically for competitive exams. By combining a custom JSON dataset of 50+ JEE papers with an **Ensemble Retrieval strategy** (Semantic + Keyword), this system ensures that the LLM retrieves the full context—question, steps, and solution—intact.

## ✨ Key Features

* **🧩 Structure-Aware Chunking:** engineered a custom strategy that respects the logical boundaries of math problems. Unlike standard text splitters, this preserves the continuity between Questions, Solutions, and Topic Tags.
* **🔍 Ensemble Retrieval:** Implemented a hybrid retriever combining **BAAI/bge-m3 embeddings** (for semantic understanding) and **BM25** (for exact keyword/formula matching) to maximize retrieval accuracy.
* **📚 Math-Optimized Dataset:** Constructed a clean, structured JSON dataset from **50+ JEE Papers**, complete with parsed **LaTeX equations** and granular metadata tags.
* **🧠 History-Aware Memory:** Integrated conversational memory that tracks the solving steps, allowing users to ask follow-up doubts without the model losing context.
* **⚡ High-Performance Inference:** Powered by **Qwen-2.5-32B (via Groq)**, achieving **90% accuracy** on a pilot validation set while strictly adhering to the JEE Mains syllabus.

---

## 🛠️ The Architecture

1.  **Data Parsing:** PDF papers are converted to structured JSON with explicit LaTeX parsing.
2.  **Indexing:** * *Dense Index:* ChromaDB with BGE-M3 embeddings.
    * *Sparse Index:* BM25 for keyword/symbol precision.
3.  **Retrieval:** The Ensemble Retriever queries both indices and re-ranks results.
4.  **Generation:** The retrieved context is fed into Qwen-2.5-32B to generate the step-by-step solution.

---

## 🏆 Performance

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Accuracy** | **90%** | Tested on a pilot set of 100 queries. |
| **Syllabus Adherence** | **Strict** | Generation is constrained to JEE Mains topics only. |
| **Retrieval Speed** | **<200ms** | Optimized via Groq API. |

---

## 💻 Tech Stack

* **Framework:** LangChain
* **LLM:** Qwen-2.5-32B (Groq API)
* **Vector Database:** ChromaDB
* **Embeddings:** BAAI/bge-m3
* **Sparse Retrieval:** BM25
* **Data Format:** JSON + LaTeX
