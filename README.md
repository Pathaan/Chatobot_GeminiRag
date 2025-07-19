

# 🦅 *Chatbot* — RAG App using Google Gemini + LangChain

A Mars-themed, eagle-branded **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**, using **Google Gemini** via LangChain, and a local vectorstore powered by **Chroma**.


**Retrieval-Augmented Generation (RAG)** is a method that improves large language models (LLMs), like ChatGPT, by allowing them to fetch and use information from external sources **before** generating a response. This helps the model give **more accurate, current, and domain-specific answers**.

## 🚀 Why Use RAG?

Traditional LLMs only rely on what they learned during training, which means they can be outdated or miss domain-specific knowledge (like company data, legal updates, or scientific research). RAG solves this by allowing the model to **search and read documents in real-time** to support its responses.

For example, a chatbot using RAG can:

* Refer to internal company documents
* Look up updated legal policies
* Avoid “hallucinations” (making up fake info)

---

## 🔍 How RAG Works

RAG follows a 4-step process:

### 1. **Indexing**

* The first step is to prepare the documents the model might need later.
* These documents are broken into chunks and converted into **embeddings** (numerical representations).
* These embeddings are stored in a **vector database** for quick search later.

### 2. **Retrieval**

* When a user asks a question, the system searches this database to find the most relevant pieces of text.

### 3. **Augmentation**

* The retrieved text is added (or "stuffed") into the model’s input prompt.
* This gives the model more context before answering.

### 4. **Generation**

* The model uses both the **user’s question** and the **retrieved information** to create a response.

---

## 💡 Real Benefits of RAG

* **More Accurate**: Because it pulls from real, updated sources
* **Less Hallucination**: Fewer made-up facts
* **No Need to Retrain**: Updating the documents is cheaper than retraining the entire model
* **Verifiable Sources**: The model can cite where it got its information

---

## 🧠 Limitations of RAG

Even with RAG, the model can still:

* Misunderstand the context of correct information
* Blend outdated and updated info incorrectly
* Pull data from biased or misleading sources
* Hallucinate around the source material
* Provide wrong answers instead of saying “I don’t know”

---

## 🛠️ Advanced Features in RAG

### 🔧 Encoder Improvements

* Converts text into dense (meaning-focused) or sparse (word-focused) vectors
* Use of better math like **dot products** and **ANN (approximate nearest neighbors)** for faster and more accurate retrieval

### 📑 Chunking

How documents are broken up affects what the model can retrieve:

* **Fixed-size chunks** with overlaps
* **Sentence-based chunks** using tools like spaCy or NLTK
* **File-based chunks** (e.g., for PDFs, code, HTML)

### 🔗 Knowledge Graphs

Instead of raw documents, you can use **graphs** of connected facts. These can be stored and retrieved more precisely and are sometimes called **GraphRAG**.

### 🔍 Hybrid Search

Combines:

* **Vector search** (meaning-based)
* **Keyword search** (exact text match)
  This improves recall when one method alone isn’t enough.

---

## 📊 Evaluation & Benchmarks

RAG systems are tested using:

* **BEIR** (info retrieval tasks)
* **Natural Questions**, **Google QA**
* **LegalBench-RAG** (for legal document QA)

These tests check both the **accuracy of retrieved documents** and the **quality of generated answers**.


## Summary

| Feature             | With RAG     | Without RAG     |
| ------------------- | ------------ | --------------- |
| Updated Info        | ✅ Yes        | ❌ No            |
| Fact-Checking       | ✅ Possible   | ❌ Often missing |
| Hallucinations      | 🔻 Reduced   | 🔺 Common       |
| Retraining Needed   | ❌ Less often | ✅ Frequently    |
| Domain-Specific Use | ✅ Easy       | ❌ Difficult     |

---

## 🚀 Features

- 🧠 **RAG with Gemini API** — Question answering from custom PDF files
- 📄 **PDF Upload** — Drop in your own document
- 📚 **Chroma Vectorstore** — Fast, local retrieval from embedded chunks
- 🎨 **Mars Red Theme** — Custom background and eagle logo UI
- 💬 **Chat History** — Remembers your conversation per session



## 🖥️ Demo UI

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f84ed281-580f-4940-996a-a122d5acd0b6" />
  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/be101518-9a27-49fd-9f8f-b7a9311cff63" />

*Eagle-themed, Mars-colored Streamlit chatbot with memory and file upload*



## 📦 Requirements

Make sure you have Python 3.9+ and install dependencies:

```bash
pip install -r requirements.txt
````

Create a `.env` file for your Google Gemini key:
```
python -m venv myenv
```
```
GOOGLE_API_KEY=your_google_generative_api_key
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

## 🧠 How it Works

1. Upload a PDF (e.g. research paper or textbook)
2. Text is chunked and embedded using `GoogleGenerativeAIEmbeddings`
3. Stored in ChromaDB for vector search
4. On query, LangChain retrieves top relevant chunks
5. Gemini answers your question using the context

---

## 🛠️ Tech Stack

| Layer         | Tech                         |
| ------------- | ---------------------------- |
| UI            | Streamlit                    |
| LLM           | Google Gemini via LangChain  |
| Embeddings    | GoogleGenerativeAIEmbeddings |
| Vectorstore   | Chroma (in-memory)           |
| Document Load | LangChain PDF Loader         |
| Framework     | LangChain + Streamlit        |

---

## 📁 Project Structure

```
geminirag/
├── app.py
├── eagle_logo.png
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```



## 📜 License

BSD 3-Clause License – use, modify, and fly freely like an eagle 🦅

---

## 🧑‍💻 Author

**Md Shahrukh**
[@Linkedin](https://linkedin.com/in/md-shahrukh-locky/)
Actuarial Analyst | Data Scientist | ISI Kolkata
GitHub: [@Pathaan](https://github.com/Pathaan)

---
