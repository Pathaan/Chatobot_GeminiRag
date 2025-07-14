# Chatbot


* PDF-based RAG
* Google Gemini
* Chroma vector DB
* Mars-colored UI
* Eagle logo branding


### ✅ `README.md`

````markdown
# 🦅 *Chatbot* — RAG App using Google Gemini + LangChain

A Mars-themed, eagle-branded **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**, using **Google Gemini** via LangChain, and a local vectorstore powered by **Chroma**.

---

## 🚀 Features

- 🧠 **RAG with Gemini API** — Question answering from custom PDF files
- 📄 **PDF Upload** — Drop in your own document
- 📚 **Chroma Vectorstore** — Fast, local retrieval from embedded chunks
- 🎨 **Mars Red Theme** — Custom background and eagle logo UI
- 💬 **Chat History** — Remembers your conversation per session

---

## 🖥️ Demo UI

![screenshot](example.png)  
*Eagle-themed, Mars-colored Streamlit chatbot with memory and file upload*

---

## 📦 Requirements

Make sure you have Python 3.9+ and install dependencies:

```bash
pip install -r requirements.txt
````

Create a `.env` file for your Google Gemini key:

```
GOOGLE_API_KEY=your_google_generative_ai_key
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

---

## 🌍 Deployment Options

* [Hugging Face Spaces](https://huggingface.co/spaces) – easiest for Streamlit apps
* [Render](https://render.com) – scalable & custom domains
* [Streamlit Community Cloud](https://streamlit.io/cloud)

Need deployment help? [Ask here](#) or [contact me](https://github.com/Pathaan).

---

## 📜 License

MIT License – use, modify, and fly freely like an eagle 🦅

---

## 🧑‍💻 Author

**Md Shahrukh**
Actuarial Analyst | Data Scientist | ISI Kolkata
GitHub: [@Pathaan](https://github.com/Pathaan)

---
