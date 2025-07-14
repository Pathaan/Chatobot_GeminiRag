# import streamlit as st
# import time
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate




# from dotenv import load_dotenv
# load_dotenv()


# st.title("RAG Application built on Gemini Model")

# loader = PyPDFLoader("survpaper.pdf")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
# docs = text_splitter.split_documents(data)


# vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0,max_tokens=None,timeout=None)


# query = st.chat_input("Say something: ") 
# prompt = query

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# if query:
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     response = rag_chain.invoke({"input": query})
#     #print(response["answer"])

#     st.write(response["answer"])
import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page config and custom CSS
st.set_page_config(page_title="Chat Bot", page_icon = "🦅", layout="wide")
st.markdown("""
     <style>
        html, body, [class*="css"] {
            background-color: #B22222;
            color: #FFFFFF;
            font-style: italic;
            font-family: 'Segoe UI', sans-serif;
        }

        .custom-logo span {
            color: #FFFFFF;
            font-style: italic;
        }

        .stTextInput > div > div > input,
        .stChatInput {
            background-color: #FFFFFF;
            color: #000000;
            font-style: italic;
        }

        .stMarkdown, .stText, .stChatMessage {
            font-style: italic !important;
            color: #FFFFFF !important;
        }

        .css-1cpxqw2, .css-ffhzg2 {
            background-color: #B22222 !important;
        }

        /* Remove shadows and borders for clean look */
        .stTextInput, .stChatInput {
            box-shadow: none !important;
            border: none !important;
        }
    </style>

    <div class="custom-logo">
        <img src="eagle_logo.png" alt="🦅" style="width: 40px; height: 40px;">
        <span>Chatbot</span>
    </div>
""", unsafe_allow_html=True)

st.title("📄💬 RAG Chatbot with Gemini")

# Chat history setup
if "messages" not in st.session_state:
    st.session_state["messages"] = []

uploaded_file = st.file_uploader("📎 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 Processing the uploaded document..."):
        # Save uploaded file
        with open("temp_uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load and split the PDF
        loader = PyPDFLoader("temp_uploaded_file.pdf")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)

        # Embed and index the documents
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # Initialize Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=100
        )

        # Chat prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # Chat input
        query = st.chat_input("💬 Ask something about the document...")

        if query:
            # Save user query
            st.session_state["messages"].append({"role": "user", "content": query})

            with st.spinner("🤖 Generating response..."):
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                try:
                    response = rag_chain.invoke({"input": query})
                    answer = response["answer"]

                    # Save assistant response
                    st.session_state["messages"].append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})

        # Display chat history
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f"🗣️ **You**: {msg['content']}")
            else:
                st.markdown(f"💡 **Bot**: {msg['content']}")

else:
    st.info("👆 Please upload a PDF document to begin.")
