import streamlit as st
import os
import tempfile
import subprocess
from datetime import datetime
import pickle
import chromadb

from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
# V-- Make sure 'langchain-chroma' is in your requirements.txt
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# ----------------------
# CONFIG
# ----------------------
# Make sure to set GOOGLE_API_KEY in your Render environment variables/secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Path to vectorstore (we'll fetch from GitHub if missing)
def fetch_vectorstore_from_github():
    temp_dir = os.path.join(tempfile.gettempdir(), "vectorstore")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        repo_url = "https://github.com/Zenngeist/DITU_CHATBOT.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, temp_dir],
            check=True
        )
        subprocess.run(
            ["git", "-C", temp_dir, "sparse-checkout", "set", "vectorstore/"],
            check=True
        )
    # The actual vectorstore files will be inside a 'vectorstore' subdirectory
    return os.path.join(temp_dir, "vectorstore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vectorstore")
DOC_STORE_FILE_PATH = os.path.join(BASE_DIR, "docstore.pkl")

# If the vectorstore directory doesn't exist locally, fetch it from the sparse checkout
if not os.path.exists(VECTOR_STORE_PATH):
    print(f"Local vectorstore not found at {VECTOR_STORE_PATH}, fetching from GitHub...")
    VECTOR_STORE_PATH = fetch_vectorstore_from_github()
    print(f"Vectorstore fetched to temporary directory: {VECTOR_STORE_PATH}")


# ----------------------
# LOAD RAG CHAIN
# ----------------------
@st.cache_resource
def load_advanced_rag_chain():
    print("Running Multi-Query RAG Chain Setup")

    # Load docstore
    try:
        # If the docstore isn't local, it should be in the parent of the temp vectorstore dir
        docstore_path = DOC_STORE_FILE_PATH
        if "temp" in VECTOR_STORE_PATH:
             docstore_path = os.path.join(os.path.dirname(VECTOR_STORE_PATH), "docstore.pkl")

        with open(docstore_path, "rb") as f:
            raw_docstore = pickle.load(f)
        store = InMemoryStore()
        store.mset(raw_docstore.items()) # Use mset for bulk loading
        print("  ✓ Parent document store loaded.")
    except FileNotFoundError:
        raise FileNotFoundError(f"'{docstore_path}' not found. Ensure docstore.pkl is in your repo.")

    # Embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    # Chroma vectorstore (local) - New Method
    # 1. Create a persistent client pointing to the directory
    persistent_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    # 2. Load the vector store using the client and existing collection name
    vector_store = Chroma(
        client=persistent_client,
        collection_name="final_retrieval_system",
        embedding_function=embedding_model,
    )
    print("  ✓ Chroma vector store loaded with new client method.")

    # Text splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # Retriever
    base_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    print("  ✓ Base ParentDocumentRetriever reconstructed.")

    # Multi-query retriever
    # CHANGED MODEL to 'gemini-pro' to resolve 404 Not Found error.
    llm_for_queries = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash,
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_for_queries
    )
    print("  ✓ Multi-Query Retriever ready.")

    # Prompts
    condense_question_template = (
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\n"
        "Chat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
    )
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

    qa_prompt_template = """
You are a helpful and precise university assistant chatbot for DIT University.

- Use ONLY the information in the provided Context when answering factual queries about schedules, policies, dates, events, timetables, or any document-specific data.
- Do NOT invent factual claims that are not present in the Context.
- When using the Context, rephrase and paraphrase, avoid copying >8 consecutive words verbatim.

- For problem-solving requests, show concise step-by-step reasoning (3-6 lines).

- If info not found in Context, reply exactly:
"I searched through all available documents, but I could not find information on that topic. Please check the official DIT University website or ERP portal."

Context:
{context}

Question: {question}

Helpful Answer:
"""
    QA_PROMPT = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

    # CHANGED MODEL to 'gemini-pro' to resolve 404 Not Found error.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        google_api_key=GOOGLE_API_KEY
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=multi_query_retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    print("RAG Chain Setup Complete")
    return qa_chain

# ----------------------
# STREAMLIT UI
# ----------------------
st.set_page_config(page_title="DIT University AI Assistant", page_icon="🎓")
st.title("DIT University AI Assistant")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Please add it to your environment secrets.")
    st.stop()

qa_chain = load_advanced_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey there! I'm here to help with anything DIT University related."}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Quick actions / Kickstarter questions
def handle_query(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Searching documents..."):
        try:
            response = qa_chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state.chat_history.extend([
                HumanMessage(content=prompt),
                AIMessage(content=answer)
            ])
        except Exception as e:
            error_message = f"Oof, something went wrong: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.write(error_message)

if 'kickstarter_used' not in st.session_state:
    st.session_state.kickstarter_used = False

if not st.session_state.kickstarter_used:
    st.markdown("**Quick actions — click to ask a question**")
    cols = st.columns(2)
    kickstarter_questions = [
        ("What is the Mid term schedule?", "What is the Mid term schedule?"),
        ("When is Youthopia fest?", "When is Youthopia fest?"),
        ("Time Table for Today", f"What is the timetable for {datetime.now().strftime('%A')}?"),
        ("What's on the mess menu?", f"What food is available in mess on {datetime.now().strftime('%A')}?")
    ]
    for i, (label, actual_prompt) in enumerate(kickstarter_questions):
        with cols[i % 2]:
            if st.button(label, use_container_width=True):
                st.session_state.kickstarter_used = True
                handle_query(actual_prompt)
                st.rerun()

# Chat input
if prompt := st.chat_input("Ask me something about the university..."):
    st.session_state.kickstarter_used = True
    handle_query(prompt)

