import os
import sqlite3
from typing import List, Tuple

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.genai as genai


# ----------------- CONFIG -----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_PATH = os.getenv("ATTENDANCE_DB_PATH", "data/attendance.db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

client = genai.Client(api_key=GOOGLE_API_KEY)

# ----------------- EMBEDDINGS -----------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    return SentenceTransformer(model_name)


# ----------------- DATABASE -----------------
def get_attendance_rows(db_path: str) -> List[Tuple[str, str, str]]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            cur.execute("SELECT student_name, date, time FROM attendance")
            rows = cur.fetchall()
            data = [(r[0], str(r[1]), f"time:{r[2]}") for r in rows]
        except:
            cur.execute("SELECT student_name, date, status FROM attendance")
            rows = cur.fetchall()
            data = [(r[0], str(r[1]), str(r[2])) for r in rows]
        conn.close()
        return data
    except:
        return []


def format_rows_for_rag(rows):
    return [f"Student: {s} | Date: {d} | Status: {st}" for s, d, st in rows]


# ----------------- VECTOR STORE -----------------
@st.cache_resource(show_spinner=False)
def build_faiss_index(texts, _embedder):
    if not texts:
        dim = 384
        return faiss.IndexFlatIP(dim), np.zeros((0, dim), dtype=np.float32)

    embeddings = _embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index, embeddings


def retrieve(query, texts, index, embedder, k=5):
    if not texts:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    _, I = index.search(q_emb, min(k, len(texts)))
    return [texts[i] for i in I[0] if 0 <= i < len(texts)]


# ----------------- GEMINI -----------------
def ask_gemini(query, retrieved_data):
    context = "\n".join(retrieved_data[:5]) if retrieved_data else "No records found."

    prompt = f"""
You are an attendance assistant.
Answer ONLY using the records below.

Records:
{context}

Question:
{query}
"""

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"LLM error: {e}"


# ----------------- STREAMLIT UI -----------------
def render_chatbot():
    st.title("🤖 Smart Attendance Chatbot")

    embedder = load_embedder(EMBEDDING_MODEL_NAME)

    if "rag_ready" not in st.session_state:
        rows = get_attendance_rows(DB_PATH)
        texts = format_rows_for_rag(rows)
        index, _ = build_faiss_index(texts, embedder)
        st.session_state.texts = texts
        st.session_state.index = index
        st.session_state.rag_ready = True

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask about attendance")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        retrieved = retrieve(q, st.session_state.texts, st.session_state.index, embedder)
        ans = ask_gemini(q, retrieved)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()
