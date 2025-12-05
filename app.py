# app.py

import os
import sys

# ============================================================
# FIX IMPORT PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.graph_ml_assistant import graph_app

# ============================================================
# STREAMLIT UI
# ============================================================
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


st.set_page_config(page_title="ML Course Assistant", page_icon="🤖")

st.title("🤖 ML Course Assistant (Groq Powered)")
st.write("Ask any ML question!")


# ----------------- SESSION STATE -----------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------- DISPLAY CHAT HISTORY -----------------

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)


# ----------------- USER INPUT -----------------

user_input = st.chat_input("Ask your question...")

if user_input:

    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Prepare state for LangGraph
    state = {"messages": st.session_state.messages}

    # Call LangGraph agent
    result_state = graph_app.invoke(state)

    # Update conversation
    st.session_state.messages = result_state["messages"]

    # Force Streamlit to rerender immediately
    st.rerun()
