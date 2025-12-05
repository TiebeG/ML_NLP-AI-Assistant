import os
import sys
import uuid
import datetime
import streamlit as st

# ----------------------------------------------------
# Ensure backend import works
# ----------------------------------------------------
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.graph_ml_assistant import graph_app
from langchain_core.messages import HumanMessage, AIMessage


# ----------------------------------------------------
# Session Init
# ----------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "rename_id" not in st.session_state:
    st.session_state.rename_id = None


# ----------------------------------------------------
# Helper: Auto title generation
# ----------------------------------------------------
def auto_title_from_text(text):
    words = text.strip().split()
    title = " ".join(words[:6]).title()
    return title if title else "New Chat"


# ----------------------------------------------------
# Create New Chat
# ----------------------------------------------------
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "name": "New Chat",
        "messages": [],
        "last_updated": datetime.datetime.now().timestamp(),
    }
    st.session_state.current_chat = chat_id


# ----------------------------------------------------
# ✨ GLOBAL CSS — Fix sidebar & layout cleanly
# ----------------------------------------------------
st.markdown("""
<style>
/* Wider sidebar */
section[data-testid="stSidebar"] {
    width: 320px !important;
    background-color: #1f1f23 !important;
    padding: 6px 12px;
}

/* Chat row */
.chat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}

/* Chat button */
.chat-name-btn button {
    width: 100% !important;
    height: 32px !important;     /* Smaller */
    padding: 0 10px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}

/* ICON BUTTONS — clean and aligned */
.icon-btn button {
    width: 30px !important;
    height: 30px !important;
    padding: 0 !important;
    font-size: 16px !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Scroll container */
.chat-list {
    max-height: 65vh;
    overflow-y: auto;
    padding-right: 6px;
}

/* Hide scrollbar */
.chat-list::-webkit-scrollbar {
    width: 4px;
}
.chat-list::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True
)


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:

    st.markdown("## 💬 ML Assistant")

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.markdown("---")
    st.markdown("### Chats")

    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda item: item[1]["last_updated"],
        reverse=True,
    )

    # Scrollable container
    st.markdown('<div class="chat-list">', unsafe_allow_html=True)

    for chat_id, chat_data in sorted_chats:
        st.markdown('<div class="chat-row">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([8, 1, 1])

        # Chat name button (ellipsis)
        with col1:
            st.markdown('<div class="chat-name-btn">', unsafe_allow_html=True)
            if st.button(chat_data["name"], key=f"select_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Rename
        with col2:
            st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
            if st.button("✏️", key=f"rename_{chat_id}", help="Rename chat"):
                st.session_state.rename_id = chat_id
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Delete
        with col3:
            st.markdown('<div class="icon-btn">', unsafe_allow_html=True)
            if st.button("❌", key=f"delete_{chat_id}", help="Delete chat"):
                del st.session_state.chats[chat_id]
                if st.session_state.current_chat == chat_id:
                    st.session_state.current_chat = None
                st.session_state.rename_id = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Rename modal
    if st.session_state.rename_id:
        st.markdown("---")
        edit_id = st.session_state.rename_id
        current_name = st.session_state.chats[edit_id]["name"]

        new_name = st.text_input("Rename Chat", value=current_name)
        save, cancel = st.columns(2)

        with save:
            if st.button("Save"):
                if new_name.strip():
                    st.session_state.chats[edit_id]["name"] = new_name.strip()
                st.session_state.rename_id = None
                st.rerun()

        with cancel:
            if st.button("Cancel"):
                st.session_state.rename_id = None
                st.rerun()


# ----------------------------------------------------
# MAIN CHAT WINDOW
# ----------------------------------------------------
if st.session_state.current_chat is None:

    st.markdown("### 👋 Welcome!")
    st.markdown(
        "Start by creating a new chat from the sidebar using **➕ New Chat**.\n\n"
        "Your ML assistant is ready to help with course topics, quizzes, explanations, and more."
    )
    st.stop()


# If a chat *is* selected:
chat_id = st.session_state.current_chat
chat_data = st.session_state.chats[chat_id]
messages = chat_data["messages"]

# Chat title
st.markdown(f"## {chat_data['name']}")


# Show previous messages
for msg in messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ----------------------------------------------------
# INPUT HANDLING
# ----------------------------------------------------
user_input = st.chat_input("Ask anything from the ML course...")

if user_input:

    # Auto rename chat on first message
    if chat_data["name"] == "New Chat" and len(messages) == 0:
        st.session_state.chats[chat_id]["name"] = auto_title_from_text(user_input)

    messages.append({"role": "user", "content": user_input})

    graph_state = {
        "messages": [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in messages
        ],
        "route": None,
        "chapter": None
    }

    result = graph_app.invoke(graph_state)
    ai_response = result["messages"][-1].content

    messages.append({"role": "assistant", "content": ai_response})

    # Move chat to top
    st.session_state.chats[chat_id]["last_updated"] = datetime.datetime.now().timestamp()

    st.rerun()
