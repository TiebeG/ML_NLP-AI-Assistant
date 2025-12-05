# backend/graph_ml_assistant.py

import os
import sys
import traceback

print("DEBUG: Importing graph_ml_assistant.py")

# -------------------------------------------------------
# Ensure project root is on Python path
# -------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"DEBUG: Added project root to sys.path -> {PROJECT_ROOT}")

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------
# Imports
# -------------------------------------------------------

try:
    from typing_extensions import TypedDict
    from typing import List
    from langchain_core.messages import (
        AnyMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
    )
    from langchain_groq import ChatGroq
    from langgraph.graph import StateGraph, END

    # Internal backend imports
    from backend.tools_rag import course_docs_search
    from backend.router_agent import classify_query
    from backend.quiz_agent import generate_quiz

except Exception as e:
    print("ERROR: Failed during imports in graph_ml_assistant.py")
    traceback.print_exc()
    raise


# -------------------------------------------------------
# Graph State Definition
# -------------------------------------------------------

class GraphState(TypedDict):
    messages: List[AnyMessage]
    route: str | None
    chapter: str | None


# -------------------------------------------------------
# LLM for teacher nodes
# -------------------------------------------------------

llm_teacher = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
)


# -------------------------------------------------------
# Router Node
# -------------------------------------------------------

def router_node(state: GraphState) -> GraphState:
    last_input = state["messages"][-1].content
    route_info = classify_query(last_input)

    state["route"] = route_info.get("type")
    state["chapter"] = route_info.get("chapter")

    print(f"DEBUG: Router decided route={state['route']} chapter={state['chapter']}")
    return state


# -------------------------------------------------------
# Teacher RAG Node
# -------------------------------------------------------

def teacher_rag_node(state: GraphState) -> GraphState:
    user_msg = state["messages"][-1].content

    rag_context = course_docs_search.invoke(user_msg)

    system_msg = SystemMessage(
        content=(
            "Use the following course excerpts to answer.\n\n"
            f"{rag_context}\n"
        )
    )

    result = llm_teacher.invoke([system_msg] + state["messages"])
    state["messages"].append(result)

    print("DEBUG: teacher_rag_node executed.")
    return state


# -------------------------------------------------------
# Teacher General Node
# -------------------------------------------------------

def teacher_general_node(state: GraphState) -> GraphState:
    system_msg = SystemMessage(
        content=(
            "You are a Machine Learning teaching assistant. "
            "Explain clearly with examples, without using course documents."
        )
    )

    result = llm_teacher.invoke([system_msg] + state["messages"])
    state["messages"].append(result)

    print("DEBUG: teacher_general_node executed.")
    return state


# -------------------------------------------------------
# Quiz Node
# -------------------------------------------------------

def quiz_node(state: GraphState) -> GraphState:
    chapter = state.get("chapter")
    quiz = generate_quiz(chapter=chapter, n_questions=5)

    ai_msg = AIMessage(content=quiz)
    state["messages"].append(ai_msg)

    print(f"DEBUG: quiz_node executed (chapter={chapter})")
    return state


# -------------------------------------------------------
# Build LangGraph
# -------------------------------------------------------

def build_graph():
    print("DEBUG: Building graph...")

    builder = StateGraph(GraphState)

    builder.add_node("router", router_node)
    builder.add_node("teacher_rag", teacher_rag_node)
    builder.add_node("teacher_general", teacher_general_node)
    builder.add_node("quiz", quiz_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        lambda state: state.get("route"),
        {
            "rag_query": "teacher_rag",
            "general_explanation": "teacher_general",
            "quiz_request": "quiz",
        }
    )

    builder.add_edge("teacher_rag", END)
    builder.add_edge("teacher_general", END)
    builder.add_edge("quiz", END)

    graph = builder.compile()
    print("DEBUG: Graph successfully built.")
    return graph


# -------------------------------------------------------
# Create graph_app with full error handling
# -------------------------------------------------------

try:
    graph_app = build_graph()
    print("DEBUG: graph_app CREATED SUCCESSFULLY")
except Exception as e:
    print("ERROR: Failed to build graph_app")
    traceback.print_exc()
    graph_app = None
