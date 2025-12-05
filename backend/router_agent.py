# backend/router_agent.py

import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# -------------------------------------------------------
# Router LLM (deterministic)
# -------------------------------------------------------
router_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0
)

# -------------------------------------------------------
# Helper — Extract chapter numbers like 1, 2.1, 3.4, etc.
# -------------------------------------------------------
def extract_chapter(query: str):
    match = re.search(r"\b(\d+\.\d+|\d+)\b", query)
    return match.group(1) if match else None


# -------------------------------------------------------
# Helper — Clean LLM output to valid route strings
# -------------------------------------------------------
def clean_label(raw: str) -> str:
    # Lowercase
    label = raw.strip().lower()

    # Remove numbering like "1) rag_query"
    label = re.sub(r"^\s*\d+[\.\)\:\-]\s*", "", label)

    # Remove bullet-list formatting
    label = label.replace("*", "").strip()

    # Normalize known patterns
    if "rag" in label or "document" in label or "course" in label:
        return "rag_query"

    if "general" in label or "explanation" in label:
        return "general_explanation"

    if "quiz" in label or "test" in label or "practice" in label:
        return "quiz_request"

    # Fallback → general explanation
    return "general_explanation"


# -------------------------------------------------------
# Main router function (very robust)
# -------------------------------------------------------
def classify_query(query: str) -> dict:
    """
    Returns clean dict:
        {
            "type": "rag_query" | "general_explanation" | "quiz_request",
            "chapter": "1" | "2.5" | None
        }
    """

    lower = query.lower()

    # ---------------------------------------------------
    # 1) QUIZ INTENT (rule-based → fastest + safest)
    # ---------------------------------------------------
    quiz_keywords = ["quiz", "questions", "test me", "practice", "exam"]
    if any(k in lower for k in quiz_keywords):
        chapter = extract_chapter(query)
        return {"type": "quiz_request", "chapter": chapter}

    # ---------------------------------------------------
    # 2) LLM classification
    # ---------------------------------------------------
    system = SystemMessage(
        content=(
            "Classify the user's message strictly as one of:\n"
            "- rag_query   (asks about course documents/slides)\n"
            "- general_explanation   (asks for ML explanation)\n\n"
            "Respond with EXACTLY one label, no numbering, no punctuation."
        )
    )

    result = router_llm.invoke([system, HumanMessage(content=query)])
    raw_label = result.content.strip()

    # Clean & normalize
    cleaned = clean_label(raw_label)

    # Always return correct structure
    return {"type": cleaned, "chapter": extract_chapter(query)}
