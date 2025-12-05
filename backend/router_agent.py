# backend/router_agent.py

import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

router_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0  # deterministic
)

def extract_chapter(query: str):
    """
    Extracts chapter like:
    1, 2, 3, 1.1, 1.2, 2.5 etc.
    """
    match = re.search(r"\b(\d+\.\d+|\d+)\b", query)
    return match.group(1) if match else None


def classify_query(query: str) -> dict:
    """
    Classifies the query and extracts chapter if needed.
    Returns:
        {
            "type": "rag_query" | "general_explanation" | "quiz_request",
            "chapter": "1" | "2.3" | None
        }
    """

    # Quiz intent detection (simple rule-based)
    quiz_keywords = ["quiz", "questions", "test me", "practice", "exam"]
    is_quiz = any(word in query.lower() for word in quiz_keywords)

    if is_quiz:
        chapter = extract_chapter(query)
        return {"type": "quiz_request", "chapter": chapter}

    # Otherwise use the LLM to classify
    system = SystemMessage(
        content=(
            "Classify the user's message strictly as one of:\n"
            "1) rag_query → asks about course content or slides\n"
            "2) general_explanation → asks for conceptual explanation\n\n"
            "Respond with exactly one label."
        )
    )

    result = router_llm.invoke([system, HumanMessage(content=query)])
    label = result.content.strip().lower()

    return {"type": label, "chapter": None}
