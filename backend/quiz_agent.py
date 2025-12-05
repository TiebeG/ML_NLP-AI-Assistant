from multiprocessing import pool
import json, random, os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

DISCUSSION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "course_materials", "discussion_topics.json"
)

with open(DISCUSSION_PATH, "r", encoding="utf-8") as f:
    TOPICS = json.load(f)

quiz_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
)

def generate_quiz(chapter=None, n_questions=5):
    """
    Create a quiz from discussion topics.
    chapter="1" gives all 1.x topics.
    chapter="1.3" gives only topic 1.3.
    """
    pool = TOPICS

    if chapter:
        pool = [t for t in TOPICS if t["id"].startswith(chapter)]

    if not pool:
        return f"No topics found for chapter {chapter}"

    if chapter:
        # keep all topics for that chapter, do NOT sample or shuffle
        chosen = pool  
    else:
        # random quiz when no chapter specified
        chosen = random.sample(pool, min(n_questions, len(pool)))


    topic_block = "\n\n".join([f"{t['id']} | {t['question']}" for t in chosen])

    system = SystemMessage(
        content=(
            "You are an ML course quiz generator.\n\n"
            "IMPORTANT FORMATTING RULES (YOU MUST FOLLOW THESE EXACTLY):\n"
            "1. Each topic must appear in the exact order provided.\n"
            "2. For each topic, output:\n"
            "   ## Topic {id} — {short_summary}\n\n"
            "3. Then output a multiple-choice question using THIS EXACT FORMAT:\n"
            "   **Multiple Choice Question:**\n"
            "   ```mcq\n"
            "   A) option text\n"
            "   B) option text\n"
            "   C) option text\n"
            "   D) option text\n"
            "   ```\n\n"
            "   (ALL answer options MUST be inside the code block, each on their own line.)\n"
            "   (Do NOT place options on the same line. Ever.)\n\n"
            "4. Then output:\n"
            "   **Reflection:** <question>\n\n"
            "5. Add a blank line after each topic.\n"
            "6. Do NOT reorder topics.\n"
            "7. Do NOT add explanations, only questions.\n"
        )
    )

    human = HumanMessage(content=topic_block)

    return quiz_llm.invoke([system, human]).content
