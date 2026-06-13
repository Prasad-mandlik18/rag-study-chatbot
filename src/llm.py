import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load model
model = genai.GenerativeModel("gemini-2.5-flash")


def generate_rag_answer(question, context_chunks):
    """
    Generate an answer using retrieved chunks.
    If no chunks are available, use Gemini's general knowledge.
    """

    # Fallback: no retrieved context
    if not context_chunks:
        prompt = f"""
You are a helpful AI assistant.

Answer the following question using your own knowledge.

Question:
{question}
"""
        response = model.generate_content(prompt)
        return "🌐 **Answer from Gemini (General Knowledge):**\n\n" + response.text

    # RAG mode
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI Study Assistant.

Use ONLY the provided context to answer the question.

If the context does not contain enough information, reply exactly:
NOT_FOUND

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)
    answer = response.text.strip()

    # Fallback if Gemini says the context isn't enough
    if "NOT_FOUND" in answer:
        fallback_prompt = f"""
You are a helpful AI assistant.

Answer the following question using your own knowledge.

Question:
{question}
"""
        fallback_response = model.generate_content(fallback_prompt)
        return "🌐 **Answer from Gemini (General Knowledge):**\n\n" + fallback_response.text

    return "📚 **Answer from Uploaded Study Material:**\n\n" + answer