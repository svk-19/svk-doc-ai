import google.generativeai as genai
import os

from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# --------------------------------------------------
# CHAT ANSWERS
# --------------------------------------------------

def generate_answer(
    question,
    retrieved_chunks,
    memory_context=""
):

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are SVK Doc AI, a professional AI document assistant.

Your goal is to provide intelligent, professional, and detailed answers using the retrieved document context.

Rules:

1. If the answer exists in the context:
   - Give a complete professional answer.
   - Use bullet points when appropriate.
   - Explain clearly instead of copying raw text.

2. If the answer is partially available:
   - Answer using available information.
   - Mention which details are not explicitly available.

3. If the exact answer is not found:
   - Do NOT invent facts.
   - Explain what the document is mainly discussing.
   - Provide the closest relevant information available.

4. Make responses sound natural and professional.

5. Avoid one-line answers whenever possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    
    
    try:
        response = model.generate_content(
            prompt
        )
        return response.text
    except Exception:
        return (
            "⚠️ Gemini API quota exceeded or temporarily unavailable.\n\n"
            "Please wait a few minutes and try again."
        )


# --------------------------------------------------
# PDF SUMMARY
# --------------------------------------------------

def generate_pdf_summary(text):

    prompt = f"""
You are an expert document analyst.

Read the PDF content below and generate:

1. Executive Summary
2. Key Points
3. Important Facts
4. Action Items (if any)

PDF Content:

{text}

Generate a professional summary.
"""

    try:
        response = model.generate_content(
            prompt
        )
        return response.text
    except Exception:
        
        return (
        "⚠️ Gemini API quota exceeded or temporarily unavailable.\n\n"
        "Please wait a few minutes and try again."
    )