import streamlit as st
import PyPDF2
import os
from google import genai
from google.genai.types import GenerateContentConfig

# ========== CONFIGURATION ==========
st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="🤖")

st.title("📘 AI PDF Chat Assistant")
st.write("Upload a PDF and ask questions — powered by **Gemini 2.5 Flash** ⚡")

# Initialize Gemini client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in your environment or Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# ========== FUNCTIONS ==========

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, max_len=2000):
    """Split text into smaller chunks for processing."""
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]

def ask_ai(pdf_text, question):
    """Summarize chunks and answer the user's question."""
    chunks = split_text(pdf_text)
    summaries = []

    # Step 1: Summarize each chunk
    for i, chunk in enumerate(chunks):
        st.info(f"Processing chunk {i+1}/{len(chunks)} ...")
        prompt = f"Summarize this text concisely in 60 words:\n\n{chunk}"

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200
                )
            )
            summaries.append(response.text)
        except Exception as e:
            st.warning(f"⚠️ Error summarizing chunk {i+1}: {e}")

    # Step 2: Combine summaries and ask the main question
    combined_summary = "\n".join(summaries)
    final_prompt = f"Based on the summarized content below, answer the question accurately.\n\n{combined_summary}\n\nQuestion: {question}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt,
            config=GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=400
            )
        )
        return response.text
    except Exception as e:
        return f"Error while generating answer: {e}"

# ========== STREAMLIT APP UI ==========

uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded and text extracted successfully!")

    question = st.text_input("💬 Ask a question about your PDF:")

    if st.button("Ask AI"):
        if question.strip():
            with st.spinner("🤔 AI is thinking..."):
                answer = ask_ai(pdf_text, question)
                st.subheader("🧠 AI Answer:")
                st.write(answer)
        else:
            st.warning("Please type a question before clicking 'Ask AI'.")
