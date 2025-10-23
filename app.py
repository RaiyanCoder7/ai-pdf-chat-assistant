import streamlit as st
import PyPDF2
import os
import google.generativeai as genai

# Configure API key from Streamlit Secrets
# Make sure you set your GOOGLE_API_KEY in Streamlit Secrets
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Streamlit page settings
st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="🤖")
st.title("📘 AI PDF Chat Assistant")
st.write("Upload a PDF and ask questions. AI responds fast using Gemini Lite model.")

# -----------------------------
# Function to extract text from PDF
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# -----------------------------
# Function to split text into chunks
# -----------------------------
def split_text(text, max_len=2000):
    """
    Splits text into chunks of max_len characters for faster AI processing.
    """
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

# -----------------------------
# Function to ask AI with chunking and summarization
# -----------------------------
def ask_ai(pdf_text, question):
    """
    Summarizes chunks and answers the user's question based on the PDF content.
    """
    chunks = split_text(pdf_text)
    summaries = []

    # Summarize each chunk
    for i, chunk in enumerate(chunks):
        st.write(f"Processing chunk {i+1}/{len(chunks)}...")
        response = genai.generate_text(
            model="gemini-2.5-flash-lite",
            prompt=f"Summarize this text concisely in 50 words:\n{chunk}",
            temperature=0.2,
            max_output_tokens=200
        )
        summaries.append(response.text)

    combined_summary = "\n".join(summaries)

    # Answer the question based on combined summaries
    final_prompt = f"Based on the text below, answer the question concisely:\n\n{combined_summary}\n\nQuestion: {question}"
    response = genai.generate_text(
        model="gemini-2.5-flash-lite",
        prompt=final_prompt,
        temperature=0.2,
        max_output_tokens=300
    )
    return response.text

# -----------------------------
# Streamlit UI
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    if pdf_text.strip() == "":
        st.error("❌ Could not extract text from this PDF.")
    else:
        st.success("✅ PDF uploaded and processed successfully!")

        question = st.text_input("Ask a question about your PDF:")

        if st.button("Ask"):
            if question.strip():
                with st.spinner("AI is thinking... this will be fast ⏳"):
                    try:
                        answer = ask_ai(pdf_text, question)
                        st.subheader("💬 Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error while generating answer: {e}")
            else:
                st.warning("Please enter a question.")
