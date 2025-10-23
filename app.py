import streamlit as st
import PyPDF2
import os
import google.generativeai as genai

# Load API key from Streamlit Secrets
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="🤖")
st.title("📘 AI PDF Chat Assistant")
st.write("Upload a PDF and ask questions about its content using Google Gemini AI.")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to split text into chunks
def split_text(text, max_len=3000):
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

# Function to ask AI (optimized for Cloud)
def ask_ai(pdf_text, question):
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    chunks = split_text(pdf_text)
    summaries = []

    # Summarize each chunk first
    for chunk in chunks:
        prompt = f"Summarize this text concisely:\n\n{chunk}"
        summary = model.generate_content(prompt).text
        summaries.append(summary)

    combined_summary = "\n".join(summaries)
    final_prompt = f"Based on the text below, answer the question:\n\n{combined_summary}\n\nQuestion: {question}"
    answer = model.generate_content(final_prompt).text
    return answer

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded and processed successfully!")

    question = st.text_input("Ask a question about your PDF:")

    if st.button("Ask"):
        if question.strip():
            with st.spinner("AI is thinking... this should be faster ⏳"):
                answer = ask_ai(pdf_text, question)
                st.subheader("💬 Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")
