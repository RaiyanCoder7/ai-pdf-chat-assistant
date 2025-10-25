import streamlit as st
import PyPDF2
import os
import google.generativeai as genai

# Configure the Gemini API key from Streamlit secrets
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="🤖")
st.title("📘 AI PDF Chat Assistant")
st.write("Upload a PDF and ask questions — AI will quickly summarize and answer using Gemini 2.5 Flash Lite.")

# Extract text from the uploaded PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Split text into manageable chunks
def split_text(text, max_length=2000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Generate answer from Gemini
def ask_ai(pdf_text, question):
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

    chunks = split_text(pdf_text)
    summaries = []

    for i, chunk in enumerate(chunks):
        st.info(f"Summarizing part {i+1}/{len(chunks)}...")
        summary_prompt = f"Summarize this text briefly in 50 words:\n\n{chunk}"
        summary_response = model.generate_content(summary_prompt)
        summaries.append(summary_response.text)

    combined_summary = "\n".join(summaries)

    final_prompt = f"""
    Based on the summarized text below, answer the question clearly and accurately.

    Summarized Text:
    {combined_summary}

    Question: {question}
    """

    response = model.generate_content(final_prompt)
    return response.text

# Streamlit UI
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded and processed successfully!")

    question = st.text_input("💭 Ask a question about your PDF:")
    if st.button("Ask AI"):
        if question.strip():
            with st.spinner("AI is thinking..."):
                try:
                    answer = ask_ai(pdf_text, question)
                    st.subheader("💬 Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error while generating answer: {e}")
        else:
            st.warning("Please enter a valid question.")
