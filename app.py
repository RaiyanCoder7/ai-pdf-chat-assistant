import streamlit as st
import PyPDF2
import os
import google.generativeai as genai

# Configure API key from Streamlit Secrets
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="🤖")
st.title("📘 AI PDF Chat Assistant")
st.write("Upload a PDF and ask questions. AI responds fast using Gemini Lite model.")

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Split text into manageable chunks
def split_text(text, max_len=2000):
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

# Ask AI with chunking and summarization
def ask_ai(pdf_text, question):
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    chunks = split_text(pdf_text)
    summaries = []

    for i, chunk in enumerate(chunks):
        st.write(f"Processing chunk {i+1}/{len(chunks)}...")

        response = model.generate_content(
            request={
                "prompt": [{"type": "text", "text": f"Summarize this text concisely in 50 words:\n{chunk}"}],
                "maxOutputTokens": 200
            }
        )
        summary = response.result[0].content[0].text
        summaries.append(summary)

    combined_summary = "\n".join(summaries)

    final_prompt = f"Based on the text below, answer the question concisely:\n\n{combined_summary}\n\nQuestion: {question}"
    
    response = model.generate_content(
        request={
            "prompt": [{"type": "text", "text": final_prompt}],
            "maxOutputTokens": 300
        }
    )
    answer = response.result[0].content[0].text
    return answer


# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded and processed successfully!")

    question = st.text_input("Ask a question about your PDF:")

    if st.button("Ask"):
        if question.strip():
            with st.spinner("AI is thinking... this will be fast ⏳"):
                answer = ask_ai(pdf_text, question)
                st.subheader("💬 Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")


