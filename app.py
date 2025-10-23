import streamlit as st
import google.generativeai as genai
import PyPDF2
import os

# Load the API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to chat with Gemini
def ask_ai(pdf_text, question):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"The user uploaded a PDF. Based on the text below, answer their question.\n\nPDF Content:\n{pdf_text}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="AI PDF Chat Assistant", page_icon="🤖")
st.title("📘 AI PDF Chat Assistant")
st.write("Upload a PDF and ask questions about its content using Google Gemini AI.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded and processed successfully!")

    question = st.text_input("Ask a question about your PDF:")

    if st.button("Ask"):
        if question.strip():
            with st.spinner("Thinking... 🤔"):
                answer = ask_ai(pdf_text, question)
                st.subheader("💬 Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")


