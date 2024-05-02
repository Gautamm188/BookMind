import streamlit as st
import pdfplumber
from transformers import pipeline

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def main():
    st.title("BookMind")
    pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
    if pdf_file is not None:
        st.write("PDF uploaded successfully!")
        text = extract_text_from_pdf(pdf_file)
        st.write("Text extracted from PDF:")
        st.write(text)
        
        question = st.text_input("Ask your question:")
        if st.button("Get Answer"):
            qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            answer = qa_model(question=question, context=text)
            st.write("Answer:", answer['answer'])

if __name__ == "__main__":
    main()
