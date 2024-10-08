import streamlit as st
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Set Streamlit page configuration at the very beginning
st.set_page_config(page_title="HR Interview Bot", page_icon="🤖")

# Load environment variables and configure Google API
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Whisper model for voice transcription
@st.cache_resource
def load_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

whisper_model = load_whisper_model()

# Function to process uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to generate HR interview questions
def generate_question(vector_store, previous_questions):
    question_prompt = f"""
Given the following context, generate a thoughtful HR interview question that pertains specifically to the candidate's uploaded resume.
 Ensure the question is open-ended, allowing the candidate to elaborate on their skills, experiences, and qualifications. 
 Avoid repeating any of these previously asked questions: {previous_questions}.

The question should effectively gauge the candidate's experience level and alignment with the role's requirements.

Question:
"""
    
    docs = vector_store.similarity_search(question_prompt, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    question = model.predict(context + "\n" + question_prompt)
    
    return question.strip()

# Function to record and transcribe audio
def record_and_transcribe_audio(duration, sample_rate):
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_filename = temp_audio.name
        import scipy.io.wavfile as wavfile
        wavfile.write(temp_filename, sample_rate, recording)
    
    segments, _ = whisper_model.transcribe(temp_filename, beam_size=5)
    os.unlink(temp_filename)
    
    transcription = " ".join([segment.text for segment in segments])
    return transcription

# Main HR Interview Bot function
def hr_interview_bot():
    st.title("HR Interview Bot")
    
    with st.sidebar:
        st.title("Upload PDF")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.vector_store_ready = True
                st.success("Done")

    if 'vector_store_ready' not in st.session_state:
        st.session_state.vector_store_ready = False

    if 'current_question' not in st.session_state:
        st.session_state.current_question = None

    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0

    if 'previous_questions' not in st.session_state:
        st.session_state.previous_questions = []

    if 'answers' not in st.session_state:
        st.session_state.answers = []

    if st.session_state.vector_store_ready:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings)

        if st.session_state.current_question is None:
            st.session_state.current_question = generate_question(vector_store, st.session_state.previous_questions)
            st.session_state.previous_questions.append(st.session_state.current_question)

        st.write(f"Question {st.session_state.question_count + 1}: {st.session_state.current_question}")
        
        if st.button("Start Recording"):
            st.write("Recording...")
            transcription = record_and_transcribe_audio(duration=5, sample_rate=16000)
            st.session_state.answers.append(transcription)
            st.write(f"Your Answer: {transcription}")
        
        if st.button("Next Question"):
            if st.session_state.answers:
                st.session_state.question_count += 1
                if st.session_state.question_count < 5:
                    st.session_state.current_question = generate_question(vector_store, st.session_state.previous_questions)
                    st.session_state.previous_questions.append(st.session_state.current_question)
                else:
                    st.session_state.current_question = None
                st.experimental_rerun()
            else:
                st.warning("Please record an answer before moving to the next question.")

        if st.session_state.question_count >= 5:
            if st.button("Generate Interview Summary"):
                summary_prompt = f"""
Based on the following interview answers, provide a brief report on the candidate's performance, including areas where they excelled, 
areas for improvement, and any inaccuracies or weaknesses in their responses.
{chr(10).join([f"Question: {q}{chr(10)}Answer: {a}{chr(10)}" for q, a in zip(st.session_state.previous_questions, st.session_state.answers)])}
Provide a concise paragraph summarizing the candidate's background, strengths, and potential fit for the role. Assess their proficiency in answering questions,
 the accuracy of their responses, their communication skills, clarity of thought, and problem-solving abilities. 
 Highlight both positive aspects and constructive feedback to help the candidate improve their performance.
"""
                
                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                summary = model.predict(summary_prompt)
                
                st.write("Interview Summary:")
                st.write(summary)

def main():
    hr_interview_bot()

if __name__ == "__main__":
    main()
