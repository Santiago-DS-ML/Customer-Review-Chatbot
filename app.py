# imports
import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

# config_page
st.set_page_config(
    page_title="ReviewBot",
    layout="wide"
)

st.title("🤖 ReviewBot")
st.subheader("Customer Review RAG Chatbot")

#GEMINI CONFIG
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

llm = genai.GenerativeModel("gemini-2.5-flash")

#LOAD EMBEDDING MODEL
@st.cache_resource
def load_model():

    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

model = load_model()

#FILE UPLOAD
upload = st.file_uploader(
    "Upload CSV reviews",
    type=["csv"]
)

#PIPELINE PRINCIPAL
if upload:

    df = pd.read_csv(upload)

    st.write(df.head())

#CHOIX COLONNE
    selected_column = st.selectbox(
        "Choose review column",
        df.columns
    )

    reviews = (
        df[selected_column]
        .dropna()
        .astype(str)
        .tolist()
    )

#EMBEDDINGS REVIEWS
    review_embeddings = model.encode(
        reviews,
        show_progress_bar=True
    )

#QUESTION UTILISATEUR
    question = st.chat_input(
        "Ask a question about customer reviews..."
    )

#EMBEDDING QUESTION
    if question:

        question_embedding = model.encode(
            [question]
        )

#RECHERCHE SÉMANTIQUE
        similarities = cosine_similarity(
            question_embedding,
            review_embeddings
        )[0]

#TOP REVIEWS
        top_indices = np.argsort(
            similarities
        )[::-1][:5]

#CONTEXTE
        relevant_reviews = []

        for idx in top_indices:

            relevant_reviews.append(
                reviews[idx]
            )
# PROMPT ENRICHI
        context = "\n".join(
            relevant_reviews
        )
                prompt = f"""
        You are a customer insight analyst.

        Use these customer reviews
        to answer the question.

        Reviews:
        {context}

        Question:
        {question}

        Give a clear business-oriented answer.
        """

#GÉNÉRATION RÉPONSE
        response = llm.generate_content(
            prompt
        )

#AFFICHAGE
        st.chat_message("user").write(
            question
        )

        st.chat_message("assistant").write(
            response.text
        )