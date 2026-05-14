import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="ReviewBot",
    layout="wide"
)

# ==================================================
# CUSTOM UI / UX
# ==================================================

st.markdown("""
<style>

/* Background principal */
.stApp {
    background-color: #0E1117;
}

/* Texte général */
html, body, [class*="css"] {
    color: white;
    font-size: 16px;
}

/* Titres */
h1, h2, h3 {
    color: white;
}

/* Messages USER */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background-color: #8B0000;
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 10px;
    color: white;
}

/* Messages BOT */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background-color: #D4A017;
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 10px;
    color: black;
}

/* Input box */
.stChatInputContainer {
    background-color: #1E1E1E;
}

/* Texte input */
textarea {
    color: white !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161A23;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================

st.title("🤖 ReviewBot")
st.subheader("AI-Powered Customer Review Chatbot")

# ==================================================
# GEMINI CONFIG
# ==================================================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

model = load_model()

# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("📁 Conversations")

# Save conversation button
if st.sidebar.button("💾 Save Current Conversation"):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.session_state.chat_history.append({
        "timestamp": timestamp,
        "messages": st.session_state.messages.copy()
    })

    st.sidebar.success(
        "Conversation saved!"
    )

# Display old conversations
for idx, chat in enumerate(
    st.session_state.chat_history
):

    with st.sidebar.expander(
        f"Conversation {idx+1} - {chat['timestamp']}"
    ):

        for msg in chat["messages"]:

            st.write(
                f"**{msg['role']}** : {msg['content']}"
            )

# ==================================================
# FILE UPLOAD
# ==================================================

upload = st.file_uploader(
    "📂 Upload your customer reviews CSV",
    type=["csv"]
)

# ==================================================
# MAIN PIPELINE
# ==================================================

if upload:

    # Load dataset
    df = pd.read_csv(upload)

    st.write("## 📊 Dataset Preview")

    st.dataframe(df.head())

    # Select review column
    selected_column = st.selectbox(
        "Choose review column",
        df.columns
    )

    # Extract reviews
    reviews = (
        df[selected_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    st.success(
        f"{len(reviews)} reviews loaded."
    )

    # ==================================================
    # EMBEDDINGS
    # ==================================================

    with st.spinner(
        "Generating semantic embeddings..."
    ):

        review_embeddings = model.encode(
            reviews,
            show_progress_bar=True
        )

    st.success(
        "Embeddings generated successfully."
    )

    # ==================================================
    # DISPLAY CHAT HISTORY
    # ==================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # ==================================================
    # USER QUESTION
    # ==================================================

    question = st.chat_input(
        "Ask a question about customer reviews..."
    )

    # ==================================================
    # RAG PIPELINE
    # ==================================================

    if question:

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Display user message
        with st.chat_message("user"):

            st.markdown(question)

        # ==================================================
        # AI RESPONSE
        # ==================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "🔍 Searching relevant reviews and generating response..."
            ):

                # Question embedding
                question_embedding = model.encode(
                    [question]
                )

                # Semantic search
                similarities = cosine_similarity(
                    question_embedding,
                    review_embeddings
                )[0]

                # Top reviews
                top_indices = np.argsort(
                    similarities
                )[::-1][:5]

                relevant_reviews = []

                for idx in top_indices:

                    relevant_reviews.append(
                        reviews[idx]
                    )

                # Context
                context = "\n".join(
                    relevant_reviews
                )

                # Prompt
                prompt = f"""
                You are a customer insight analyst.

                Use the customer reviews below
                to answer the user's question.

                Customer Reviews:
                {context}

                User Question:
                {question}

                Give a clear and business-oriented answer.
                """

                # LLM generation
                response = llm.generate_content(
                    prompt
                )

                ai_response = response.text

                st.markdown(ai_response)

        # ==================================================
        # SAVE AI MESSAGE
        # ==================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": ai_response
            }
        )
