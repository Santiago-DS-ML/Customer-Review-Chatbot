import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ReviewBot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Refined dark editorial theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Root tokens ───────────────────────────── */
:root {
    --bg:       #0D0F14;
    --surface:  #13161D;
    --card:     #191C25;
    --border:   #252935;
    --accent:   #5BFFC2;
    --accent2:  #FF6B6B;
    --accent3:  #C0A4FF;
    --text:     #E8EAF0;
    --muted:    #6B7280;
    --font-display: 'DM Serif Display', serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'DM Mono', monospace;
}

/* ── Global reset — force light text everywhere ─ */
html, body { background-color: var(--bg) !important; color: var(--text) !important; }

/* Catch ALL Streamlit wrappers: stApp, stMain, stVerticalBlock, etc. */
.stApp,
.stApp > *,
section[data-testid="stMain"],
section[data-testid="stMain"] > *,
div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlock"] > *,
div[data-testid="column"] > *,
.element-container,
.element-container > * {
    background-color: transparent;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* Force text color on every p, span, div, label that Streamlit renders */
.stApp p,
.stApp span,
.stApp div,
.stApp label,
.stApp small,
.stApp li,
.stApp strong,
.stApp em,
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"] *,
[data-testid="stCaptionContainer"] * {
    color: var(--text) !important;
}

/* Muted / caption text */
.stApp small,
[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

/* ── App background ─────────────────────────── */
.stApp { background: var(--bg) !important; }
section[data-testid="stMain"] { background: var(--bg) !important; }
div[data-testid="stDecoration"] { display: none; }

/* ── Hide default Streamlit chrome ──────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1200px;
}

/* ── SIDEBAR — force all text light ─────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebarContent"],
[data-testid="stSidebarContent"] * {
    color: var(--text) !important;
}
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem !important; }

/* ── SELECTBOX — dropdown text ───────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stSelectbox"] > div > div > div,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] p {
    color: var(--text) !important;
}
[data-testid="stSelectbox"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
/* Dropdown overlay list */
[data-baseweb="popover"] *,
[data-baseweb="menu"] *,
[role="listbox"] *,
[role="option"] {
    background: var(--card) !important;
    color: var(--text) !important;
}
[role="option"]:hover { background: var(--border) !important; }

/* ── FILE UPLOADER ───────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: var(--text) !important;
}

/* ── BUTTONS ─────────────────────────────────── */
.stButton > button {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s, background 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(91,255,194,0.06) !important;
    color: var(--accent) !important;
}

/* ── SPINNER text ────────────────────────────── */
[data-testid="stSpinner"] > div > span { color: var(--text) !important; }
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.sidebar-brand-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent3) 100%);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.sidebar-brand-name {
    font-family: var(--font-display) !important;
    font-size: 1.25rem;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

/* Sidebar section labels */
.sidebar-section {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.5rem 0 0.6rem;
}

/* Stats cards in sidebar */
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-label { font-size: 0.78rem; color: var(--muted); }
.stat-value {
    font-family: var(--font-mono);
    font-size: 0.95rem;
    color: var(--accent);
    font-weight: 500;
}

/* ── MAIN HEADER ────────────────────────────── */
.main-header {
    margin-bottom: 2.5rem;
}
.main-header h1 {
    font-family: var(--font-display) !important;
    font-size: 2.8rem !important;
    font-weight: 400 !important;
    letter-spacing: -0.03em !important;
    color: var(--text) !important;
    line-height: 1.1 !important;
    margin-bottom: 0.3rem !important;
}
.main-header h1 em {
    font-style: italic;
    color: var(--accent);
}
.main-header p {
    font-size: 0.95rem;
    color: var(--muted);
    margin: 0;
    letter-spacing: 0.01em;
}

/* ── DATA PREVIEW TABLE ──────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── CHAT MESSAGES ───────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1.4rem !important;
}

/* ── AI message block — deep blue-slate ──────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label="assistant avatar"]) [data-testid="stChatMessageContent"] {
    background: #1A2236 !important;
    border: 1px solid #2A3A5C !important;
    border-left: 3px solid var(--accent3) !important;
    border-radius: 4px 14px 14px 14px !important;
    padding: 1rem 1.2rem !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([aria-label="assistant avatar"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] span,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] strong {
    color: #E8EAF0 !important;
}

/* ── USER message block — warm amber tint ─────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label="user avatar"]) [data-testid="stChatMessageContent"] {
    background: #221C10 !important;
    border: 1px solid #3D3010 !important;
    border-right: 3px solid #F5A623 !important;
    border-radius: 14px 4px 14px 14px !important;
    padding: 1rem 1.2rem !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([aria-label="user avatar"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] span,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] strong {
    color: #F5ECD7 !important;
}

/* ── CHAT INPUT ──────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    margin-top: 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(91, 255, 194, 0.08) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
}

/* ── PROGRESS BAR ────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent3)) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
    background: var(--border) !important;
    border-radius: 4px !important;
}

/* ── SPINNER ─────────────────────────────────── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── ALERTS / INFO ───────────────────────────── */
[data-testid="stAlert"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

/* ── DIVIDER ─────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── SCROLLBAR ───────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── CUSTOM COMPONENTS ───────────────────────── */
.upload-hint {
    text-align: center;
    padding: 3rem 2rem;
    border: 1.5px dashed var(--border);
    border-radius: 16px;
    margin-top: 2rem;
    color: var(--muted);
}
.upload-hint .hint-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.upload-hint p { font-size: 0.9rem; line-height: 1.5; margin: 0; }
.upload-hint strong { color: var(--text); }

.context-pill {
    display: inline-block;
    background: rgba(91, 255, 194, 0.08);
    border: 1px solid rgba(91, 255, 194, 0.2);
    color: var(--accent);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    margin-bottom: 1rem;
}

.chat-divider {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0;
    color: var(--muted);
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.chat-divider::before, .chat-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.source-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.72rem;
    font-family: var(--font-mono);
    color: var(--muted);
    margin-right: 4px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "reviews" not in st.session_state:
    st.session_state.reviews = []
if "review_embeddings" not in st.session_state:
    st.session_state.review_embeddings = None
if "df_shape" not in st.session_state:
    st.session_state.df_shape = None
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None

# ─────────────────────────────────────────────
#  GEMINI & MODEL SETUP
# ─────────────────────────────────────────────
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">📊</div>
        <span class="sidebar-brand-name">ReviewBot</span>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
    upload = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="CSV file containing customer reviews",
        label_visibility="collapsed"
    )
    st.caption("📁 Drag & drop or click to upload a CSV file")

    # Column selector (shown only when file loaded)
    if upload is not None:
        df = pd.read_csv(upload)
        st.markdown('<div class="sidebar-section">Review Column</div>', unsafe_allow_html=True)
        selected_column = st.selectbox(
            "Choose review column",
            df.columns,
            label_visibility="collapsed"
        )

        # Stats
        st.markdown('<div class="sidebar-section">Dataset Stats</div>', unsafe_allow_html=True)
        n_reviews = int(df[selected_column].dropna().shape[0])
        n_cols = len(df.columns)

        st.markdown(f"""
        <div class="stat-card">
            <span class="stat-label">Total reviews</span>
            <span class="stat-value">{n_reviews:,}</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Columns</span>
            <span class="stat-value">{n_cols}</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Conversations</span>
            <span class="stat-value">{len(st.session_state.messages) // 2}</span>
        </div>
        """, unsafe_allow_html=True)

        # Clear chat
        st.markdown('<div class="sidebar-section">Actions</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        selected_column = None
        df = None

    # Footer
    st.markdown("<br>" * 4, unsafe_allow_html=True)
    st.caption("Powered by Gemini 2.5 Flash · MiniLM-L6")

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>Customer <em>Insight</em> Chat</h1>
    <p>Ask questions about your reviews — get business-oriented answers, instantly.</p>
</div>
""", unsafe_allow_html=True)

# ── No file uploaded state ────────────────────
if upload is None:
    st.markdown("""
    <div class="upload-hint">
        <div class="hint-icon">📂</div>
        <p>
            <strong>Upload a CSV file to get started.</strong><br><br>
            Drop your customer reviews CSV in the sidebar.<br>
            Then ask any question — ReviewBot will analyse the most relevant reviews<br>
            and give you clear, actionable insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── File loaded: process embeddings ──────────
reviews = df[selected_column].dropna().astype(str).tolist()

# Only recompute embeddings when column or file changes
state_key = f"{upload.name}_{selected_column}"
if st.session_state.get("_embed_key") != state_key:
    with st.spinner("Computing semantic embeddings…"):
        st.session_state.review_embeddings = model.encode(reviews, show_progress_bar=False)
        st.session_state.reviews = reviews
        st.session_state._embed_key = state_key

review_embeddings = st.session_state.review_embeddings
reviews = st.session_state.reviews

# Context pill
st.markdown(f"""
<div>
    <span class="context-pill">✦ {len(reviews):,} reviews indexed · {selected_column}</span>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown('<div class="chat-divider">Start the conversation</div>', unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            # Show source tags for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                sources_html = "".join([
                    f'<span class="source-tag">review #{s}</span>'
                    for s in msg["sources"]
                ])
                st.markdown(f"<div style='margin-top:0.6rem'>{sources_html}</div>", unsafe_allow_html=True)

# ── Chat input ────────────────────────────────
question = st.chat_input("Ask a question about customer reviews…")

if question:
    # Show user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Semantic search + generation
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analysing reviews…"):
            # Embed question
            q_emb = model.encode([question])

            # Cosine similarity search
            sims = cosine_similarity(q_emb, review_embeddings)[0]
            top_idx = np.argsort(sims)[::-1][:5]
            relevant = [reviews[i] for i in top_idx]

            # Build prompt
            context = "\n".join([f"- {r}" for r in relevant])
            prompt = f"""You are a senior customer insight analyst.
Use the following customer reviews to answer the question.
Be clear, structured, and business-oriented.
If relevant, highlight patterns, sentiments or actionable recommendations.

Customer Reviews:
{context}

Question: {question}

Answer:"""

            # Generate
            response = llm.generate_content(prompt)
            answer = response.text

        st.markdown(answer)

        # Source tags
        source_nums = [int(i) + 1 for i in top_idx]
        sources_html = "".join([f'<span class="source-tag">review #{s}</span>' for s in source_nums])
        st.markdown(f"<div style='margin-top:0.6rem'><small style='color:var(--muted);font-size:0.72rem;'>Sources: </small>{sources_html}</div>", unsafe_allow_html=True)

    # Persist to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_nums
    })
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

/* Boutons */
.stButton>button {
    background-color: #D4A017;
    color: black;
    border-radius: 10px;
    border: none;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# TITRE
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
# FILE UPLOAD
# ==================================================

upload = st.file_uploader(
    "📂 Upload your customer reviews CSV",
    type=["csv"]
)

# ==================================================
# CHAT HISTORY
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

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
    # DISPLAY OLD CHAT
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

        # ==============================================
        # QUESTION EMBEDDING
        # ==============================================

        question_embedding = model.encode(
            [question]
        )

        # ==============================================
        # SEMANTIC SEARCH
        # ==============================================

        similarities = cosine_similarity(
            question_embedding,
            review_embeddings
        )[0]

        # ==============================================
        # TOP REVIEWS
        # ==============================================

        top_indices = np.argsort(
            similarities
        )[::-1][:5]

        relevant_reviews = []

        for idx in top_indices:

            relevant_reviews.append(
                reviews[idx]
            )

        # ==============================================
        # CONTEXT
        # ==============================================

        context = "\n".join(
            relevant_reviews
        )

        # ==============================================
        # PROMPT
        # ==============================================

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

        # ==============================================
        # LLM RESPONSE
        # ==============================================

        with st.chat_message("assistant"):

            with st.spinner(
                "Analyzing customer feedback..."
            ):

                response = llm.generate_content(
                    prompt
                )

                ai_response = response.text

                st.markdown(ai_response)

        # ==============================================
        # SAVE AI RESPONSE
        # ==============================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": ai_response
            }
        )
