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
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Tokens ─────────────────────────────────── */
:root {
    --bg:        #0B0C10;
    --surface:   #12141A;
    --card:      #1A1D26;
    --border:    #2A2D3A;
    --text:      #F0F2F8;
    --muted:     #8B90A0;
    --user-bg:   #2D0A00;
    --user-bd:   #E53E1A;
    --user-text: #FFE8E0;
    --ai-bg:     #1A1500;
    --ai-bd:     #F5C518;
    --ai-text:   #FFF8DC;
    --accent:    #F5C518;
    --red:       #E53E1A;
}

/* ── Hard reset ──────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
.stApp { background: var(--bg) !important; }
.stApp, .stApp *:not(style):not(script) {
    font-family: 'Inter', sans-serif !important;
}

/* All Streamlit wrappers transparent + light text */
section[data-testid="stMain"],
div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlockSeparator"],
div[data-testid="column"],
div[data-testid="stHorizontalBlock"],
div[data-testid="stMarkdownContainer"],
div[data-testid="stCaptionContainer"],
div[data-testid="stText"],
div.element-container {
    background: transparent !important;
    color: var(--text) !important;
}

/* Force every text node light */
.stApp p, .stApp span, .stApp li,
.stApp h1, .stApp h2, .stApp h3,
.stApp label, .stApp small,
.stApp strong, .stApp em, .stApp a,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em {
    color: var(--text) !important;
}

/* ── Hide chrome ─────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
.block-container {
    padding: 2rem 2.5rem 5rem !important;
    max-width: 1100px;
}

/* ── SIDEBAR ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .block-container { padding: 2rem 1.4rem !important; }

/* ── Buttons ─────────────────────────────────── */
.stButton > button {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(245,197,24,0.06) !important;
}

/* ── Selectbox ───────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label * {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] > div > div *,
[data-testid="stSelectbox"] span,
[data-baseweb="select"] * {
    color: var(--text) !important;
    background: var(--card) !important;
}
[data-baseweb="popover"] *, [data-baseweb="menu"] *,
[role="listbox"] *, [role="option"] {
    background: var(--card) !important;
    color: var(--text) !important;
}
[role="option"]:hover { background: var(--border) !important; }

/* ── File uploader ───────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }

/* ── Progress / spinner ──────────────────────── */
[data-testid="stProgress"] > div { background: var(--border) !important; border-radius: 4px !important; }
[data-testid="stProgress"] > div > div { background: var(--accent) !important; border-radius: 4px !important; }
[data-testid="stSpinner"] > div > span { color: var(--text) !important; }

/* ── Caption ─────────────────────────────────── */
.stApp small,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * { color: var(--muted) !important; }

/* ═══════════════════════════════════════════════
   CHAT BUBBLES
═══════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1.25rem !important;
}

/* ── USER — dark red + orange-red left bar ────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label="user avatar"]) [data-testid="stChatMessageContent"] {
    background: var(--user-bg) !important;
    border: 1px solid rgba(229,62,26,0.30) !important;
    border-left: 4px solid var(--red) !important;
    border-radius: 0 14px 14px 14px !important;
    padding: 0.9rem 1.15rem !important;
    line-height: 1.65 !important;
}
/* USER text nodes — warm cream */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] span,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] div,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] em,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] li,
[data-testid="stChatMessage"]:has([aria-label="user avatar"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label="user avatar"]) [data-testid="stChatMessageContent"] * {
    color: var(--user-text) !important;
}

/* ── AI — dark yellow + gold left bar ────────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label="assistant avatar"]) [data-testid="stChatMessageContent"] {
    background: var(--ai-bg) !important;
    border: 1px solid rgba(245,197,24,0.28) !important;
    border-left: 4px solid var(--ai-bd) !important;
    border-radius: 0 14px 14px 14px !important;
    padding: 0.9rem 1.15rem !important;
    line-height: 1.65 !important;
}
/* AI text nodes — warm white */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] span,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] div,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] em,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] li,
[data-testid="stChatMessage"]:has([aria-label="assistant avatar"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label="assistant avatar"]) [data-testid="stChatMessageContent"] * {
    color: var(--ai-text) !important;
}

/* ── CHAT INPUT ──────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(245,197,24,0.10) !important;
}
/* Text visible while typing */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    background: transparent !important;
    color: var(--text) !important;
    caret-color: var(--accent) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] button { color: var(--accent) !important; }

/* ── Scrollbar ───────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Custom UI components ────────────────────── */
.rb-brand {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.rb-brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--red), var(--accent));
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; flex-shrink: 0;
}
.rb-brand-name {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.2rem; font-weight: 700;
    color: var(--text) !important; letter-spacing: -0.01em;
}
.rb-section {
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--muted) !important; margin: 1.5rem 0 0.55rem;
}
.rb-stat {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 9px; padding: 0.75rem 1rem; margin-bottom: 0.45rem;
    display: flex; justify-content: space-between; align-items: center;
}
.rb-stat-label { font-size: 0.77rem; color: var(--muted) !important; }
.rb-stat-value {
    font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700;
    color: var(--accent) !important;
}
.rb-header { margin-bottom: 2.2rem; }
.rb-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important; font-weight: 800 !important;
    letter-spacing: -0.04em !important; color: var(--text) !important;
    line-height: 1.05 !important; margin-bottom: 0.35rem !important;
}
.rb-header h1 span { color: var(--accent); }
.rb-header p { font-size: 0.92rem; color: var(--muted) !important; margin: 0; }
.rb-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(245,197,24,0.08); border: 1px solid rgba(245,197,24,0.22);
    color: var(--accent) !important; border-radius: 20px;
    padding: 0.22rem 0.75rem; font-size: 0.73rem;
    font-family: 'Syne', sans-serif; font-weight: 600; margin-bottom: 0.75rem;
}
.rb-empty {
    text-align: center; padding: 3.5rem 2rem;
    border: 1.5px dashed var(--border); border-radius: 16px; margin-top: 2rem;
}
.rb-empty .icon { font-size: 2.8rem; margin-bottom: 0.8rem; }
.rb-empty p { font-size: 0.9rem; color: var(--muted) !important; line-height: 1.6; margin: 0; }
.rb-empty strong { color: var(--text) !important; }
.rb-divider {
    display: flex; align-items: center; gap: 0.75rem; margin: 1.5rem 0;
    color: var(--muted) !important; font-size: 0.7rem;
    letter-spacing: 0.07em; text-transform: uppercase;
}
.rb-divider::before, .rb-divider::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}
.rb-source {
    display: inline-flex; align-items: center;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 5px; padding: 0.12rem 0.45rem;
    font-size: 0.7rem; color: var(--muted) !important;
    margin-right: 3px; margin-bottom: 3px;
}
.rb-legend {
    display: flex; gap: 1.2rem; margin-bottom: 1rem; align-items: center;
}
.rb-dot {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.75rem; color: var(--muted) !important;
}
.rb-dot::before {
    content: ''; width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
}
.rb-dot.user::before { background: var(--red); }
.rb-dot.ai::before   { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "messages"          not in st.session_state: st.session_state.messages = []
if "reviews"           not in st.session_state: st.session_state.reviews = []
if "review_embeddings" not in st.session_state: st.session_state.review_embeddings = None
if "_embed_key"        not in st.session_state: st.session_state._embed_key = None

# ─────────────────────────────────────────────
#  GEMINI & EMBEDDING MODEL
# ─────────────────────────────────────────────
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_model()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="rb-brand">
        <div class="rb-brand-icon">📊</div>
        <span class="rb-brand-name">ReviewBot</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rb-section">Data Source</div>', unsafe_allow_html=True)
    upload = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
        help="CSV file with customer reviews"
    )
    st.caption("📁 Drop a CSV file with customer reviews")

    df = None
    selected_column = None

    if upload is not None:
        df = pd.read_csv(upload)

        st.markdown('<div class="rb-section">Review Column</div>', unsafe_allow_html=True)
        selected_column = st.selectbox("Column", df.columns, label_visibility="collapsed")

        n_reviews = int(df[selected_column].dropna().shape[0])

        st.markdown('<div class="rb-section">Dataset</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rb-stat">
            <span class="rb-stat-label">Reviews</span>
            <span class="rb-stat-value">{n_reviews:,}</span>
        </div>
        <div class="rb-stat">
            <span class="rb-stat-label">Columns</span>
            <span class="rb-stat-value">{len(df.columns)}</span>
        </div>
        <div class="rb-stat">
            <span class="rb-stat-label">Q&amp;A turns</span>
            <span class="rb-stat-value">{len(st.session_state.messages) // 2}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="rb-section">Actions</div>', unsafe_allow_html=True)
        if st.button("🗑  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("<br>" * 3, unsafe_allow_html=True)
    st.caption("Gemini 2.5 Flash · MiniLM-L6-v2")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div class="rb-header">
    <h1>Review<span>Bot</span></h1>
    <p>Ask anything about your customer reviews — get instant business insights.</p>
</div>
""", unsafe_allow_html=True)

# ── Empty state ───────────────────────────────
if upload is None:
    st.markdown("""
    <div class="rb-empty">
        <div class="icon">📂</div>
        <p>
            <strong>Upload a CSV file to get started.</strong><br><br>
            Drop your customer reviews file in the sidebar,<br>
            select the review column, then ask any question.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Compute / retrieve embeddings ─────────────
reviews = df[selected_column].dropna().astype(str).tolist()
embed_key = f"{upload.name}|{selected_column}"

if st.session_state._embed_key != embed_key:
    with st.spinner("Indexing reviews…"):
        st.session_state.review_embeddings = embed_model.encode(reviews, show_progress_bar=False)
        st.session_state.reviews = reviews
        st.session_state._embed_key = embed_key

review_embeddings = st.session_state.review_embeddings
reviews = st.session_state.reviews

# Context pill + colour legend
st.markdown(f"""
<div>
    <span class="rb-pill">✦ {len(reviews):,} reviews indexed &middot; {selected_column}</span>
</div>
<div class="rb-legend">
    <span class="rb-dot user">You</span>
    <span class="rb-dot ai">ReviewBot</span>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────
if not st.session_state.messages:
    st.markdown('<div class="rb-divider">Start the conversation</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            src_html = "".join(
                f'<span class="rb-source">review #{s}</span>' for s in msg["sources"]
            )
            st.markdown(
                f"<div style='margin-top:0.5rem'>{src_html}</div>",
                unsafe_allow_html=True
            )

# ── Chat input ────────────────────────────────
question = st.chat_input("Ask a question about customer reviews…")

if question:
    # Show user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate answer
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analysing reviews…"):
            q_emb    = embed_model.encode([question])
            sims     = cosine_similarity(q_emb, review_embeddings)[0]
            top_idx  = np.argsort(sims)[::-1][:5]
            relevant = [reviews[i] for i in top_idx]

            context = "\n".join(f"- {r}" for r in relevant)
            prompt  = f"""You are a senior customer insight analyst.
Use the following customer reviews to answer the question.
Be clear, structured and business-oriented.
Highlight patterns, sentiments or actionable recommendations when relevant.

Reviews:
{context}

Question: {question}

Answer:"""
            response = llm.generate_content(prompt)
            answer   = response.text

        st.markdown(answer)

        source_nums = [int(i) + 1 for i in top_idx]
        src_html = "".join(
            f'<span class="rb-source">review #{s}</span>' for s in source_nums
        )
        st.markdown(
            f"<div style='margin-top:0.5rem'>"
            f"<small style='color:#8B90A0;font-size:0.7rem'>Sources: </small>"
            f"{src_html}</div>",
            unsafe_allow_html=True
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_nums
    })
