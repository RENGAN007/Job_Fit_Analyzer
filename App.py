import streamlit as st
import os
import re
import numpy as np
import pdfplumber
import requests
from groq import Groq

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job-Fit Analyzer",
    page_icon="🎯",
    layout="wide"
)

# ─── API Key Loaders ─────────────────────────────────────────

def load_secret(key_name):
    """Load a secret from Streamlit secrets → env var → api.txt fallback."""
    try:
        val = st.secrets.get(key_name)
        if val:
            return str(val).strip()
    except Exception:
        pass
    val = os.getenv(key_name)
    if val:
        return val.strip()
    try:
        with open("api.txt", "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(rf'{key_name}\s*=\s*["\']?([^"\']+)["\']?', line.strip())
                if match:
                    return match.group(1).strip()
    except FileNotFoundError:
        pass
    return None

# ─── PDF Extraction ──────────────────────────────────────────

def extract_pdf_text(uploaded_file):
    text = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

# ─── Text Utilities ──────────────────────────────────────────

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=150, overlap=70):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# ─── Embeddings via HuggingFace Inference API ────────────────

@st.cache_data(show_spinner=False)
def get_embeddings(texts_tuple: tuple) -> list:
    texts = list(texts_tuple)
    hf_key = load_secret("HF_API_KEY")
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    if not hf_key:
        raise ValueError(
            "HF_API_KEY not found. Add HF_API_KEY to .streamlit/secrets.toml "
            "or set it as an environment variable."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {hf_key}",
    }
    endpoint = (
        f"https://router.huggingface.co/hf-inference/models/"
        f"{model_id}/pipeline/feature-extraction"
    )
    response = requests.post(
        endpoint,
        headers=headers,
        json={"inputs": texts, "normalize": True},
        timeout=120
    )

    if response.status_code == 401:
        raise ValueError("HuggingFace authentication failed (401). Check HF_API_KEY.")
    if response.status_code != 200:
        raise ValueError(f"HuggingFace Embedding API error {response.status_code}: {response.text}")

    payload = response.json()
    if not payload:
        raise ValueError("HuggingFace Embedding API returned an empty response.")

    if isinstance(payload[0], list) and isinstance(payload[0][0], (int, float)):
        return payload

    sentence_vectors = []
    for token_vecs in payload:
        if not token_vecs:
            sentence_vectors.append([0.0] * len(payload[0][0]))
            continue
        dims = len(token_vecs[0])
        pooled = [sum(tv[i] for tv in token_vecs) / len(token_vecs) for i in range(dims)]
        sentence_vectors.append(pooled)
    return sentence_vectors

# ─── In-Memory Vector Search (replaces ChromaDB) ─────────────

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def query_top_chunks(job_chunks, job_embeddings, resume_embedding, n=4):
    scored = [
        (cosine_similarity(resume_embedding, emb), chunk)
        for emb, chunk in zip(job_embeddings, job_chunks)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:n]]

# ─── Core RAG + Generation Pipeline ─────────────────────────

def run_analysis(resume_text: str, job_description: str) -> str:
    groq_key = load_secret("GROQ_API_KEY")
    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to .streamlit/secrets.toml "
            "on Streamlit Cloud, or to api.txt locally."
        )

    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)
    job_chunks = chunk_text(job_clean)

    with st.spinner("🔢 Generating embeddings via HuggingFace API..."):
        job_embeddings = get_embeddings(tuple(job_chunks))
        resume_embedding = get_embeddings((resume_clean,))[0]

    top_chunks = query_top_chunks(job_chunks, job_embeddings, resume_embedding, n=min(4, len(job_chunks)))
    retrieved_context = "\n\n".join(
        f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)
    )

    system_prompt = (
        "You are an expert resume coach and ATS analyst.\n"
        "Your job is to compare a resume against a job description and provide:\n"
        "1. ATS match summary\n2. Matched keywords\n3. Missing keywords\n"
        "4. Resume rewrite suggestions\n5. A short tailored cover letter\n\n"
        "Rules:\n"
        "- Do not invent experience.\n"
        "- Only use information from the resume and job description.\n"
        "- Keep the output practical, specific, and concise."
    )
    user_prompt = (
        f"RESUME:\n{resume_clean}\n\n"
        f"JOB DESCRIPTION:\n{job_clean}\n\n"
        f"RETRIEVED JOB CONTEXT:\n{retrieved_context}\n\n"
        "Instructions:\n"
        "- Give a realistic ATS match estimate from 0 to 100.\n"
        "- List the strongest matched keywords.\n"
        "- List missing keywords that matter most.\n"
        "- Be strict about missing keywords.\n"
        "- Suggest 5 resume improvements.\n"
        "- Write a tailored cover letter of about 200-300 words.\n\n"
        "Use this exact format:\n\n"
        "ATS SUMMARY:\n...\n\n"
        "MATCHED KEYWORDS:\n...\n\n"
        "MISSING KEYWORDS:\n...\n\n"
        "REWRITE SUGGESTIONS:\n- ...\n- ...\n- ...\n\n"
        "COVER LETTER:\n..."
    )

    groq_model = load_secret("GROQ_MODEL") or "llama-3.1-8b-instant"
    with st.spinner(f"🤖 Generating analysis with Groq ({groq_model})..."):
        client_groq = Groq(api_key=groq_key)
        response = client_groq.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
        )
    return response.choices[0].message.content

# ─── Output Parsing Helpers ──────────────────────────────────

SECTION_KEYS = [
    "ATS SUMMARY:",
    "MATCHED KEYWORDS:",
    "MISSING KEYWORDS:",
    "REWRITE SUGGESTIONS:",
    "COVER LETTER:",
]

def parse_section(output: str, key: str):
    if key not in output:
        return None
    text = output.split(key, 1)[1]
    for other in SECTION_KEYS:
        if other != key and other in text:
            text = text.split(other, 1)[0]
    return text.strip()

def extract_ats_score(text: str):
    for m in re.finditer(r'\b(\d{1,3})\b', text or ""):
        score = int(m.group(1))
        if 0 <= score <= 100:
            return score
    return None

# ─── UI ──────────────────────────────────────────────────────

st.title("🎯 AI Job-Fit Analyzer")
st.caption(
    "Upload your resume and paste a job description to get an ATS score, "
    "keyword gap analysis, and a tailored cover letter — powered by "
    "Groq + HuggingFace embeddings."
)

with st.sidebar:
    st.header("ℹ️ How It Works")
    st.markdown(
        "1. **Upload** your resume (PDF)\n"
        "2. **Paste** the job description\n"
        "3. **Click Analyze** — the app:\n"
        "   - Extracts & chunks the JD\n"
        "   - Embeds via HuggingFace API\n"
        "   - Retrieves top chunks (cosine similarity)\n"
        "   - Sends to Groq LLM for analysis\n"
        "4. **Download** your report"
    )
    st.divider()
    st.markdown("**Models Used**")
    st.code("Embeddings: all-MiniLM-L6-v2\nLLM: llama-3.1-8b-instant (Groq)", language="text")

st.divider()

col1, col2 = st.columns(2)
with col1:
    uploaded_resume = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
with col2:
    job_description = st.text_area(
        "📋 Paste Job Description",
        height=300,
        placeholder="Paste the full job description here..."
    )

st.divider()

if st.button("🚀 Analyze My Resume", type="primary", use_container_width=True):
    if not uploaded_resume:
        st.error("⚠️ Please upload your resume PDF.")
    elif not job_description.strip():
        st.error("⚠️ Please paste a job description.")
    else:
        st.session_state.pop("output_text", None)

        with st.spinner("🔍 Extracting text from your resume PDF..."):
            try:
                resume_text = extract_pdf_text(uploaded_resume)
                if not resume_text.strip():
                    st.error("⚠️ Could not extract text from this PDF. Try a non-scanned PDF.")
                    st.stop()
            except Exception as e:
                st.error(f"⚠️ PDF extraction failed: {e}")
                st.stop()

        try:
            output_text = run_analysis(resume_text, job_description)
            st.session_state["output_text"] = output_text
        except ValueError as e:
            st.error(f"⚠️ {e}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
            st.stop()

if "output_text" in st.session_state:
    output_text = st.session_state["output_text"]

    st.success("✅ Analysis Complete!")
    st.divider()

    SECTION_META = {
        "ATS SUMMARY:":         "📊 ATS Match Summary",
        "MATCHED KEYWORDS:":    "✅ Matched Keywords",
        "MISSING KEYWORDS:":    "❌ Missing Keywords",
        "REWRITE SUGGESTIONS:": "✏️ Rewrite Suggestions",
        "COVER LETTER:":        "📝 Tailored Cover Letter",
    }

    for key, display_name in SECTION_META.items():
        section_text = parse_section(output_text, key)
        if not section_text:
            continue
        with st.expander(display_name, expanded=True):
            if key == "ATS SUMMARY:":
                score = extract_ats_score(section_text)
                if score is not None:
                    st.markdown(f"### ATS Score: **{score} / 100**")
                    color = (
                        "#22c55e" if score >= 70 else
                        "#f97316" if score >= 50 else
                        "#ef4444"
                    )
                    st.markdown(
                        f'<div style="background:#e5e7eb;border-radius:8px;height:24px;width:100%;margin-bottom:12px">'
                        f'<div style="background:{color};width:{score}%;height:24px;border-radius:8px;'
                        f'transition:width 0.6s ease"></div></div>',
                        unsafe_allow_html=True
                    )
            st.markdown(section_text)

    st.divider()

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download Report (.txt)",
            data=output_text,
            file_name="job_coach_output.txt",
            mime="text/plain",
            use_container_width=True
        )
    with dl2:
        st.download_button(
            "⬇️ Download Report (.md)",
            data=output_text,
            file_name="job_coach_output.md",
            mime="text/markdown",
            use_container_width=True
        )
