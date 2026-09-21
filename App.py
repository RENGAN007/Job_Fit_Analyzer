import streamlit as st
import os
import re
import numpy as np
import pdfplumber
import requests
from groq import Groq
 
from job_scraper import scrape_job_url
from keyword_match import extract_jd_keywords, match_keywords
from pdf_report import build_pdf_report
from history_db import save_analysis, load_all_analyses, delete_analysis, clear_all_analyses
from job_fetcher import fetch_recent_jobs
 
# Groq retired llama-3.1-8b-instant (and llama-3.3-70b-versatile) on 2026-08-16
# for free/developer tiers. openai/gpt-oss-20b is the documented replacement.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEPRECATED_GROQ_MODELS = {
    "llama-3.1-8b-instant": DEFAULT_GROQ_MODEL,
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "llama3-8b-8192": DEFAULT_GROQ_MODEL,
    "llama3-70b-8192": "openai/gpt-oss-120b",
}
 
 
# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎯 Jobbie-AI - Get your resume ready for the job market",
    page_icon="📝",
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
 
 
def resolve_groq_model() -> str:
    model = load_secret("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    return DEPRECATED_GROQ_MODELS.get(model, model)
 
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
    return scored[:n]
 
 
# ─── ATS Scoring (JD-derived keywords, deterministic matching) ───
 
def compute_semantic_component(resume_embedding, job_embeddings, top_n=4) -> float:
    if not job_embeddings:
        return 0.0
    similarities = [cosine_similarity(resume_embedding, emb) for emb in job_embeddings]
    top_sims = sorted(similarities, reverse=True)[:min(top_n, len(similarities))]
    avg_sim = float(np.mean(top_sims))
    return float(np.clip(avg_sim / 0.45 * 100, 0, 100))
 
 
def compute_ats_score(
    resume_text: str,
    resume_embedding,
    job_text: str,
    job_embeddings,
    groq_client=None,
    groq_model: str = "",
) -> tuple[int, dict]:
    """
    Score = literal keyword coverage of THIS job description (55%)
          + semantic similarity of resume to the JD (45%).
 
    Keywords are extracted from the job description itself and every one is
    verified to occur verbatim in the JD. The resume comparison is exact-match
    in Python, so matched/missing are reproducible and auditable.
    """
    jd_keywords = extract_jd_keywords(job_text, groq_client, groq_model)
    match_info  = match_keywords(resume_text, jd_keywords)
    semantic    = compute_semantic_component(resume_embedding, job_embeddings)
    keyword_pct = match_info["keyword_coverage_pct"]
 
    score = (0.55 * keyword_pct) + (0.45 * semantic)
 
    match_info["semantic_component"] = round(semantic, 1)
    match_info["final_score"] = int(round(np.clip(score, 0, 100)))
    return match_info["final_score"], match_info
 
 
# ─── Core RAG + Generation Pipeline ─────────────────────────
 
def run_analysis(resume_text: str, job_description: str) -> tuple[str, int, dict]:
    groq_key = load_secret("GROQ_API_KEY")
    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to .streamlit/secrets.toml "
            "on Streamlit Cloud, or to api.txt locally."
        )
 
    groq_model  = resolve_groq_model()
    client_groq = Groq(api_key=groq_key)
 
    resume_clean = clean_text(resume_text)
    job_clean    = clean_text(job_description)
    job_chunks   = chunk_text(job_clean)
 
    with st.spinner("🔢 Generating embeddings via HuggingFace API..."):
        job_embeddings   = get_embeddings(tuple(job_chunks))
        resume_embedding = get_embeddings((resume_clean,))[0]
 
    with st.spinner("🔑 Extracting keywords from the job description..."):
        ats_score, match_info = compute_ats_score(
            resume_clean, resume_embedding, job_clean, job_embeddings,
            groq_client=client_groq, groq_model=groq_model,
        )
 
    matched_kw = ", ".join(match_info["matched"]) or "None detected"
    missing_kw = ", ".join(match_info["missing"]) or "None detected"
 
    top_scored = query_top_chunks(
        job_chunks, job_embeddings, resume_embedding, n=min(4, len(job_chunks))
    )
    retrieved_context = "\n\n".join(
        f"Chunk {i+1} (similarity {sim:.2f}):\n{chunk}"
        for i, (sim, chunk) in enumerate(top_scored)
    )
 
    system_prompt = (
        "You are an expert resume coach and ATS analyst.\n"
        "The ATS score and the matched/missing keyword lists have already been "
        "computed deterministically by exact text matching. They are FACTS.\n\n"
        "Rules:\n"
        "- Copy the MATCHED KEYWORDS and MISSING KEYWORDS lists EXACTLY as given. "
        "Do not add, remove, rename, merge or reorder any keyword.\n"
        "- Do not invent a different numeric score.\n"
        "- Do not invent experience that is not in the resume.\n"
        "- Only use information from the resume, the job description, and the "
        "retrieved job-description excerpts below.\n"
        "- Keep the output practical, specific, and concise."
    )
 
    user_prompt = (
        f"ATS MATCH SCORE (computed): {ats_score}/100\n"
        f"Keyword coverage: {match_info['keyword_coverage_pct']}% "
        f"({len(match_info['matched'])} of {len(match_info['jd_keywords'])} "
        f"job-description keywords found in the resume)\n"
        f"Semantic relevance: {match_info['semantic_component']}%\n\n"
        f"MATCHED KEYWORDS (copy verbatim):\n{matched_kw}\n\n"
        f"MISSING KEYWORDS (copy verbatim):\n{missing_kw}\n\n"
        f"MOST RELEVANT JOB DESCRIPTION EXCERPTS:\n{retrieved_context}\n\n"
        f"RESUME:\n{resume_clean[:6000]}\n\n"
        "Instructions:\n"
        f"- Begin ATS SUMMARY with the exact line: ATS Match Score: {ats_score}/100\n"
        "- Explain the score by referring to the matched and missing keywords above.\n"
        "- Reproduce both keyword lists exactly, comma-separated.\n"
        "- Suggest 5 resume improvements, each targeting a specific MISSING keyword "
        "that the candidate could plausibly evidence.\n"
        "- Write a tailored cover letter of about 200-300 words.\n\n"
        "Use this exact format:\n\n"
        "ATS SUMMARY:\n"
        f"ATS Match Score: {ats_score}/100\n"
        "...\n\n"
        "MATCHED KEYWORDS:\n...\n\n"
        "MISSING KEYWORDS:\n...\n\n"
        "REWRITE SUGGESTIONS:\n- ...\n- ...\n- ...\n\n"
        "COVER LETTER:\n..."
    )
 
    with st.spinner(f"🤖 Generating analysis with Groq ({groq_model})..."):
        response = client_groq.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
        )
 
    return response.choices[0].message.content, ats_score, match_info
 
 
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
 
    explicit = re.search(
        r"ATS\s+Match\s+Score:\s*(\d{1,3})\s*/\s*100",
        text or "",
        re.IGNORECASE,
    )
    if explicit:
        score = int(explicit.group(1))
        if 0 <= score <= 100:
            return score
    for m in re.finditer(r'\b(\d{1,3})\b', text or ""):
 
        score = int(m.group(1))
        if 0 <= score <= 100:
            return score
    return None
 
 
def inject_ats_score(output_text: str, score: int) -> str:
    """Ensure the canonical embedding score appears in the ATS summary."""
    summary = parse_section(output_text, "ATS SUMMARY:")
    if summary is None:
        return output_text
    narrative = re.sub(
        r"^(?:Estimated\s+)?ATS\s+(?:Match\s+)?(?:match\s+)?(?:score|estimate)[:\s]*"
        r"\d{1,3}(?:\s*(?:out of|/)\s*100)?[^\n]*\n*",
        "",
        summary,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    new_summary = f"ATS Match Score: {score}/100\n\n{narrative}"
    return output_text.replace(
        f"ATS SUMMARY:\n{summary}",
        f"ATS SUMMARY:\n{new_summary}",
        1,
    )
 
 
def inject_keywords(output_text: str, match_info: dict) -> str:
    """
    Overwrite the model's keyword sections with the computed ones.
    Belt-and-braces: even if the LLM ignores the copy-verbatim instruction,
    what the user sees (and what lands in the PDF) is the exact-match result.
    """
    replacements = {
        "MATCHED KEYWORDS:": ", ".join(match_info.get("matched", [])) or "None detected",
        "MISSING KEYWORDS:": ", ".join(match_info.get("missing", [])) or "None detected",
    }
    for key, value in replacements.items():
        current = parse_section(output_text, key)
        if current is None:
            continue
        output_text = output_text.replace(
            f"{key}\n{current}", f"{key}\n{value}", 1
        )
    return output_text
 
# ─── UI ──────────────────────────────────────────────────────
 
st.title("🎯 Jobbie-AI-Resume Analyser")
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
        "   - Does keyword matching and semantic analysis\n"
        "   - Calculates ATS score\n"
        "   - Generates a summary of the analysis\n"
        "   - Generates a tailored cover letter\n"
        "   - Generates a rewrite suggestions\n"
       
         "4. **Download** your report (PDF, TXT, or MD)"
        "   - Shows recent job openings near your location"
        "4. **Download** your report"
     )
    st.divider()
    st.markdown("**Models Used**")
    st.code(f"Embeddings: all-MiniLM-L6-v2\nLLM: {DEFAULT_GROQ_MODEL} (Groq)", language="text")
 
    st.divider()
    st.header("🕘 Past Analyses")
 
    history = load_all_analyses()
 
    if not history:
        st.caption("No analyses saved yet.")
    else:
        st.caption(f"{len(history)} analyse(s) saved")
 
        for row in history:
            score = row["ats_score"]
            color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            label = f"{color} {score}/100 · {row['timestamp']}"
 
            with st.expander(label, expanded=False):
                st.markdown(f"**📄 Resume:** {row['resume_filename']}")
                st.markdown(f"**📋 JD snippet:** {row['job_snippet']}")
 
                if st.button("📂 Load this result", key=f"load_{row['id']}"):
                    st.session_state["output_text"]     = row["output_text"]
                    st.session_state["ats_score"]       = row["ats_score"]
                    st.session_state["resume_filename"] = row["resume_filename"]
                    st.session_state["job_description"] = row["job_snippet"]
                    # stale keyword evidence must not leak across analyses
                    st.session_state.pop("match_info", None)
                    st.rerun()
 
                if st.button("🗑️ Delete", key=f"del_{row['id']}"):
                    delete_analysis(row["id"])
                    st.rerun()
 
        st.divider()
        if st.button("🧹 Clear All History", use_container_width=True):
            clear_all_analyses()
            st.rerun()
 
col1, col2 = st.columns(2)
with col1:
    uploaded_resume = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
with col2:
    jd_tab1, jd_tab2 = st.tabs(["📋 Paste Job Description", "🔗 Fetch from URL"])
 
    with jd_tab1:
        job_description_input = st.text_area(
            "Paste the full job description here",
            height=280,
            placeholder="Paste the full job description here...",
            key="jd_text_input"
        )
 
    with jd_tab2:
        job_url = st.text_input(
            "Job URL",
            placeholder="https://www.linkedin.com/jobs/view/... or seek.com.au/job/...",
            key="jd_url_input"
        )
        fetch_col1, fetch_col2 = st.columns([1, 3])
 
        with fetch_col1:
            fetch_clicked = st.button("⬇️ Fetch JD", use_container_width=True)
 
        with fetch_col2:
            st.caption("Supports LinkedIn, Seek, Indeed, and most job boards")
 
        if fetch_clicked:
            if not job_url.strip():
                st.error("⚠️ Please enter a job URL.")
            else:
                with st.spinner("🔍 Fetching job description from URL..."):
                    try:
                        fetched_jd = scrape_job_url(job_url)
                        st.session_state["fetched_jd"] = fetched_jd
                        st.success(f"✅ Fetched {len(fetched_jd.split())} words from job posting!")
                    except Exception as e:
                        st.error(f"⚠️ Could not fetch: {e}")
                        st.session_state.pop("fetched_jd", None)
 
        if "fetched_jd" in st.session_state:
            edited_jd = st.text_area(
                "Fetched Job Description (editable)",
                value=st.session_state["fetched_jd"],
                height=200,
                key="fetched_jd_display"
            )
            st.session_state["fetched_jd"] = edited_jd
 
    job_description = (
        st.session_state.get("fetched_jd", "")
        if st.session_state.get("fetched_jd")
        else job_description_input
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
            output_text, ats_score, match_info = run_analysis(resume_text, job_description)
            output_text = inject_ats_score(output_text, ats_score)
            output_text = inject_keywords(output_text, match_info)
            st.session_state["output_text"] = output_text
            st.session_state["ats_score"] = ats_score
            st.session_state["match_info"] = match_info
            st.session_state["resume_filename"] = uploaded_resume.name
            st.session_state["job_description"] = job_description
            save_analysis(uploaded_resume.name, job_description, ats_score, output_text)
            adzuna_id  = load_secret("ADZUNA_APP_ID")
            adzuna_key = load_secret("ADZUNA_APP_KEY")
            if adzuna_id and adzuna_key:
                try:
                    groq_key    = load_secret("GROQ_API_KEY")
                    groq_model  = resolve_groq_model()
                    client_groq = Groq(api_key=groq_key)
                    with st.spinner("🔎 Finding recent job openings near your location..."):
                        jobs, jt, skills, loc = fetch_recent_jobs(
                            resume_text    = resume_text,
                            groq_client    = client_groq,
                            groq_model     = groq_model,
                            adzuna_app_id  = adzuna_id,
                            adzuna_app_key = adzuna_key,
                            country        = "au",
                            results        = 8,
                        )
                    st.session_state["recent_jobs"]         = jobs
                    st.session_state["jobs_query_title"]    = jt
                    st.session_state["jobs_query_skills"]   = skills
                    st.session_state["jobs_query_location"] = loc
                    st.session_state.pop("jobs_error", None)
                except Exception as e:
                    st.session_state["recent_jobs"] = []
                    st.session_state["jobs_error"]  = str(e)
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
 
    match_info = st.session_state.get("match_info") or {}
 
    for key, display_name in SECTION_META.items():
        section_text = parse_section(output_text, key)
        if not section_text:
            continue
 
        # Keyword sections are rendered from the computed result, with evidence.
        if key in ("MATCHED KEYWORDS:", "MISSING KEYWORDS:") and match_info:
            is_matched = key == "MATCHED KEYWORDS:"
            items = match_info.get("matched" if is_matched else "missing", [])
            total = len(match_info.get("jd_keywords", [])) or 1
            with st.expander(f"{display_name} ({len(items)}/{total})", expanded=True):
                if not items:
                    st.caption("None detected.")
                    continue
                evidence = match_info.get("evidence", {})
                for kw in items:
                    if is_matched:
                        found_as = evidence.get(kw, kw)
                        note = "" if found_as == kw else f" — found in resume as *“{found_as}”*"
                        st.markdown(f"✅ **{kw}**{note}")
                    else:
                        st.markdown(f"❌ **{kw}** — not found anywhere in your resume")
            continue
 
        with st.expander(display_name, expanded=True):
            if key == "ATS SUMMARY:":
                score = st.session_state.get("ats_score")
                if score is None:
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
                    if match_info:
                        st.caption(
                            f"Keyword coverage **{match_info['keyword_coverage_pct']}%** "
                            f"({len(match_info['matched'])}/{len(match_info['jd_keywords'])} "
                            f"JD keywords, weight 55%) · "
                            f"Semantic relevance **{match_info['semantic_component']}%** "
                            f"(weight 45%)"
                        )
            st.markdown(section_text)
 
    st.divider()
 
    resume_filename = st.session_state.get("resume_filename", "")
    job_desc = st.session_state.get("job_description", "")
    try:
        pdf_bytes = build_pdf_report(
            output_text,
            resume_filename=resume_filename,
            job_description=job_desc,
            ats_score=st.session_state.get("ats_score"),
        )
    except Exception as e:
        pdf_bytes = None
        st.warning(f"PDF generation failed: {e}")
 
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        if pdf_bytes:
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="job_fit_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            st.button(
                "⬇️ Download PDF Report",
                disabled=True,
                use_container_width=True,
            )
    with dl2:
        st.download_button(
            "⬇️ Download Report (.txt)",
            data=output_text,
            file_name="job_coach_output.txt",
            mime="text/plain",
            use_container_width=True
        )
    with dl3:
        st.download_button(
            "⬇️ Download Report (.md)",
            data=output_text,
            file_name="job_coach_output.md",
            mime="text/markdown",
            use_container_width=True
        )
 
# ─── Recent Job Listings ──────────────────────────────────────────────────────
if "output_text" in st.session_state:
    recent_jobs = st.session_state.get("recent_jobs", [])
    st.divider()
 
    if "jobs_error" in st.session_state:
        st.error(f"⚠️ Job fetch error: {st.session_state['jobs_error']}")
    elif recent_jobs:
        jt          = st.session_state.get("jobs_query_title", "")
        skills      = st.session_state.get("jobs_query_skills", "")
        loc_display = st.session_state.get("jobs_query_location", "")
        loc_label   = f" · 📍 Near **{loc_display}**" if loc_display else " · 🌏 Australia-wide"
        st.subheader("🔎 Recent Job Openings")
        st.caption(f"Matched on: **{jt}** · Skills: {skills}{loc_label} · {len(recent_jobs)} listing(s) found")
 
        for job in recent_jobs:
            with st.container():
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"### [{job['title']}]({job['url']})")
                    st.markdown(
                        f"🏢 **{job['company']}** &nbsp;|&nbsp; "
                        f"📍 {job['location']} &nbsp;|&nbsp; "
                        f"💰 {job['salary']} &nbsp;|&nbsp; "
                        f"🕐 {job['posted']}"
                    )
                    st.caption(job['snippet'])
                with c2:
                    st.link_button("Apply →", job['url'], use_container_width=True)
                st.divider()
    elif "jobs_query_title" in st.session_state:
        loc_display = st.session_state.get("jobs_query_location", "")
        loc_msg     = f" near **{loc_display}**" if loc_display else ""
        st.warning(
            f"🔎 No job listings found{loc_msg} even after broadening the search. "
            f"Extracted title: **{st.session_state.get('jobs_query_title', 'unknown')}**. "
            "Try re-running the analysis."
        )
    else:
        st.info(
            "💡 Add **ADZUNA_APP_ID** and **ADZUNA_APP_KEY** to your Streamlit secrets "
            "to enable nearby job listings. Get free keys at [developer.adzuna.com](https://developer.adzuna.com)"
        )
