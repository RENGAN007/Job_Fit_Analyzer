"""
Deterministic JD-keyword extraction and resume matching.

Design rule: the LLM may *propose* keywords, but every keyword is
verified to appear verbatim in the job description before it is kept,
and the resume comparison is done in Python — never by the LLM.
This makes matched/missing keywords reproducible and un-hallucinatable.
"""

from __future__ import annotations
import re

# ─────────────────────────────────────────────────────────────
# Alias table: canonical label -> surface forms that mean the same thing.
# This is NOT the keyword list. It only teaches the matcher that
# "HuggingFace" == "Hugging Face" and "scikit-learn" == "sklearn".
# Keywords themselves come from the JD text.
# ─────────────────────────────────────────────────────────────
ALIASES: dict[str, list[str]] = {
    "python": ["python3"],
    "machine learning": ["ml", "machine-learning"],
    "deep learning": ["deep-learning", "dl"],
    "nlp": ["natural language processing"],
    "hugging face transformers": ["hugging face", "huggingface", "hf transformers", "transformers"],
    "pytorch": ["torch"],
    "tensorflow": ["tf", "keras"],
    "spacy": ["spacy"],
    "nltk": ["natural language toolkit"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "named entity recognition": ["ner"],
    "sentiment analysis": ["sentiment classification", "opinion mining"],
    "language modeling": ["language modelling", "language model", "llm"],
    "text classification": ["document classification", "text classifier"],
    "tokenization": ["tokenizing", "tokenisation", "tokenize", "tokeniser", "tokenizer"],
    "data preprocessing": ["preprocessing", "data cleaning", "text preprocessing"],
    "hyperparameter tuning": ["hyperparameters", "hyperparameter optimization", "hyperparameter optimisation"],
    "question answering": ["qa system", "question-answering"],
    "summarization": ["summarisation", "text summarization"],
    "sql": ["mysql", "postgresql", "postgres", "sqlite"],
    "data structures": ["data structure"],
    "algorithms": ["algorithm"],
    "probability": ["probabilistic"],
    "statistics": ["statistical", "stats"],
    "communication": ["communicate", "communicating", "communication skills"],
    "problem solving": ["problem-solving", "analytical thinking"],
    "documentation": ["documenting", "document"],
    "model evaluation": ["evaluating models", "model performance", "evaluation metrics"],
    "chatbots": ["chatbot", "conversational ai", "dialogue system"],
    "collaboration": ["collaborative", "teamwork", "team environment"],
}

# Seed vocabulary the deterministic extractor looks for in the JD.
# Broad on purpose — anything here that is NOT in the JD is discarded.
SEED_TERMS: list[str] = [
    # languages / core
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala", "go", "sql", "nosql",
    "git", "github", "linux", "bash", "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd",
    # ml / ai
    "machine learning", "deep learning", "neural networks", "transformers", "llm",
    "computer vision", "reinforcement learning", "feature engineering", "model deployment",
    "mlops", "model evaluation", "hyperparameter tuning", "cross-validation", "regression",
    "classification", "clustering", "supervised learning", "unsupervised learning",
    # nlp
    "nlp", "natural language processing", "tokenization", "lemmatization", "stemming",
    "named entity recognition", "sentiment analysis", "language modeling", "text classification",
    "topic modeling", "word embeddings", "word2vec", "bert", "gpt", "rag", "prompt engineering",
    "question answering", "summarization", "machine translation", "speech recognition", "chatbots",
    # libraries
    "pytorch", "tensorflow", "keras", "scikit-learn", "spacy", "nltk", "gensim",
    "hugging face transformers", "pandas", "numpy", "matplotlib", "seaborn", "opencv",
    "langchain", "streamlit", "flask", "fastapi", "django",
    # data
    "data preprocessing", "data analysis", "data visualization", "data pipelines", "etl",
    "power bi", "tableau", "excel", "big data", "spark", "hadoop", "data structures",
    "algorithms", "probability", "statistics", "a/b testing",
    # soft / process
    "communication", "problem solving", "documentation", "collaboration", "agile", "scrum",
    "stakeholder management", "presentation", "mentoring", "research",
]

_STOP_SECTION_WORDS = {
    "overview", "responsibilities", "requirements", "qualifications", "skills",
    "about", "role", "position", "apply", "benefits", "opportunities",
}


# ─────────────────────────────────────────────────────────────
# Normalisation & literal matching
# ─────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation that isn't part of a tech token, collapse spaces."""
    text = (text or "").lower()
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    text = re.sub(r"[^\w\s+#./-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def term_present(term: str, normalized_text: str) -> bool:
    """True only if `term` (or a trivial plural) literally occurs in the text."""
    t = normalize(term)
    if not t or len(t) < 2:
        return False
    if " " in t or "/" in t or "+" in t or "#" in t:
        return t in normalized_text
    return bool(re.search(rf"\b{re.escape(t)}s?\b", normalized_text))


def _surface_forms(keyword: str) -> list[str]:
    """Canonical keyword plus its known aliases."""
    k = normalize(keyword)
    forms = [k]
    forms += ALIASES.get(k, [])
    # reverse lookup: if the keyword IS an alias, pull in its canonical family
    for canon, alias_list in ALIASES.items():
        if k in [normalize(a) for a in alias_list]:
            forms.append(canon)
            forms += alias_list
    return list(dict.fromkeys(normalize(f) for f in forms if f))


# ─────────────────────────────────────────────────────────────
# Step 1 — extract keywords FROM THE JD
# ─────────────────────────────────────────────────────────────

def _extract_seed_keywords(jd_text: str) -> list[str]:
    """Deterministic pass: which seed terms literally appear in this JD?"""
    norm = normalize(jd_text)
    return [t for t in SEED_TERMS if term_present(t, norm)]


def _extract_llm_keywords(jd_text: str, groq_client, model: str, limit: int = 30) -> list[str]:
    """
    Ask the LLM for keywords, then throw away anything that is not a
    verbatim span of the JD. Hallucinated terms cannot survive this filter.
    """
    if groq_client is None:
        return []

    prompt = (
        "Extract the hiring keywords from the JOB DESCRIPTION below.\n\n"
        "STRICT RULES:\n"
        "- Every keyword MUST be copied verbatim from the job description text.\n"
        "- Do NOT paraphrase, generalise, or invent terms.\n"
        "- Prefer concrete skills, tools, technologies, methods and qualifications.\n"
        "- Ignore company boilerplate, benefits, and generic filler.\n"
        f"- Return at most {limit} keywords.\n"
        "- Output ONE keyword per line. No numbering, no bullets, no commentary.\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:6000]}"
    )
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content or ""
    except Exception:
        return []

    norm_jd = normalize(jd_text)
    out = []
    for line in raw.splitlines():
        kw = re.sub(r"^[\s\-•*\d.)]+", "", line).strip().strip(",.;:")
        if not kw or len(kw) < 2 or len(kw.split()) > 5:
            continue
        if normalize(kw) in _STOP_SECTION_WORDS:
            continue
        # THE GUARD: keyword must literally exist in the JD
        if term_present(kw, norm_jd):
            out.append(kw.lower())
    return out


def _canonical(kw: str) -> str:
    """Map a surface form onto its canonical label (itself if unknown)."""
    k = normalize(kw)
    for canon, alias_list in ALIASES.items():
        if k == canon or k in [normalize(a) for a in alias_list]:
            return canon
    return k


def _dedupe(keywords: list[str]) -> list[str]:
    """Collapse aliases onto one canonical entry and drop substring duplicates."""
    seen: dict[str, str] = {}
    for kw in keywords:
        canon = _canonical(kw)
        # keep the most specific label for the group
        if canon not in seen or len(canon) > len(seen[canon]):
            seen[canon] = canon
    result = list(seen.values())
    # drop a keyword fully contained in a longer kept one
    final: list[str] = []
    for kw in sorted(result, key=lambda s: -len(s)):
        n = normalize(kw)
        if any(n != normalize(o) and n in normalize(o) for o in final):
            continue
        final.append(kw)
    return final


def _relevance(kw: str, norm_jd: str, is_seed: bool) -> tuple:
    """
    Rank keywords so the cap keeps what the JD actually emphasises.
    Sort key (ascending): seed-first, then most-mentioned, then most-specific.
    """
    hits = sum(len(re.findall(rf"\b{re.escape(normalize(f))}", norm_jd)) for f in _surface_forms(kw))
    specificity = len(normalize(kw).split())
    return (0 if is_seed else 1, -hits, -specificity, normalize(kw))


def extract_jd_keywords(jd_text: str, groq_client=None, model: str = "", limit: int = 25) -> list[str]:
    """Union of the seed pass and the verbatim-verified LLM pass, ranked by JD emphasis."""
    seeds = _extract_seed_keywords(jd_text)
    llm = _extract_llm_keywords(jd_text, groq_client, model) if groq_client else []
    merged = _dedupe(seeds + llm)

    norm_jd = normalize(jd_text)
    seed_canon = {_canonical(s) for s in seeds}
    merged.sort(key=lambda k: _relevance(k, norm_jd, _canonical(k) in seed_canon))
    return merged[:limit]


# ─────────────────────────────────────────────────────────────
# Step 2 — compare against the resume (pure Python)
# ─────────────────────────────────────────────────────────────

def match_keywords(resume_text: str, jd_keywords: list[str]) -> dict:
    """
    Returns matched / missing keyword lists plus evidence for each match.
    `evidence` records which surface form was found, so the UI can prove it.
    """
    norm_resume = normalize(resume_text)
    matched, missing, evidence = [], [], {}

    for kw in jd_keywords:
        hit = next((f for f in _surface_forms(kw) if term_present(f, norm_resume)), None)
        if hit:
            matched.append(kw)
            evidence[kw] = hit
        else:
            missing.append(kw)

    coverage = len(matched) / len(jd_keywords) if jd_keywords else 0.0
    return {
        "matched": matched,
        "missing": missing,
        "evidence": evidence,
        "coverage": coverage,
        "keyword_coverage_pct": round(coverage * 100, 1),
        "jd_keywords": jd_keywords,
    }
