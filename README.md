# 🎯 Jobbie-AI — Resume & Job-Fit Analyzer

> An AI-powered resume analysis and job-matching application that combines **NLP embeddings, semantic similarity, keyword analysis, ATS-style scoring, and LLM-powered feedback** to help candidates tailor their resumes to specific job opportunities.

**Status:** 🚧 Under active development

---

## 📌 Overview

**Jobbie-AI** is an end-to-end AI application designed to help job seekers understand how well their resume matches a particular job description.

The application allows users to:

* Upload their resume as a PDF
* Paste a job description or fetch one from a job URL
* Extract and preprocess resume and job-description text
* Generate semantic embeddings using **Sentence Transformers**
* Compare resume and job-description content using **cosine similarity**
* Identify matched and missing skill categories
* Calculate an ATS-style match score
* Generate AI-powered resume improvement suggestions
* Generate a tailored cover letter
* Save and revisit previous analyses
* Download analysis reports as PDF, TXT or Markdown
* Find recent job openings based on the candidate's profile

The project is designed as a practical demonstration of how **NLP, semantic search, retrieval, LLMs and API integration** can be combined into a real-world AI application.

---

# ✨ Key Features

## 📄 1. Resume PDF Processing

Users can upload their resume in PDF format.

The application extracts text using `pdfplumber` and performs basic preprocessing before analysis.

### Processing pipeline

```text
Resume PDF
    ↓
PDF Text Extraction
    ↓
Text Cleaning
    ↓
Resume Representation
```

---

## 🧹 2. Text Cleaning & Chunking

Job descriptions are cleaned and divided into overlapping text chunks.

The current implementation uses:

* Whitespace normalization
* Control-character removal
* Word-based chunking
* Overlapping chunks

This allows the system to work with different sections of longer job descriptions during semantic retrieval.

---

## 🧠 3. Transformer-Based Embeddings

Jobbie-AI uses:

**`sentence-transformers/all-MiniLM-L6-v2`**

through the Hugging Face Inference API to generate dense vector representations of resume and job-description content.

The system represents:

```text
Resume → Embedding Vector
Job Description → Multiple Chunk Embeddings
```

These vectors allow the application to compare meaning rather than relying only on exact keyword matches.

---

## 🔎 4. Semantic Similarity Search

The application calculates **cosine similarity** between the resume embedding and job-description chunk embeddings.

The most relevant job-description chunks are retrieved based on similarity.

Conceptually:

```text
Job Description
      ↓
   Chunking
      ↓
Embedding for each chunk
      ↓
Cosine Similarity
      ↓
Top relevant chunks
      ↓
Context for AI analysis
```

This provides a semantic matching layer that can identify related concepts even when the resume and job description do not use exactly the same wording.

---

# 📊 5. ATS-Style Scoring

Jobbie-AI combines two major components:

### Keyword / Skill Coverage

The system identifies skill categories present in the job description and checks whether corresponding terms appear in the resume.

Examples include:

* Programming
* Data Analysis
* Customer Service
* Communication
* Software Engineering
* Cloud / DevOps
* CRM
* Leadership
* Reporting

### Semantic Relevance

The system calculates semantic similarity between the resume and relevant job-description content.

### Current scoring approach

```text
ATS Score
    │
    ├── 35% Keyword / Skill Coverage
    │
    └── 65% Semantic Relevance
```

The resulting score is constrained to an interpretable range and presented as an ATS-style score out of 100.

> **Important:** This is a project-defined ATS-style score and should not be interpreted as the actual score produced by a commercial Applicant Tracking System.

---

# 🤖 6. LLM-Powered Resume Analysis

After calculating the semantic and keyword components, Jobbie-AI uses a **Groq-hosted LLM** to generate a structured analysis.

The application currently supports configuring the model through environment variables, with:

```text
llama-3.1-8b-instant
```

as the default model.

The LLM produces:

### ATS Summary

Provides an interpretation of the candidate's overall fit.

### Matched Keywords

Highlights relevant skills and categories already represented in the resume.

### Missing Keywords

Identifies potentially relevant skills or categories that are not detected in the resume.

### Resume Rewrite Suggestions

Provides actionable recommendations for improving the resume.

### Tailored Cover Letter

Generates a job-specific cover letter based only on information available in the resume and job description.

The prompt explicitly instructs the model **not to invent experience**.

---

# 🔗 7. Job Description URL Fetching

Users can either:

1. Paste a job description manually, or
2. Provide a job posting URL.

The application includes a job scraping component that can retrieve job descriptions from supported job boards.

The fetched description can then be edited before analysis.

---

# 🔎 8. Recent Job Recommendations

Jobbie-AI can also retrieve recent job listings using the **Adzuna API**.

The system uses information extracted from the candidate's resume to construct a job-search query and retrieve relevant opportunities.

The application can display:

* Job title
* Company
* Location
* Salary information
* Posting date
* Job description snippet
* Application URL

This turns the application from a resume-analysis tool into a broader **job-fit assistant**.

---

# 🕘 9. Analysis History

Previous resume analyses can be stored and revisited.

Users can:

* View previous analyses
* Load previous results
* Delete individual analyses
* Clear analysis history

This functionality is handled through the project's history database module.

---

# 📥 10. Report Generation

Users can download their analysis in multiple formats:

* PDF
* TXT
* Markdown

The PDF report includes the generated analysis, resume filename, job description and ATS score.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     User Resume     │
                    │        PDF          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PDF Text Extract  │
                    │     pdfplumber      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Cleaning     │
                    │   & Preprocessing   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Resume Embedding    │
                    │ all-MiniLM-L6-v2    │
                    └──────────┬──────────┘
                               │
                               │
                               ▼
┌──────────────────┐   ┌─────────────────────┐
│ Job Description  │──►│ Chunking & Cleaning │
└──────────────────┘   └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Job Chunk Embeddings│
                       │ all-MiniLM-L6-v2    │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Cosine Similarity   │
                       │ Semantic Retrieval  │
                       └──────────┬──────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌──────────────────┐       ┌──────────────────┐
          │ Keyword Matching │       │ Semantic Score   │
          └────────┬─────────┘       └────────┬─────────┘
                   │                          │
                   └────────────┬─────────────┘
                                ▼
                     ┌─────────────────────┐
                     │   ATS-Style Score   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     Groq LLM        │
                     │ Analysis & Feedback  │
                     └──────────┬──────────┘
                                │
                                ▼
              ┌──────────────────────────────────┐
              │ Summary • Keywords • Suggestions │
              │       • Cover Letter             │
              └──────────────────────────────────┘
```

---

# 🛠️ Tech Stack

| Category             | Technology                 |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| UI Framework         | Streamlit                  |
| PDF Processing       | pdfplumber                 |
| NLP Embeddings       | Sentence Transformers      |
| Embedding Model      | `all-MiniLM-L6-v2`         |
| LLM                  | Groq API                   |
| LLM Model            | `llama-3.1-8b-instant`     |
| Vector Similarity    | Cosine Similarity          |
| Numerical Computing  | NumPy                      |
| Job Search API       | Adzuna API                 |
| Job URL Processing   | Custom job scraper         |
| Report Generation    | Python PDF generation      |
| Data Persistence     | Project history database   |
| External APIs        | Hugging Face, Groq, Adzuna |

---

# 📁 Project Structure

```text
Jobbie-AI/
│
├── app.py
│
├── job_scraper.py
├── job_fetcher.py
├── history_db.py
├── pdf_report.py
│
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── secrets.toml
│
└── ...
```

> The exact structure may change as the project continues to be developed.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Jobbie-AI.git

cd Jobbie-AI
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 API Configuration

Jobbie-AI requires API credentials for the external AI services.

Create:

```text
.streamlit/secrets.toml
```

and add:

```toml
HF_API_KEY = "your_huggingface_api_key"
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3.1-8b-instant"

ADZUNA_APP_ID = "your_adzuna_app_id"
ADZUNA_APP_KEY = "your_adzuna_app_key"
```

### API Usage

| API          | Purpose                             |
| ------------ | ----------------------------------- |
| Hugging Face | Generate text embeddings            |
| Groq         | Generate AI-powered resume analysis |
| Adzuna       | Retrieve relevant job listings      |

**Never commit API keys to GitHub.**

Make sure your secrets file is included in `.gitignore`:

```gitignore
.streamlit/secrets.toml
api.txt
.env
__pycache__/
*.pyc
```

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🚀 Usage

### Step 1 — Upload your resume

Upload a text-based PDF resume.

### Step 2 — Add a job description

Either:

```text
Paste Job Description
```

or:

```text
Enter Job URL → Fetch JD
```

### Step 3 — Analyze

Click:

```text
🚀 Analyze My Resume
```

### Step 4 — Review the results

The application generates:

```text
ATS Match Score
       ↓
ATS Summary
       ↓
Matched Keywords
       ↓
Missing Keywords
       ↓
Resume Rewrite Suggestions
       ↓
Tailored Cover Letter
```

### Step 5 — Download

Download the generated report as:

```text
PDF
TXT
Markdown
```

---

# 🧪 Example Workflow

```text
Resume.pdf
     +
AI Engineer Job Description
     │
     ▼
PDF Extraction
     │
     ▼
Text Preprocessing
     │
     ▼
Embedding Generation
     │
     ▼
Semantic Similarity
     │
     ├───────────────┐
     ▼               ▼
Keyword Matching   Semantic Score
     │               │
     └───────┬───────┘
             ▼
       ATS-Style Score
             │
             ▼
          Groq LLM
             │
             ▼
 ┌─────────────────────────┐
 │ ATS Summary             │
 │ Matched Keywords        │
 │ Missing Keywords        │
 │ Rewrite Suggestions     │
 │ Tailored Cover Letter   │
 └─────────────────────────┘
```

---

# 🔬 NLP / ML Concepts Demonstrated

This project demonstrates practical implementation of several AI concepts:

### Natural Language Processing

* Text extraction
* Text cleaning
* Text chunking
* Keyword matching
* Semantic representation

### Transformer Models

* Sentence embeddings
* `all-MiniLM-L6-v2`
* Hugging Face inference

### Semantic Search

* Vector representations
* Cosine similarity
* Top-k retrieval

### LLM Application Development

* Prompt engineering
* Structured LLM outputs
* Resume analysis
* Controlled generation

### Machine Learning Engineering

* API integration
* Data preprocessing
* Caching
* Error handling
* Application deployment architecture

---

# 🎯 Why I Built This

Traditional resume screening often relies heavily on keyword matching.

However, two pieces of text can express similar ideas using completely different wording.

For example:

```text
Resume:
"Built predictive models using Python."

Job Description:
"Experience developing machine learning solutions using Python."
```

A purely keyword-based system may miss some of the relationship between these statements.

Jobbie-AI therefore combines:

```text
Keyword Coverage
       +
Semantic Similarity
       +
LLM Analysis
```

to provide a more informative view of resume-job alignment.

---

# ⚠️ Limitations

Jobbie-AI is a **demonstration project and is still under development**.

The ATS score is a project-defined scoring mechanism and does **not represent the score of a specific commercial ATS platform**.

Current limitations include:

* Keyword matching uses predefined skill categories.
* Semantic scoring depends on the embedding model.
* PDF extraction quality depends on the structure of the uploaded PDF.
* Job-board scraping may fail when websites change their page structure or restrict automated access.
* LLM-generated suggestions should be reviewed by the user.
* Job recommendations depend on the availability and quality of external API results.

---

# 🔮 Future Improvements

Planned improvements include:

* [ ] Add more sophisticated skill/entity extraction
* [ ] Improve domain-specific skill detection
* [ ] Add multilingual resume support
* [ ] Add vector database support
* [ ] Experiment with alternative embedding models
* [ ] Add dedicated evaluation datasets
* [ ] Benchmark semantic matching performance
* [ ] Add resume section-level analysis
* [ ] Improve job recommendation ranking
* [ ] Add user authentication
* [ ] Improve database architecture
* [ ] Add automated unit and integration tests
* [ ] Deploy a production version

---

# 📈 Future Evaluation Strategy

A future version of Jobbie-AI can be evaluated using a manually labelled dataset containing:

```text
Resume
   +
Job Description
   ↓
Human-labelled Match Score
   ↓
Model-generated Match Score
   ↓
Evaluation
```

Potential evaluation metrics include:

* Precision
* Recall
* F1-score
* Mean Absolute Error
* Spearman correlation
* Ranking metrics such as NDCG

This would help measure whether semantic similarity and the ATS-style scoring approach correlate with human assessments of job fit.

---

# 🔒 Privacy & Security

Resumes can contain sensitive personal information.

When deploying or modifying this application:

* Do not commit user resumes to GitHub.
* Do not commit API keys.
* Avoid storing personally identifiable information unnecessarily.
* Review third-party API data handling policies.
* Use secure secret management in production.

---

# 👨‍💻 Author

**Sriregan Ganesa Rengan**

Master of Artificial Intelligence — RMIT University

Bachelor of Artificial Intelligence and Data Science — Bannari Amman Institute of Technology

### Connect

* LinkedIn: https://www.linkedin.com/in/srirengan-g
* GitHub: `YOUR_GITHUB_URL`

---

# ⭐ Project Status

🚧 **Active Development**

Jobbie-AI is a portfolio project demonstrating the integration of:

**NLP + Transformer Embeddings + Semantic Search + LLMs + APIs + Streamlit**

The application is continuously being improved with new features, better evaluation methods and additional AI capabilities.

---

## ⭐ If you find this project useful

Feel free to star the repository and explore the implementation.
