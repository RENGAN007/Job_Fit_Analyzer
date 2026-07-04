import requests
import re
from datetime import datetime, timezone


ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs"

BROAD_FALLBACKS = [
    "software engineer",
    "developer",
    "analyst",
    "engineer",
    "manager",
]

# Australian state capitals — used to detect if extracted location is a suburb or city
AU_CAPITALS = {"sydney", "melbourne", "brisbane", "perth", "adelaide",
               "canberra", "hobart", "darwin"}


def _extract_resume_info(resume_text: str, groq_client, model: str) -> dict:
    """Use Groq to extract job title, skills, and location from the resume."""
    prompt = (
        "Extract the following from this resume:\n"
        "1. Most likely job title the person is applying for (1-4 words)\n"
        "2. Top 3 technical skills as comma-separated list\n"
        "3. Current suburb or city where the person lives (look for address, "
        "   suburb, city, postcode — return ONLY the suburb or city name, "
        "   e.g. \'Parramatta\' or \'Melbourne\'. If unsure, return \'\')\n\n"
        "Reply in EXACTLY this format, three lines only:\n"
        "TITLE: <job title>\n"
        "SKILLS: <skill1>, <skill2>, <skill3>\n"
        "LOCATION: <suburb or city or empty>\n\n"
        f"RESUME:\n{resume_text[:3000]}"
    )
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    text = response.choices[0].message.content.strip()

    title_match    = re.search(r"TITLE:\s*(.+)",    text)
    skills_match   = re.search(r"SKILLS:\s*(.+)",   text)
    location_match = re.search(r"LOCATION:\s*(.+)", text)

    raw_title    = title_match.group(1).strip()    if title_match    else "software engineer"
    skills       = skills_match.group(1).strip()   if skills_match   else ""
    raw_location = location_match.group(1).strip() if location_match else ""

    # Clean title
    title = re.sub(r'[<>"\'\n\r]', "", raw_title).strip()
    title = " ".join(title.split()[:4]) or "software engineer"

    # Clean location — remove postcodes, country names, extra words
    location = re.sub(r'[<>"\'\n\r]', "", raw_location).strip()
    location = re.sub(r'\b\d{4}\b', "", location).strip()  # remove 4-digit postcodes
    location = re.sub(r'(?i)\b(australia|au|nsw|vic|qld|wa|sa|act|tas|nt)\b', "", location).strip()
    location = " ".join(location.split())  # normalise whitespace

    # Reject clearly bad values
    bad_values = {"", "n/a", "none", "unknown", "not specified", "not provided", "na"}
    if location.lower() in bad_values or len(location) < 2:
        location = ""

    return {"title": title, "skills": skills, "location": location}


def _search_adzuna(what: str, app_id: str, app_key: str,
                   country: str, max_days_old: int, results: int,
                   where: str = "", distance: int = 20) -> list:
    """Single Adzuna API call, returns raw results list."""
    params = {
        "app_id":           app_id,
        "app_key":          app_key,
        "results_per_page": results,
        "what":             what,
        "max_days_old":     max_days_old,
        "sort_by":          "date",
        "content-type":     "application/json",
    }
    if where:
        params["where"]    = where
        params["distance"] = distance

    url  = f"{ADZUNA_BASE}/{country}/search/1"
    resp = requests.get(url, params=params, timeout=15)

    if resp.status_code == 401:
        raise ValueError("Adzuna authentication failed. Check ADZUNA_APP_ID and ADZUNA_APP_KEY.")
    if resp.status_code != 200:
        raise ValueError(f"Adzuna API error {resp.status_code}: {resp.text[:200]}")

    return resp.json().get("results", [])


def _format_salary(job: dict) -> str:
    min_s = job.get("salary_min")
    max_s = job.get("salary_max")
    if min_s and max_s:
        return f"${int(min_s):,} – ${int(max_s):,}"
    elif min_s:
        return f"${int(min_s):,}+"
    return "Not specified"


def _parse_jobs(raw_jobs: list) -> list:
    jobs = []
    for j in raw_jobs:
        created_raw = j.get("created", "")
        try:
            dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - dt).days
            posted_label = "Today" if days_ago == 0 else f"{days_ago}d ago"
        except Exception:
            posted_label = ""
        jobs.append({
            "title":    j.get("title", "N/A"),
            "company":  j.get("company", {}).get("display_name", "N/A"),
            "location": j.get("location", {}).get("display_name", "N/A"),
            "salary":   _format_salary(j),
            "posted":   posted_label,
            "url":      j.get("redirect_url", "#"),
            "snippet":  (j.get("description", "")[:180].strip() + "..."),
        })
    return jobs


def fetch_recent_jobs(
    resume_text: str,
    groq_client,
    groq_model: str,
    adzuna_app_id: str,
    adzuna_app_key: str,
    country: str = "au",
    results: int = 8,
) -> tuple[list[dict], str, str, str]:
    """
    Fetch Adzuna job listings near the resume's location.

    Search strategy (tries each until results found):
      With location:
        1. Exact suburb, 20km radius, 14 days
        2. Exact suburb, 20km radius, 30 days
        3. Exact suburb, 40km radius, 30 days  (widen slightly if too few)
      Without location (or if location search fails):
        4. Full title, no location, 14 days
        5. Broad fallbacks, no location, 30 days

    Returns: (jobs_list, job_title, skills, location_used)
    """
    extracted = _extract_resume_info(resume_text, groq_client, groq_model)
    job_title = extracted["title"]
    skills    = extracted["skills"]
    location  = extracted["location"]

    first_word = job_title.split()[0] if job_title else ""
    title_queries = list(dict.fromkeys(filter(None, [
        job_title,
        first_word if first_word.lower() != job_title.lower() else None,
    ] + BROAD_FALLBACKS)))

    raw_jobs = []

    # ── Phase 1: Search near resume location ──────────────────────────
    if location:
        for query in title_queries:
            for days, dist in [(14, 20), (30, 20), (30, 40)]:
                raw_jobs = _search_adzuna(
                    what=query, app_id=adzuna_app_id, app_key=adzuna_app_key,
                    country=country, max_days_old=days, results=results,
                    where=location, distance=dist,
                )
                if raw_jobs:
                    break
            if raw_jobs:
                break

    # ── Phase 2: Fallback — no location filter ─────────────────────────
    if not raw_jobs:
        for query in title_queries:
            for days in [14, 30]:
                raw_jobs = _search_adzuna(
                    what=query, app_id=adzuna_app_id, app_key=adzuna_app_key,
                    country=country, max_days_old=days, results=results,
                )
                if raw_jobs:
                    break
            if raw_jobs:
                break

    return _parse_jobs(raw_jobs), job_title, skills, location
