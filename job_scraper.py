from bs4 import BeautifulSoup
import requests
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def _clean_scraped(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def scrape_linkedin(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        raise ValueError(
            f"LinkedIn returned status {resp.status_code}. "
            "The job may be expired or login-gated."
        )
    soup = BeautifulSoup(resp.text, "html.parser")

    selectors = [
        {"class": "show-more-less-html__markup"},
        {"class": "description__text"},
        {"class": "job-description"},
    ]
    for attrs in selectors:
        block = soup.find("div", attrs)
        if block:
            return _clean_scraped(block.get_text(separator="\n"))

    main = soup.find("main") or soup.find("body")
    if main:
        texts = [t.get_text(" ") for t in main.find_all(["p", "li"])]
        result = "\n".join(t.strip() for t in texts if len(t.strip()) > 30)
        if result:
            return _clean_scraped(result)

    raise ValueError(
        "Could not extract job description from LinkedIn. "
        "The page may require login."
    )

def scrape_seek(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Seek returned status {resp.status_code}.")
    soup = BeautifulSoup(resp.text, "html.parser")

    block = soup.find("div", {"data-automation": "jobDescription"})
    if block:
        return _clean_scraped(block.get_text(separator="\n"))

    block = soup.find("div", {"class": re.compile(r"job.*description", re.I)})
    if block:
        return _clean_scraped(block.get_text(separator="\n"))

    raise ValueError("Could not extract job description from Seek.")

def scrape_indeed(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Indeed returned status {resp.status_code}.")
    soup = BeautifulSoup(resp.text, "html.parser")

    block = soup.find("div", {"id": "jobDescriptionText"})
    if block:
        return _clean_scraped(block.get_text(separator="\n"))

    block = soup.find("div", {"class": re.compile(r"jobsearch-jobDescription", re.I)})
    if block:
        return _clean_scraped(block.get_text(separator="\n"))

    raise ValueError("Could not extract job description from Indeed.")

def scrape_job_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        raise ValueError("Please enter a valid URL starting with http:// or https://")

    if "linkedin.com" in url:
        return scrape_linkedin(url)
    elif "seek.com" in url:
        return scrape_seek(url)
    elif "indeed.com" in url:
        return scrape_indeed(url)
    else:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            raise ValueError(f"Could not fetch URL (status {resp.status_code}).")
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["nav", "header", "footer", "script", "style"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
        cleaned = _clean_scraped(text)
        if len(cleaned) < 100:
            raise ValueError("Could not extract meaningful job description from this URL.")
        return cleaned