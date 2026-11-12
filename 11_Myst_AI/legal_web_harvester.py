import json, os, re, ssl, urllib.parse, urllib.request

ROOT = r"E:\Myst_Studio\11_Myst_AI"
CORPUS_PATH = os.path.join(ROOT, "data", "mr_ai_corpus.json")
TOPICS = ["Artificial intelligence", "Machine learning", "Transformer (deep learning)", "Python (programming language)", "SQLite"]

def fetch_summary(title):
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title) if False else None
    import urllib.parse
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "MystAI-MrAI/1.0 (educational)"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
        extract = data.get("extract", "")
        if extract:
            return f"{title}: {extract[:400]}"
    except Exception as e:
        return None
    return None

def main():
    corpus = []
    if os.path.exists(CORPUS_PATH):
        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                corpus = json.load(f)
        except Exception:
            corpus = []
    seen = {item.get("q", "").lower() for item in corpus if isinstance(item, dict)}
    added = 0
    for t in TOPICS:
        text = fetch_summary(t)
        if text:
            q = f"what is {t.lower()}"
            if q not in seen:
                corpus.append({"q": q, "a": text})
                seen.add(q)
                added += 1
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"Harvested {added} new entries. Total: {len(corpus)}")

if __name__ == "__main__":
    main()
