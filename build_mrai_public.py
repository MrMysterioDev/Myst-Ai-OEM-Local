import os, py_compile

ROOT = r"E:\Myst_Studio\11_Myst_AI"
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

server_file = os.path.join(ROOT, "myst_ai_server.py")
if not os.path.exists(server_file):
    raise SystemExit(f"Missing server: {server_file}")
py_compile.compile(server_file, doraise=True)

corpus_path = os.path.join(ROOT, "data", "mr_ai_corpus.json")
if not os.path.exists(corpus_path):
    import json
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump([], f)

print("[OK] Myst_Studio build passed")
