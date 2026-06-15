import os, re

BASE = r"E:\wolkplace\enterprise-rag-cs"
configs = [
    "query_rewriter/config.py",
    "hybrid_retriever/config.py",
    "context_compressor/config.py",
    "hallucination_guard/config.py",
    "safety_enforcer/config.py",
    "escalation_handler/config.py",
    "observability/config.py",
]

for rel in configs:
    path = os.path.join(BASE, rel)
    if not os.path.exists(path):
        print(f"  SKIP {rel}: not found")
        continue
    with open(path, encoding="utf-8") as f:
        c = f.read()

    # Fix 1: _find_env_file returns None instead of raising
    c = c.replace(
        "    raise FileNotFoundError(\"Cannot find .env file\")",
        "    return None"
    ).replace(
        "raise FileNotFoundError(\"Cannot find .env file. Expected at E:/wolkplace/.env\")",
        "return None"
    ).replace(
        "    raise FileNotFoundError(\"Cannot find .env file. Expected at E:/wolkplace/.env\")",
        "    return None"
    )

    # Fix 2: _load_dotenv handles None path
    old = """def _load_dotenv() -> None:
    env_path = _find_env_file()
    if not env_path.exists():
        return"""
    new = """def _load_dotenv() -> None:
    env_path = _find_env_file()
    if env_path is None or not env_path.exists():
        return"""
    c = c.replace(old, new).replace(
        "if not env_path.exists():",
        "if env_path is None or not env_path.exists():"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(c)
    print(f"  FIXED {rel}")

print("\nDone")
