import hashlib
import hmac
import json
import os

USERS_FILE = "users.json"


def hash_password(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000
    ).hex()


def _load_users():
    """Users from Streamlit Secrets (cloud) if set, otherwise from users.json (your laptop)."""
    try:
        import streamlit as st
        raw = st.secrets.get("USERS_JSON")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None


def check_login(username, password):
    """Returns {"name", "groups", "admin"} if the login is right, otherwise None."""
    users = _load_users()
    if not users:
        return None
    name = (username or "").strip().lower()
    user = users.get(name)
    candidate = hash_password(password or "", user["salt"] if user else "00" * 16)
    if user and hmac.compare_digest(candidate, user["hash"]):
        return {"name": name, "groups": list(user["groups"]), "admin": bool(user.get("admin"))}
    return None