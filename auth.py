import hashlib
import hmac
import json
import os

USERS_FILE = "users.json"


def hash_password(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000
    ).hex()


def check_login(username, password):
    """Returns {"name", "groups", "admin"} if the login is right, otherwise None."""
    if not os.path.exists(USERS_FILE):
        return None
    with open(USERS_FILE, encoding="utf-8") as f:
        users = json.load(f)
    name = (username or "").strip().lower()
    user = users.get(name)
    candidate = hash_password(password or "", user["salt"] if user else "00" * 16)
    if user and hmac.compare_digest(candidate, user["hash"]):
        return {"name": name, "groups": list(user["groups"]), "admin": bool(user.get("admin"))}
    return None