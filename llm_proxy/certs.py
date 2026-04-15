"""Certificate helpers for llm-proxy HTTPS."""

import hashlib
import os
import shutil
import ssl
import subprocess
import tempfile


def ensure_openssl():
    if shutil.which("openssl"):
        return
    raise SystemExit("openssl is required to generate llm-proxy certificates")


def generate_server_cert(state_dir, host, ip=None):
    ensure_openssl()
    certs_dir = os.path.join(os.path.expanduser(state_dir), "certs")
    os.makedirs(certs_dir, exist_ok=True)

    server_key = os.path.join(certs_dir, "server.key")
    server_csr = os.path.join(certs_dir, "server.csr")
    server_cert = os.path.join(certs_dir, "server.crt")
    host = (host or "llm-proxy").strip()

    ext_lines = [
        "basicConstraints=CA:FALSE",
        "keyUsage=digitalSignature,keyEncipherment",
        "extendedKeyUsage=serverAuth",
        f"subjectAltName=DNS:{host}",
    ]
    if ip:
        ext_lines[-1] += f",IP:{ip}"

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write("\n".join(ext_lines) + "\n")
        ext_path = f.name

    try:
        _run(["openssl", "genrsa", "-out", server_key, "2048"])
        _run([
            "openssl", "req", "-new",
            "-key", server_key,
            "-out", server_csr,
            "-subj", f"/CN={host}/O=llm-proxy",
        ])
        _run([
            "openssl", "x509", "-req", "-signkey", server_key,
            "-in", server_csr,
            "-out", server_cert,
            "-days", "825",
            "-sha256",
            "-extfile", ext_path,
        ])
    finally:
        _safe_unlink(server_csr)
        _safe_unlink(ext_path)

    return {"cert": server_cert, "key": server_key, "fingerprint": certificate_fingerprint(server_cert)}


def certificate_fingerprint(cert_path):
    with open(cert_path, "r", encoding="utf-8") as f:
        pem = f.read()
    der = ssl.PEM_cert_to_DER_cert(pem)
    digest = hashlib.sha256(der).hexdigest()
    return ":".join(digest[i:i + 2] for i in range(0, len(digest), 2))


def make_ssl_context(cert_path, key_path):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    return context


def _run(argv):
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or str(exc)
        raise SystemExit(f"command failed: {' '.join(argv)}\n{details}") from exc


def _safe_unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
