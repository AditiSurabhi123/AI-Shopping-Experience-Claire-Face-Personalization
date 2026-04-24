"""
Shared SSL context — tolerates the common macOS python.org installer issue
where Python doesn't find the system CA bundle.

Usage:
    from ssl_ctx import SSL_CTX
    conn = http.client.HTTPSConnection(host, timeout=10, context=SSL_CTX)
"""
import os
import ssl


def _make() -> ssl.SSLContext:
    # 1) certifi if installed — cleanest
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass

    # 2) Known-good cert bundle paths
    for p in (
        "/etc/ssl/cert.pem",
        "/usr/local/etc/openssl/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
        "/opt/homebrew/etc/ca-certificates/cert.pem",
    ):
        if os.path.exists(p):
            try:
                return ssl.create_default_context(cafile=p)
            except Exception:
                continue

    # 3) Prototype fallback: unverified TLS.
    # Fine for single-user local dev; NEVER ship to production like this.
    print("  [ssl] ⚠  No CA bundle found — using UNVERIFIED SSL (prototype only). "
          "Install certifi (`pip install certifi`) or run "
          "`/Applications/Python 3.x/Install Certificates.command` to fix.")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    return ctx


SSL_CTX: ssl.SSLContext = _make()
