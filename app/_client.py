import os, requests

API = os.environ.get("SME_API", "http://localhost:8000")

def _json_or_text(resp):
    ct = resp.headers.get("content-type", "")
    if "application/json" in ct.lower():
        try:
            return resp.json()
        except Exception:
            pass
    # Fallback: return a structured error with any text body
    return {
        "status": "error",
        "status_code": resp.status_code,
        "text": resp.text[:2000] if hasattr(resp, "text") else "",
    }

def api_versions():
    try:
        r = requests.get(f"{API}/versions", timeout=5)
        return _json_or_text(r)
    except Exception as e:
        return {"status":"error","message":f"versions request failed: {e}"}

def api_upload(fileobj):
    try:
        r = requests.post(f"{API}/upload",
                          files={"file": (fileobj.name, fileobj.getvalue(), "application/octet-stream")},
                          timeout=120)
        return _json_or_text(r)
    except Exception as e:
        return {"status":"error","message":f"upload failed: {e}"}

def api_run(payload):
    try:
        r = requests.post(f"{API}/run-benchmark", json=payload, timeout=3600)
        return _json_or_text(r)
    except Exception as e:
        return {"status":"error","message":f"run failed: {e}"}

def api_results():
    try:
        r = requests.get(f"{API}/results", timeout=10)
        return _json_or_text(r)
    except Exception as e:
        return {"status":"error","message":f"results failed: {e}"}
