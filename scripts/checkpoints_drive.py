"""Model checkpoints on Google Drive: fetch them (anyone), or publish new ones (the maintainer).

    python scripts/checkpoints_drive.py fetch [NAME ...]          # -> checkpoints/NAME/, checked against the sha256
    python scripts/checkpoints_drive.py list
    python scripts/checkpoints_drive.py upload ARCHIVE.tar.gz --account you@gmail.com [--folder "robot-learning checkpoints"]

``checkpoints/manifest.json`` lists every archive: its Drive file id, size, sha256 and what is inside. ``fetch`` needs no
login (the files are shared by link). ``upload`` needs a Google account with Drive access in gcloud
(``gcloud auth login --enable-gdrive-access --no-activate``); it uploads the archive into a Drive folder, shares it
"anyone with the link can view" and records it in the manifest. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
MANIFEST = CKPT / "manifest.json"
API = "https://www.googleapis.com/drive/v3"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
DOWNLOAD = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"archives": {}}


# ------------------------------------------------------------------------------------------------ fetch
def fetch(names: list[str]) -> None:
    man = load_manifest()["archives"]
    for name in names or sorted(man):
        e = man[name]
        dst = CKPT / name
        if (dst / "MANIFEST.json").exists():
            print(f"{name}: already in {dst}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tgz = Path(tmp) / f"{name}.tar.gz"
            print(f"{name}: downloading {e['bytes'] / 1e6:.1f} MB")
            urllib.request.urlretrieve(DOWNLOAD.format(id=e["drive_id"]), tgz)
            got = sha256(tgz)
            if got != e["sha256"]:
                sys.exit(f"{name}: sha256 mismatch ({got} != {e['sha256']}); not extracted")
            with tarfile.open(tgz) as t:
                t.extractall(CKPT, filter="data")
        print(f"{name}: {dst} ({', '.join(e.get('policies', []))})")


# ------------------------------------------------------------------------------------------------ upload
def _token(account: str) -> str:
    return subprocess.check_output(["gcloud", "auth", "print-access-token", account], text=True).strip()


def _call(method: str, url: str, token: str, body: bytes | None = None, ctype: str = "application/json",
          headers: dict | None = None):
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": ctype, **(headers or {})})
    with urllib.request.urlopen(req) as r:
        return r.headers, (json.loads(r.read() or b"{}") if r.headers.get("Content-Type", "").startswith("application/json") else r.read())


def _folder(token: str, name: str) -> str:
    q = urllib.parse.quote(f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false")
    _, r = _call("GET", f"{API}/files?q={q}&fields=files(id,name)", token)
    if r.get("files"):
        return r["files"][0]["id"]
    _, r = _call("POST", f"{API}/files?fields=id", token,
                 json.dumps({"name": name, "mimeType": "application/vnd.google-apps.folder"}).encode())
    return r["id"]


def upload(archive: Path, account: str, folder: str) -> None:
    token = _token(account)
    parent = _folder(token, folder)
    meta = json.dumps({"name": archive.name, "parents": [parent]}).encode()
    h, _ = _call("POST", f"{UPLOAD}?uploadType=resumable&fields=id", token, meta,
                 headers={"X-Upload-Content-Type": "application/gzip", "X-Upload-Content-Length": str(archive.stat().st_size)})
    _, r = _call("PUT", h["Location"], token, archive.read_bytes(), ctype="application/gzip")
    fid = r["id"]
    _call("POST", f"{API}/files/{fid}/permissions?fields=id", token, json.dumps({"role": "reader", "type": "anyone"}).encode())
    name = archive.name.removesuffix(".tar.gz")
    with tarfile.open(archive) as t:
        names = t.getnames()
        inner = json.loads(t.extractfile(f"{name}/MANIFEST.json").read())
    man = load_manifest()
    man["archives"][name] = {"drive_id": fid, "url": f"https://drive.google.com/file/d/{fid}/view",
                             "bytes": archive.stat().st_size, "sha256": sha256(archive), "files": len(inner),
                             "policies": sorted({n.split("/")[1] for n in names if n.endswith("/policy.pt")}),
                             "world_model": f"{name}/dreamer/world_model.pt" in names}
    CKPT.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({name: man["archives"][name]}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("names", nargs="*")
    sub.add_parser("list")
    u = sub.add_parser("upload")
    u.add_argument("archive", type=Path)
    u.add_argument("--account", required=True)
    u.add_argument("--folder", default="robot-learning checkpoints")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.names)
    elif a.cmd == "list":
        for k, v in load_manifest()["archives"].items():
            print(f"{k:22s} {v['bytes'] / 1e6:6.1f} MB  {', '.join(v['policies'])}")
    else:
        upload(a.archive, a.account, a.folder)


if __name__ == "__main__":
    main()
