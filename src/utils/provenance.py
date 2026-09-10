"""Archive actual source bytes, including uncommitted files, for every new run."""
import hashlib
import json
import zipfile
from pathlib import Path


def snapshot_source(output):
    files = sorted({p for folder in ("src", "scripts", "configs", "tests")
                    for p in Path(folder).rglob("*")
                    if p.is_file() and p.suffix in {".py", ".sh", ".yaml"}} |
                   {Path("requirements.txt")})
    manifest = {}
    with zipfile.ZipFile(Path(output) / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            data = path.read_bytes()
            manifest[path.as_posix()] = hashlib.sha256(data).hexdigest()
            archive.writestr(path.as_posix(), data)
    (Path(output) / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return hashlib.sha256((Path(output) / "source.zip").read_bytes()).hexdigest()
