"""Record mixed batching in legacy experiment configs that lack the field.

Existing values are preserved. Originals and a manifest are backed up before
writing. Re-run after older training processes finish to cover new checkpoints.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import tempfile


def backfill(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    pending = []
    preserved = Counter()
    skipped = []
    paths = sorted(root.rglob("experiment_config.json"))
    paths += sorted(root.glob("*/*/run_metadata.json"))
    for path in paths:
        try:
            original = path.read_bytes()
            payload = json.loads(original)
            config = (
                payload["experiment_config"]
                if path.name == "run_metadata.json" else payload
            )
            if not isinstance(config, dict):
                raise ValueError("Experiment config is not a JSON object.")
            if "alignment_batching" in config:
                preserved[str(config["alignment_batching"])] += 1
                continue
            config["alignment_batching"] = "mixed"
            updated = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            if original.endswith(b"\n"):
                updated += b"\n"
            pending.append((path, original, updated))
        except (OSError, ValueError, KeyError, TypeError) as error:
            skipped.append({"path": str(path.relative_to(root)), "reason": str(error)})

    report = {
        "root": str(root),
        "value": "mixed",
        "updated": {},
        "preserved": dict(preserved),
        "skipped": skipped,
        "backup": None,
    }
    if not pending:
        return report

    backup_dir = root / ".metadata_backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir.mkdir(parents=True)
    archive_path = backup_dir / "originals.tar.gz"
    manifest = []
    with tarfile.open(archive_path, "w:gz") as archive:
        for path, original, updated in pending:
            relative = str(path.relative_to(root))
            info = tarfile.TarInfo(relative)
            info.size = len(original)
            info.mode = stat.S_IMODE(path.stat().st_mode)
            archive.addfile(info, io.BytesIO(original))
            manifest.append({
                "path": relative,
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "updated_sha256": hashlib.sha256(updated).hexdigest(),
                "status": "pending",
            })
    manifest_path = backup_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = Counter()
    for (path, original, updated), entry in zip(pending, manifest):
        # Do not overwrite a concurrent Trainer write using an older snapshot.
        if path.read_bytes() != original:
            entry["status"] = "changed_during_backfill"
            skipped.append({"path": entry["path"], "reason": entry["status"]})
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".batching-", delete=False) as output:
                temporary = Path(output.name)
                output.write(updated)
                output.flush()
                os.fsync(output.fileno())
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
            entry["status"] = "updated"
            counts[path.name] += 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report["updated"] = dict(counts)
    report["backup"] = str(archive_path)
    (backup_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="results")
    args = parser.parse_args()
    print(json.dumps(backfill(args.root), ensure_ascii=False, indent=2))
