"""Pinned download of the Pluribus hands from the PHH dataset into a cache outside git.

The phh-dataset repository holds several gigabytes of other corpora, so this fetches
only ``data/pluribus`` (plus the licence and README) with a blobless sparse git clone
at a pinned commit. Git verifies every object against the pinned tree; on top of that
the cache gets a ``MANIFEST.json`` (path -> SHA-256 of each file) whose own SHA-256
must equal :data:`PLURIBUS.content_sha256`. A cache that fails the check is never
used.

    python -m strategy.data.fetch            # fetch (or verify) the cache
    python -m strategy.data.fetch --verify   # verify only; exit 1 if missing or altered
    python -m strategy.data.fetch --where    # print the cache directory

The cache lives in ``$ARENA_DATA_CACHE`` or ``$XDG_CACHE_HOME/arena`` (default
``~/.cache/arena``), under ``phh-dataset/<commit>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PinnedSource:
    repo: str
    commit: str
    subdir: str
    tree: str
    """Git tree id of ``subdir`` at ``commit``."""
    files: int
    content_sha256: str
    """SHA-256 of the canonical ``MANIFEST.json`` (sorted ``path sha256`` lines)."""
    licence: str
    citation: str
    dataset_doi: str
    paper_doi: str

    def file_url(self, rel: str) -> str:
        return f"{self.repo.removesuffix('.git')}/blob/{self.commit}/{self.subdir}/{rel}"


PLURIBUS = PinnedSource(
    repo="https://github.com/uoftcprg/phh-dataset.git",
    commit="e47fbd5816372360bade4de5d712346fe1bb70f6",
    subdir="data/pluribus",
    tree="e0cd07e42ccae01d2ae9ca1b33ac06a5caf61580",
    files=10_000,
    content_sha256="1b36ee3ddead27b2494b3e8806df0f1177655fddc78481923f18093d70d2f1c9",
    licence="MIT (phh-dataset, Copyright 2024-2025 Universal, Open, Free, and Transparent Computer Poker Research Group)",
    citation="Brown and Sandholm (2019), Superhuman AI for multiplayer poker, Science 365(6456); "
             "hands converted to PHH by the phh-dataset project",
    dataset_doi="10.5281/zenodo.10796885",
    paper_doi="10.1126/science.aay2400",
)


class FetchError(RuntimeError):
    pass


def cache_root() -> Path:
    if os.environ.get("ARENA_DATA_CACHE"):
        return Path(os.environ["ARENA_DATA_CACHE"]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "arena"


def cache_dir(source: PinnedSource = PLURIBUS) -> Path:
    return cache_root() / "phh-dataset" / source.commit


def data_dir(source: PinnedSource = PLURIBUS) -> Path:
    return cache_dir(source) / source.subdir


def manifest_lines(root: Path) -> list[str]:
    files = sorted(p for p in root.rglob("*.phh") if p.is_file())
    return [f"{p.relative_to(root).as_posix()} {hashlib.sha256(p.read_bytes()).hexdigest()}" for p in files]


def manifest_hash(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest()


def verify(source: PinnedSource = PLURIBUS) -> Path:
    """The verified data directory; raises :class:`FetchError` if it is missing or altered."""
    root = data_dir(source)
    if not root.is_dir():
        raise FetchError(f"no cached copy at {root}; run python -m strategy.data.fetch")
    lines = manifest_lines(root)
    if len(lines) != source.files:
        raise FetchError(f"{root} holds {len(lines)} hands, expected {source.files}")
    got = manifest_hash(lines)
    if got != source.content_sha256:
        raise FetchError(f"{root} content hash {got} differs from the pinned {source.content_sha256}")
    return root


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise FetchError("git is required to fetch the dataset") from None
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from None
    return out.stdout.strip()


def fetch(source: PinnedSource = PLURIBUS, *, force: bool = False) -> Path:
    """Fetch into the cache (or reuse a verified copy) and return the data directory."""
    if not force:
        try:
            return verify(source)
        except FetchError:
            pass
    target = cache_dir(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".fetch-") as tmp:
        work = Path(tmp) / "repo"
        _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", "--sparse", source.repo, str(work))
        _git("sparse-checkout", "set", "--no-cone", f"/{source.subdir}/", "/LICENSE", "/README.md", cwd=work)
        _git("checkout", "--quiet", source.commit, cwd=work)
        tree = _git("rev-parse", f"HEAD:{source.subdir}", cwd=work)
        if tree != source.tree:
            raise FetchError(f"{source.subdir} at {source.commit} is tree {tree}, expected {source.tree}")
        shutil.rmtree(work / ".git")
        lines = manifest_lines(work / source.subdir)
        got = manifest_hash(lines)
        if len(lines) != source.files or got != source.content_sha256:
            raise FetchError(f"downloaded {len(lines)} hands with content hash {got}; "
                             f"pinned {source.files} with {source.content_sha256}")
        (work / "MANIFEST.json").write_text(json.dumps({
            "repo": source.repo, "commit": source.commit, "subdir": source.subdir, "tree": source.tree,
            "files": len(lines), "content_sha256": got, "licence": source.licence, "citation": source.citation,
            "dataset_doi": source.dataset_doi, "paper_doi": source.paper_doi, "manifest": lines,
        }, indent=1) + "\n", encoding="ascii")
        if target.exists():
            shutil.rmtree(target)
        work.rename(target)
    return verify(source)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true", help="only verify the cached copy")
    ap.add_argument("--where", action="store_true", help="print the cache directory and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if a verified copy exists")
    args = ap.parse_args(argv)
    if args.where:
        print(data_dir())
        return 0
    try:
        root = verify() if args.verify else fetch(force=args.force)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{PLURIBUS.files} Pluribus hands at {root} (content sha256 {PLURIBUS.content_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
