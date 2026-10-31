"""Check project prose for untranslated Vietnamese and broken local Markdown links."""

import argparse
import ast
import io
import json
import re
import subprocess
import tokenize
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTORS = (
    "Hoàng Hữu Hoan",
    "Giáp Hoàng Gia Khánh",
    "Lê Đức Anh",
    "Trần Mạnh Đức",
    "Lê Đức Thành",
    "Nguyễn Nam Khánh",
    "Gia Khánh",
    "Đức Anh",
    "Nam Khánh",
    "Thành",
    "Đức",
    "Khánh",
)
ENGLISH_TERMS = ("moiré",)
VIETNAMESE = re.compile(r"[ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯư\u1ea0-\u1ef9]")
LINK = re.compile(r"!?\[[^\]]*\]\(([^\n)]+)\)")


def python_prose(source: str) -> str:
    comments = []
    try:
        comments = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        ]
    except (tokenize.TokenError, IndentationError):
        pass
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(comments)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            docstring = ast.get_docstring(node)
            if docstring:
                comments.append(docstring)
    return "\n".join(comments)


def prose(path: Path, source: str) -> str:
    if path.suffix == ".py":
        return python_prose(source)
    if path.suffix == ".ipynb":
        cells = json.loads(source).get("cells", [])
        return "\n".join(
            (
                "".join(cell.get("source", []))
                if cell.get("cell_type") == "markdown"
                else python_prose("".join(cell.get("source", [])))
            )
            for cell in cells
        )
    return source


def anchors(source: str) -> set[str]:
    result = set()
    counts = {}
    for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", source, flags=re.MULTILINE):
        title = re.sub(r"<[^>]+>", "", title).lower()
        identifier = re.sub(r"[^\w\-\s]", "", title).replace(" ", "-")
        index = counts.get(identifier, 0)
        counts[identifier] = index + 1
        result.add(f"{identifier}-{index}" if index else identifier)
    return result


def check(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    text = unicodedata.normalize("NFC", prose(path, source))
    for name in sorted(CONTRIBUTORS + ENGLISH_TERMS, key=len, reverse=True):
        text = text.replace(name, "")
    errors = []
    if VIETNAMESE.search(text):
        errors.append(f"{path.relative_to(ROOT)}: untranslated Vietnamese prose")
    if path.suffix == ".md":
        for value in LINK.findall(source):
            target = (
                value[1 : value.index(">")]
                if value.startswith("<")
                else value.split(' "', 1)[0]
            )
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            destination = (
                (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            )
            if not destination.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing local link {target}")
            elif parsed.fragment and destination.suffix == ".md":
                identifiers = anchors(destination.read_text(encoding="utf-8-sig"))
                if unquote(parsed.fragment) not in identifiers:
                    errors.append(f"{path.relative_to(ROOT)}: missing heading {target}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", type=Path)
    args = parser.parse_args()
    if args.files:
        paths = [path.resolve() for path in args.files]
    else:
        names = (
            subprocess.check_output(
                ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                cwd=ROOT,
            )
            .decode()
            .split("\0")
        )
        paths = sorted(
            {
                ROOT / name
                for name in names
                if name
                and Path(name).suffix
                in {".md", ".py", ".ipynb", ".yaml", ".yml", ".toml", ".txt"}
            }
        )
    errors = [error for path in paths if path.is_file() for error in check(path)]
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Documentation checks passed for {len(paths)} project text files.")


if __name__ == "__main__":
    main()
