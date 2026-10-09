"""Scan publication files without printing matched secret values."""

import ast
import ipaddress
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_HOSTS = {
    "github.com", "raw.githubusercontent.com", "api.deepseek.com",
    "stackoverflow.com", "www.apache.org", "apache.org",
    "opensource.org", "www.opensource.org", "www.gnu.org",
    "www.python.org", "www.w3.org",
}
SECRET_PATTERNS = {
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "github_token": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})",
    "cloud_access_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "api_key": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b",
    "credential_url": r"https?://[^\s/]+:[^\s/@]+@",
    "personal_path": r"(?:/(?:Users|home)/[A-Za-z0-9_.-]+/|/volume[0-9]+/)",
}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".parquet", ".pkl", ".pt", ".pth", ".ckpt"}


def files_to_check():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError("Initialize the release Git repository before running the audit")
    return sorted(set(result.stdout.splitlines()))


def scan():
    findings = []
    files = files_to_check()
    for name in files:
        path = ROOT / name
        if path.is_symlink():
            findings.append((name, 0, "symlink"))
            continue
        if path.suffix.lower() in SENSITIVE_SUFFIXES or (path.name.startswith(".env") and path.name != ".env.example"):
            findings.append((name, 0, "sensitive_file"))
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append((name, 0, "binary_file"))
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in SECRET_PATTERNS.items():
                if re.search(pattern, line):
                    findings.append((name, number, label))
            for address in re.findall(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", line):
                try:
                    parsed = ipaddress.ip_address(address)
                    if parsed.is_private and not parsed.is_loopback:
                        findings.append((name, number, "private_network_address"))
                except ValueError:
                    pass
            for url in re.findall(r"https?://[^\s\"'<>]+", line):
                host = urlsplit(url.rstrip(").,;`")).hostname
                if host and host not in PUBLIC_HOSTS:
                    findings.append((name, number, "unreviewed_url_host"))
        if path.suffix == ".py":
            tree = ast.parse(text, filename=name)
            pattern_catalogs = {
                id(node.value) for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "SECRET_PATTERNS"
                        for target in node.targets)
            }
            # Static credentials need separate review even if they do not match
            # a provider-specific token pattern. Empty environment fallbacks pass.
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    if id(node) in pattern_catalogs:
                        continue
                    pairs = zip(node.keys, node.values)
                elif isinstance(node, ast.Assign):
                    pairs = [(ast.Constant(target.id), node.value)
                             for target in node.targets if isinstance(target, ast.Name)]
                else:
                    continue
                for key, value in pairs:
                    if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                        continue
                    if re.search(r"(?:password|secret|api_key|access_key)", key.value, re.I):
                        if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value:
                            findings.append((name, getattr(value, "lineno", 0), "static_credential"))
    for name, number, label in sorted(set(findings)):
        print(f"{name}:{number}: {label}")
    if findings:
        print(f"Publication audit failed: {len(set(findings))} findings; matched values were withheld")
        return 1
    print(f"Publication audit passed: {len(files)} text files; no detected credentials, private addresses, or unreviewed URL hosts")
    return 0


if __name__ == "__main__":
    sys.exit(scan())
