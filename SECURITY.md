# Security

Keep credentials in environment variables or an untracked local `.env` file.
Use `.env.example` only as a configuration template. Keep market data, model
artifacts, and generated outputs outside Git.

Factor expressions are evaluated as code in parts of the research workflow, and
saved builders use Python pickle. Load expressions and pickle files only from
sources you trust.

Before committing, run `python scripts/audit_publication.py` to check tracked
files for publication-sensitive content. The automated scan complements review;
it is not a guarantee that every possible secret can be detected.

For a security report, use GitHub's private vulnerability reporting feature on
this repository when available. Do not include credentials or private datasets
in a public issue.
