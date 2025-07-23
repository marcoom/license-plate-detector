# Contributing Guidelines

*Last updated: 23 July 2025*

Thank you for your interest in improving **License‑Plate‑Detector**! Follow the steps below to keep the workflow smooth and the codebase healthy.

---

## 1  Before You Start

- **Open an Issue first** for every bug, feature, or question. Wait for the maintainer’s green light before opening a pull request (PR).
- The project targets **Python ≥ 3.11** and is developed/tested on **Linux**. Other platforms may work but are not officially supported.

## 2  Workflow

| Step | Action |
|------|--------|
| 1 | **Fork** the repository and clone your fork. |
| 2 | Create a branch from `main`.<br>Use a clear name: `feat/<topic>`, `fix/<bug>`, `docs/<section>`, etc. |
| 3 | Commit changes following **[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)**. |
| 4 | **Push** and open a **PR** against `main`. Link the related Issue. |
| 5 | Ensure all automated checks pass. |

## 3  Quality Gates

| Check | Command | CI enforced |
|-------|---------|------------|
| Code style (Black + isort) & lint (Ruff) | `make lint` | ✅ |
| Static typing (mypy) | `make type-check` | ✅ |
| Tests | `make test` | ✅ |
| Coverage ≥ 80 % | `make test-coverage` | ❌ (currently advisory) |

> **Tip:** run `make install-dev` to set up a virtual environment with all dev tools.

## 4  Adding / Changing Code

- **Add tests** when you introduce or modify behaviour.
- Keep functions and modules small and focused.
- Update docstrings; public APIs should have clear type hints.

## 5  Documentation

The Sphinx docs rebuild automatically in CI. You only need to edit the reStructuredText/Markdown sources in `docs/source/` if you:
- add public functions/classes,
- change configuration options, or
- spot typos/translation issues.

## 6  Large Files & Models

- Avoid committing files larger than **100 MB** (videos, model weights).*  
  *Instead, provide a download link (e.g. Hugging Face Hub) or use Git LFS if essential.*

## 7  Legal & Licensing

- The project is released under **AGPL‑3.0**. By submitting code, you agree to license your contribution under the same terms.
- Make sure any dependency or snippet you add is **AGPL‑compatible**.

## 8  Security & Responsible Use

- Do **not** submit code that facilitates unlawful surveillance or violates data‑protection regulations.

## 9  Code of Conduct

Participation in this project is governed by the [Code of Conduct](./CODE_OF_CONDUCT.md). Please read it before contributing.

---

Minor wording updates and typo fixes are welcome at any time. Happy coding!
