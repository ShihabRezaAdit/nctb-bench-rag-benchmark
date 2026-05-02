# Contributing to NCTBench

Thank you for your interest in NCTBench. We welcome contributions that improve
reproducibility, extend evaluation, or fix bugs. Please read this guide before opening
a pull request.

---

## Scope of Contributions

| Type | Welcome? |
|---|---|
| Bug fixes in pipeline or evaluation scripts | Yes |
| New retriever or generator configurations | Yes |
| Additional evaluation metrics | Yes |
| Documentation improvements | Yes |
| New language or subject coverage | Yes (open an issue first) |
| Changes to the gold test set | No — requires paper revision |
| Changes to AMSV thresholds | No — alters reproducibility |

---

## How to Contribute

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Install** dependencies in a clean environment:
   ```bash
   conda create -n nctbench-dev python=3.9 -y
   conda activate nctbench-dev
   pip install -r requirements.txt
   ```

3. **Make your changes** — keep commits focused and atomic.

4. **Test** that the pipeline still runs end-to-end (at minimum Steps 3–6 and
   one evaluation configuration).

5. **Open a Pull Request** against `main` with:
   - A clear title and description of what changed and why
   - Reference to any related issue (`Fixes #123`)
   - Confirmation that existing results are not affected (or a note if they are)

---

## Reporting Issues

Please use GitHub Issues with the appropriate label:
- `bug` — something is broken
- `question` — clarification needed
- `enhancement` — new feature request
- `reproducibility` — results do not match the paper

Include your Python version, OS, GPU/CPU, and the full error traceback.

---

## Code Style

- Python 3.9+ compatible
- No external linting enforced, but follow PEP 8 conventions
- Keep functions short and well-named; avoid deep nesting
- No unnecessary comments — name things clearly instead

---

## Authors

Core authors of this work are listed in [CITATION.cff](CITATION.cff).
Contributions via pull request are credited in the git history.
