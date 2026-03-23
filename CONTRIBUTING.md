# Contributing

1. Install dependencies: `uv sync --all-extras`
2. Make your changes.
3. Ensure pre-commit hooks pass: `prek run --all-files`
4. Ensure tests pass with no regression in coverage: `uv run pytest -m "not e2e" --cov --cov-report=term-missing`
   A coverage report is also posted automatically on each PR.
5. If you modified image generation or editing logic, run e2e tests locally
   (requires GCP credentials with Vertex AI access):
   ```
   uv run pytest -m e2e -v
   ```
   Paste the e2e test output in your PR description.
6. Submit a PR.
