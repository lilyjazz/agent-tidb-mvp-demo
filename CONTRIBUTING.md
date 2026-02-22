# Contributing to TiDB Zero Agent Demo

First off, thanks for taking the time to contribute! 🎉

We welcome contributions of all forms, including code, documentation, bug reports, and feature requests.

## Development Setup

1.  **Fork the repository** on GitHub.
2.  **Clone your fork**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/agent-tidb-mvp-demo.git
    cd agent-tidb-mvp-demo
    ```
3.  **Set up the environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"  # Install dev dependencies
    ```
4.  **Configure environment variables**:
    ```bash
    cp .env.example .env
    # Edit .env with your API keys if needed
    ```

## Development Workflow

1.  **Create a branch** for your changes:
    ```bash
    git checkout -b feature/amazing-feature
    ```
2.  **Make your changes**. Keep them focused and concise.
3.  **Run checks** locally (using `Makefile` if available, or manual commands):
    ```bash
    # Linting
    ruff check .
    ruff format .

    # Type checking
    mypy src/
    ```
4.  **Commit your changes** with descriptive messages.
    ```bash
    git commit -m "feat: add amazing feature"
    ```
5.  **Push to your fork**:
    ```bash
    git push origin feature/amazing-feature
    ```
6.  **Open a Pull Request** against the `main` branch of the original repository.

## Pull Request Guidelines

*   **Description**: clearly describe what your PR does.
*   **Testing**: ensure your changes work as expected. Add tests if applicable.
*   **Linting**: ensure your code passes linting checks.
*   **Size**: keep PRs small and focused. Large PRs are harder to review.

## Reporting Bugs

Please check existing issues before opening a new one. If you find a new bug, open an issue using the Bug Report template. Include:

*   Steps to reproduce.
*   Expected vs. actual behavior.
*   Logs or screenshots if possible.

## Code Style

We use `ruff` for linting and formatting. Please ensure your code complies with the project's style.

Thank you for contributing!
