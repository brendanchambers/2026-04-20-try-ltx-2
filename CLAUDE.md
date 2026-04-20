
# Do not modify

- Do not modify any files in the `examples` directory.
- Do not modify `README.md` without asking.
- Do not modify any data or logged requests.

# Python environment

- Use `uv` for all Python dependency management and environment tasks.
- Do not use `pip`, `venv`, or `poetry`.
- Always use `uv add <package>` to add dependencies.
- Use `uv run <command>` to execute scripts.
- Use `uv sync` to install dependencies from `pyproject.toml`.
- When creating a new project, use `uv init`.

# Style

- Centralize variables like `model_name` at the top of files.
- Single files, minimal imports, extremely clear and concise. 
- Use `rich` for formatting terminal outputs.

# Usage

- Launch and run in a terminal on an Apple MacOS laptop.