# CLAUDE.md - Coding Guidelines for NPC Project

## Build and Run Commands
```
poetry install              # Install dependencies
poetry run python -m npc    # Run the application
poetry run pytest           # Run tests (when tests are added)
poetry shell                # Activate virtual environment
```

## Testing
No formal testing framework is established yet. Examples can be found in:
- `src/npc/agent/examples/mcp_test.py`

## Code Style Guidelines
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- **Imports**: Group by (1) std lib, (2) third-party, (3) project modules, with blank lines between
- **Typing**: Use type hints for all function parameters and return values
- **Formatting**: 4-space indentation, ~80-100 char line length
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **Error handling**: Use specific exception types, implement fallback mechanisms
- **Comments**: Add explanatory comments for complex logic, use TODO for future work

## Development Setup
```
poetry install
poetry run nbstripout --install  # For Jupyter notebook output stripping
```