# Contributing to GPNN

Thank you for your interest in contributing! Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

Open an issue with:
- Clear title and description
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Environment details (Python version, PyTorch version, OS)

### Pull Requests

1. **Fork and clone** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Set up environment**: `uv sync` or `pip install -e ".[dev]"`
4. **Make your changes** following these guidelines:
   - Use type hints and docstrings (Google style)
   - Follow PEP 8 (max line length: 88)
   - Write tests for new functionality
   - Update documentation as needed
5. **Run tests**: `pytest tests/`
6. **Check linting**: `ruff check . && ruff format .`
7. **Commit**: Use clear messages (e.g., "Add periodic kernel")
8. **Push and open PR** with description of changes

## Code Standards

- **Type hints**: Required on all functions
- **Docstrings**: Google style with examples
- **Tests**: Cover new features and edge cases
- **Error handling**: Use custom exceptions from `exceptions.py`

Example:

```python
def new_function(x: torch.Tensor) -> torch.Tensor:
    """
    Brief description.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor
    
    Example:
        >>> result = new_function(torch.randn(5))
    """
    return x * 2
```

## Questions?

Open an issue or email: abhi.mamg@gmail.com

Thank you for contributing! 🎉
