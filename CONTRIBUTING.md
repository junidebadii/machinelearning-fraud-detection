# Contributing to Fraud Detection ML

Thank you for your interest in contributing to the Fraud Detection ML project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Use the GitHub issue tracker
- Include detailed steps to reproduce
- Provide system information and error logs
- Use the bug report template

### Suggesting Enhancements

- Use the GitHub issue tracker
- Describe the enhancement clearly
- Explain why this enhancement would be useful
- Use the enhancement request template

### Code Contributions

- Fork the repository
- Create a feature branch
- Make your changes
- Add tests for new functionality
- Ensure all tests pass
- Submit a pull request

## Development Setup

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- Git for version control

### Local Development

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/machinelearning-fraud-detection.git
   cd machinelearning-fraud-detection
   ```

2. **Install Dependencies**
   ```bash
   poetry install
   ```

3. **Install Pre-commit Hooks**
   ```bash
   poetry run pre-commit install
   ```

4. **Run Tests**
   ```bash
   poetry run pytest
   ```

### Environment Setup

- Use Poetry for dependency management
- Virtual environment is automatically created
- All dependencies are specified in `pyproject.toml`

## Code Style

### Python Code Style

- **Formatting**: Use Black with line length 100
- **Linting**: Use Ruff for code quality
- **Type Checking**: Use MyPy for type hints
- **Imports**: Sort imports with Ruff

### Code Quality Tools

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Type check
poetry run mypy .

# Run all quality checks
poetry run pre-commit run --all-files
```

### Code Standards

- **Type Hints**: Use type hints for all functions
- **Docstrings**: Follow Google docstring format
- **Error Handling**: Use appropriate exception types
- **Logging**: Use structured logging with appropriate levels

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_model.py

# Run with verbose output
poetry run pytest -v
```

### Test Requirements

- **Coverage**: Maintain >80% code coverage
- **Unit Tests**: Test all public functions
- **Integration Tests**: Test complete workflows
- **Edge Cases**: Test error conditions and edge cases

### Test Structure

- **Test Files**: One test file per module
- **Test Classes**: Group related tests in classes
- **Test Names**: Descriptive test method names
- **Fixtures**: Use pytest fixtures for common setup

## Pull Request Process

### Before Submitting

1. **Ensure Quality**
   - All tests pass
   - Code follows style guidelines
   - Pre-commit hooks pass
   - Documentation is updated

2. **Update Documentation**
   - Update README if needed
   - Update docstrings for new functions
   - Update configuration examples

3. **Check CI Status**
   - Ensure GitHub Actions pass
   - Address any CI failures

### Pull Request Guidelines

1. **Title**: Clear, descriptive title
2. **Description**: Detailed description of changes
3. **Related Issues**: Link to related issues
4. **Screenshots**: Include screenshots for UI changes
5. **Testing**: Describe how to test changes

### Review Process

- All PRs require review
- Address review comments promptly
- Maintainers will merge after approval
- CI must pass before merging

## Reporting Bugs

### Bug Report Template

```markdown
**Bug Description**
Brief description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., macOS, Windows, Linux]
- Python Version: [e.g., 3.12.0]
- Package Version: [e.g., 0.1.0]

**Additional Information**
Any other context, logs, or screenshots
```

## Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Brief description of the feature

**Use Case**
Why this feature would be useful

**Proposed Implementation**
How you think it could be implemented

**Alternatives Considered**
Other approaches you considered

**Additional Information**
Any other context or examples
```

## Getting Help

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Pull Requests**: For code contributions

### Documentation

- **README.md**: Project overview and quick start
- **docs/**: Detailed documentation
- **Code Comments**: Inline code documentation

## Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **CHANGELOG.md**: For significant contributions
- **GitHub Contributors**: Automatic recognition

## Questions?

If you have questions about contributing, please:

1. Check the documentation first
2. Search existing issues and discussions
3. Create a new issue with your question
4. Be patient and respectful

Thank you for contributing to Fraud Detection ML!
