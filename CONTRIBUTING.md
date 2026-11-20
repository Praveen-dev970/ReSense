# Contributing to Autism Detection System

First off, thank you for considering contributing to the Autism Detection System! It's people like you that make this project better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior**
- **Actual behavior**
- **Screenshots** (if applicable)
- **Environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case** - Why is this enhancement needed?
- **Proposed solution**
- **Alternative solutions** you've considered

### Code Contributions

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** for your feature
4. **Make your changes**
5. **Test** your changes
6. **Commit** with clear messages
7. **Push** to your fork
8. **Open a Pull Request**

## Development Setup

### Prerequisites

- Python 3.7 or higher
- Git
- Virtual environment tool

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/yourusername/autism-detection.git
cd autism-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Run tests (when available)
python manage.py test

# Start development server
python manage.py runserver
```

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide:

```python
# Good
def calculate_autism_score(responses):
    """Calculate autism screening score from questionnaire responses."""
    total_score = sum(responses.values())
    return total_score

# Bad
def calc(r):
    s=sum(r.values())
    return s
```

### Django Best Practices

- Use Django ORM instead of raw SQL queries
- Implement proper error handling
- Use Django's built-in authentication
- Follow Django's MVT (Model-View-Template) pattern

### Code Documentation

- Add docstrings to all functions and classes
- Comment complex logic
- Keep comments up-to-date

```python
def process_video_frame(frame, model):
    """
    Process a single video frame for autism detection.
    
    Args:
        frame (np.ndarray): Video frame as numpy array
        model (keras.Model): Trained CNN model
        
    Returns:
        tuple: (prediction, confidence_score)
    """
    # Implementation here
    pass
```

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting)
- **refactor**: Code refactoring
- **test**: Adding tests
- **chore**: Maintenance tasks

### Examples

```bash
feat(video): add frame preprocessing function

Add new function to preprocess video frames before CNN analysis.
Includes normalization and resizing.

Closes #123
```

```bash
fix(auth): resolve login redirect issue

Fixed bug where users were not redirected to dashboard after login.
Updated redirect logic in views.py.

Fixes #456
```

## Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add/update tests** for new features
3. **Ensure all tests pass**
4. **Follow coding standards**
5. **Update CHANGELOG.md** (if exists)

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

### Review Process

1. At least one maintainer must approve the PR
2. All tests must pass
3. No merge conflicts
4. Address all review comments

## Areas for Contribution

### High Priority

- [ ] Improve CNN model accuracy
- [ ] Add unit tests
- [ ] Enhance UI/UX
- [ ] Mobile responsiveness
- [ ] API documentation

### Medium Priority

- [ ] Performance optimization
- [ ] Code refactoring
- [ ] Additional data validation
- [ ] Logging improvements

### Good First Issues

- [ ] Update documentation
- [ ] Fix typos
- [ ] Add code comments
- [ ] UI improvements
- [ ] Add example datasets

## Questions?

Feel free to:
- Open an issue for discussion
- Contact the maintainers
- Join our community chat (if available)

---

Thank you for contributing! 🎉
