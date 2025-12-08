<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Contributing to Sigil MCP Server

Thank you for considering contributing to Sigil MCP Server! This document provides guidelines for contributing to the project.

**Sigil MCP Server is open-source under AGPLv3.**

Individuals, hobbyists, and small teams are welcome to use and contribute under the open-source license. Larger organizations that can't or don't want to comply with AGPLv3 can contact me for commercial licensing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [License Agreement](#license-agreement)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## License Agreement

### Contributor License Agreement (CLA)

All contributors must agree to our [Contributor License Agreement (CLA)](CLA.md) before their contributions can be accepted. The CLA clarifies the intellectual property rights granted with your contributions.

**The CLA lets me keep licensing rights clean so I can offer both AGPLv3 for the community and commercial licenses for organizations that need them.**

**If you're just trying to fix a bug or improve the project, the CLA doesn't change that—your contribution is still open-source under AGPLv3. It just makes the legal side unambiguous for everyone.**

**Key points:**
- By contributing to this project, you agree that your contributions will be licensed under the GNU Affero General Public License v3.0 (AGPLv3)
- Your contributions become part of the open-source codebase
- If you modify and run Sigil MCP Server for users over a network, AGPLv3 requires that those users be able to get the corresponding source code for your modified version
- By signing the CLA, you allow the maintainer to offer commercial licenses alongside the AGPLv3 open-source license

Please read the full [CLA](CLA.md) for complete terms.

### Developer Certificate of Origin (DCO)

**The DCO works together with the CLA:** the CLA covers how your contribution may be licensed, and the DCO certifies that you're legally allowed to contribute it.

All contributors must sign off on their commits, certifying that they have the right to submit the code under the project's license. This is done by adding a `Signed-off-by` line to commit messages:

```
Signed-off-by: Your Name <your.email@example.com>
```

To automatically add this to your commits, use:

```bash
git commit -s -m "Your commit message"
```

By signing off, you certify that:

1. The contribution was created in whole or in part by you and you have the right to submit it under the AGPLv3 license
2. The contribution is based upon previous work that, to the best of your knowledge, is covered under an appropriate open-source license
3. You understand and agree that your contribution is public and that a record of it will be maintained indefinitely

### For Companies and Internal Use

If you're an organization that:

- wants to run Sigil MCP Server internally without AGPLv3 obligations, or
- needs to keep modifications private, or
- has a "no AGPL" policy,

please contact: davetmire85@gmail.com to discuss commercial licensing.

Commercial licensing provides:
- Freedom to use internally without open-source requirements
- Ability to keep your modifications proprietary
- Indemnification and support options
- Clear legal status for enterprise compliance

For common licensing questions, see the [Licensing FAQ in README.md](README.md#licensing-faq).

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- Virtual environment tool (`venv`)

### Setting Up Development Environment

1. **Fork and clone the repository:**

```bash
git clone https://github.com/Superuser666-Sigil/SigilDERG-Custom-MCP.git
cd SigilDERG-Custom-MCP
```

2. **Create a virtual environment:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install development dependencies:**

```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,watch,llamacpp]"
```

4. **Verify installation:**

```bash
pytest tests/
```

### Optional: Install universal-ctags

For enhanced symbol extraction:

```bash
# Ubuntu/Debian
sudo apt install universal-ctags

# macOS
brew install universal-ctags
```

## Development Process

### Branching Strategy

- `main` - Stable release branch (all development happens here)
- `feature/*` - Feature branches
- `bugfix/*` - Bug fix branches

### Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests

3. Run the test suite:
   ```bash
   pytest tests/
   pytest tests/ --cov=sigil_mcp  # With coverage
   ```

4. Format and lint your code:
   ```bash
   black sigil_mcp/ tests/
   ruff check sigil_mcp/ tests/
   mypy sigil_mcp/
   ```

5. Commit with sign-off:
   ```bash
   git commit -s -m "feat: add new feature"
   ```

6. Push and create a pull request

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://github.com/psf/black) for code formatting (line length: 100)
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Use [mypy](http://mypy-lang.org/) for type checking

### Code Quality

- Write clear, self-documenting code
- Add type hints to all function signatures
- Include docstrings for modules, classes, and functions (Google style)
- Keep functions focused and concise
- Avoid code duplication

### Testing

- Write unit tests for all new functionality
- Maintain or improve code coverage (target: >80%)
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Include integration tests for major features
- Test edge cases and error conditions

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]

Signed-off-by: Your Name <your.email@example.com>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

**Examples:**
```
feat(indexer): add support for trigram search

fix(auth): correct OAuth token refresh logic

docs(readme): update installation instructions

Signed-off-by: Jane Developer <jane@example.com>
```

### Copyright Headers

All source files must include the AGPLv3 copyright header:

```python
# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com
```

## Submitting Changes

### Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass
   - Update documentation if needed
   - Rebase on latest `main` if needed

2. **PR Description should include:**
   - Summary of changes
   - Related issue numbers (if any)
   - Testing performed
   - Breaking changes (if any)
   - Screenshots (for UI changes)

3. **PR Template:**

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123
Relates to #456

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
- [ ] Commits are signed off (DCO)
```

4. **Review Process:**
   - PRs require at least one approval
   - Address all review comments
   - Keep PRs focused and reasonably sized
   - Be responsive to feedback

## Reporting Issues

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages
- Minimal reproducible example

**Template:**

```markdown
**Description:**
Brief description of the bug

**Steps to Reproduce:**
1. Step one
2. Step two
3. ...

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.12.1
- Sigil MCP Version: 0.1.0

**Logs:**
```
[paste relevant logs]
```
```

### Feature Requests

Include:
- Clear description of the proposed feature
- Use cases and benefits
- Potential implementation approach
- Any alternatives considered

### Security Vulnerabilities

**DO NOT** report security vulnerabilities through public issues. Instead, email: davetmire85@gmail.com with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

See [SECURITY.md](docs/SECURITY.md) for more details.

## Community

### Communication Channels

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** General questions and community support
- **Email:** davetmire85@gmail.com for commercial licensing and security issues

### Recognition

Contributors are recognized in:
- GitHub's contributor graph
- Project documentation where appropriate
- Release notes for significant contributions

### Getting Help

If you need help:
1. Check existing documentation (README, docs/)
2. Search closed issues and discussions
3. Open a new discussion for questions
4. Join community forums (if available)

## Additional Resources

- [README.md](README.md) - Project overview
- [RUNBOOK.md](docs/RUNBOOK.md) - Operations guide
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Common issues
- [Architecture Decision Records](docs/) - Design decisions

---

Thank you for contributing to Sigil MCP Server! Your efforts help make this project better for everyone.
