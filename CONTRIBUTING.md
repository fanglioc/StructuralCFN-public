# Contributing to StructuralCFN

Thank you for your interest in contributing to StructuralCFN! We welcome contributions from the community, whether they are bug fixes, feature enhancements, or documentation improvements.

## How to Contribute

1. **Fork the Repository**: Create your own fork of the project on GitHub.
2. **Clone the Fork**: Clone your fork locally.
3. **Create a Branch**: Create a new branch for your changes.
4. **Make Changes**: Implement your changes and ensure they follow the project's coding style.
5. **Add Tests**: If you're adding a new feature, please include a corresponding example or test in `examples/` or `benchmarks/`.
6. **Submit a Pull Request**: Push your branch to GitHub and open a pull request against the main repository.

## Coding Style
- Follow PEP 8 for Python code.
- Ensure all new features are documented with clear docstrings.
- If you add a new `FunctionNode`, make sure to implement the `.formula()` method for symbolic interpretability.

## Research Contributions
If your contribution introduces a new structural prior or aggregation logic, please include empirical results on at least one of the core benchmarks (e.g., Diabetes or CA Housing) to demonstrate performance and interpretability.

By contributing to this project, you agree that your contributions will be licensed under the MIT License.
