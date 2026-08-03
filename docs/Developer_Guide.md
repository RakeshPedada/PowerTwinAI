# PowerTwinAI Developer Guide

**Version:** 1.0

**Status:** Active

**Last Updated:** August 2026

**Owner:** Pedada Rakesh

---

## Related Documents

- [Project README](../README.md)
- [Architecture](Architecture.md)
- [Vision Document](Vision_Document.md)
- [Current Development Status](Current_Status.md)
- [Development Roadmap](Roadmap.md)

---

## Purpose

This document provides development guidelines for contributors working on PowerTwinAI. It explains the recommended development workflow, repository organization, coding practices, documentation standards, and version control conventions followed throughout the project.

The objective of this guide is to ensure consistency, maintainability, and collaboration as PowerTwinAI evolves from a research prototype into a scalable software platform.

---

## 1. Project Overview

PowerTwinAI is an AI-assisted electrical infrastructure inspection software currently focused on image-based 2D-to-3D reconstruction, electrical component detection, defect detection, and automated inspection reporting.

The repository is organized to support long-term growth into a complete Digital Twin platform while maintaining a modular and maintainable software architecture.

Developers should understand the overall project architecture before modifying individual modules. The Architecture document should therefore be read before contributing to the implementation.

---

## 2. Repository Structure

The repository is organized according to software engineering principles that separate documentation, implementation, research material, datasets, and presentation resources.

```
PowerTwinAI/

├── docs/
├── reconstruction/
├── modules/
├── assets/
├── datasets/
├── results/
├── research/
└── presentation/
```

Each top-level directory has a clearly defined responsibility. Contributors should place new files only within their appropriate directory and avoid introducing unnecessary folders at the repository root.

---

## 3. Development Environment

The current development environment is based on Python.

Recommended tools include:

- Python
- Visual Studio Code
- Git
- GitHub
- Streamlit
- OpenCV
- Open3D
- COLMAP

Future development environments may expand as additional AI and cloud modules are introduced.

---

## 4. Installing Dependencies

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd PowerTwinAI
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Additional software such as COLMAP should be installed separately according to its official installation instructions.

---

## 5. Running the Project

PowerTwinAI currently provides a Streamlit-based user interface for running the reconstruction pipeline.

To launch the application:

```bash
streamlit run app.py
```

Developers should ensure that all required dependencies and external software (such as COLMAP) are installed before executing the application.

Future versions may provide additional execution modes including command-line interfaces, APIs, and cloud deployment.

---

## 6. Coding Guidelines

To maintain a consistent and maintainable codebase, contributors should follow these development principles:

- Write modular and reusable code.
- Keep functions focused on a single responsibility.
- Use meaningful variable and function names.
- Add comments only where they improve understanding.
- Avoid duplicate implementations.
- Maintain consistency with the existing project structure.
- Document significant architectural changes.

Code readability should always be prioritized over unnecessary complexity.

---

## 7. Git Workflow

PowerTwinAI follows a documentation-first and architecture-driven development workflow.

The recommended workflow is:

1. Review the architecture and documentation.
2. Implement the required feature.
3. Test the implementation.
4. Update the relevant documentation.
5. Commit using descriptive commit messages.
6. Push changes to the repository.

Every implementation should remain consistent with the documented system architecture.

---

## 8. Commit Message Convention

PowerTwinAI follows a structured commit message format.

Examples include:

- docs: update project documentation
- feat: add component detection module
- fix: resolve reconstruction issue
- refactor: improve reconstruction pipeline
- build: update project configuration
- chore: repository maintenance

Commit messages should describe the purpose of the change rather than the implementation details.

---

## 9. Branch Strategy

At the current stage of development, the project is maintained primarily through the main branch.

As the project grows and additional contributors join, feature branches should be introduced for independent development before merging into the main branch after review.

---

## 10. Adding New Modules

New functionality should be implemented as independent modules whenever possible.

Before introducing a new module, contributors should:

- Verify that similar functionality does not already exist.
- Ensure compatibility with the existing architecture.
- Update the Architecture document if the software structure changes.
- Update the repository documentation when new modules are introduced.

All new modules should follow the modular design principles established by PowerTwinAI.

---

## 11. Documentation Standards

Documentation is considered an integral part of the software project.

Whenever the repository structure, architecture, or functionality changes, the corresponding documentation should also be updated.

The primary documentation sequence is:

1. README
2. Vision Document
3. Architecture
4. Developer Guide
5. Current Status
6. Roadmap

Maintaining consistency between these documents is essential for long-term project maintainability.

---

## 12. Future Contributors

PowerTwinAI is intended to evolve into a collaborative engineering project.

Future contributors are encouraged to:

- Understand the project vision before implementing new features.
- Review the Architecture document before modifying the software structure.
- Follow the established coding and documentation standards.
- Preserve the modular architecture of the project.
- Contribute improvements through well-documented and maintainable code.

The long-term objective is to build a professional AI-powered Digital Twin platform through collaborative software engineering practices.

---

## 13. Related Documents

For additional information, refer to:

- [Project README](../README.md)
- [Vision Document](Vision_Document.md)
- [Architecture](Architecture.md)
- [Current Development Status](Current_Status.md)
- [Development Roadmap](Roadmap.md)

These documents together provide a complete overview of the project, its architecture, current implementation, and future direction.
