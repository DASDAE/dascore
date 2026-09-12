---
name: dascore
description: Use the installed DASCore APIs and workflow recipes when analyzing distributed fiber-optic sensing data or writing Python code that uses DASCore.
---

# DASCore

Load guidance from the DASCore installation used by the current project.

1. Identify the project's Python environment from its instructions and environment configuration. Prefer that environment's executable over a global `python` or `dascore` command. If multiple environments remain plausible, ask which one to use.
2. Run that executable with `-m dascore doc`. Check the reported Python executable, package location, and version against the project. Retain the absolute virtual-environment executable path for this task; resolving its symlink can bypass the environment.
3. Prefix every subsequent Python command with the recorded absolute executable path. Invoke the CLI as `<absolute-python> -m dascore ...`, including `skill dascore` to load the routing recipe and `skills` to discover specialized workflows. Read relevant APIs with `doc <name>` and find unfamiliar APIs or procedures with `doc-search <query>`.

Resolve the environment again for each project. Do not switch DASCore versions merely to obtain guidance. If the selected installation lacks CLI or skill support, use its installed docstrings and explain the limitation; follow the project's dependency conventions when installing optional CLI support.

The loaded recipes supply the version-specific procedures. Follow the user's task scope and existing authorization when applying them.
