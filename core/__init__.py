"""
Cross-cutting primitives shared by every layer.

- paths: project root discovery and every on-disk location the app uses
- console: rich-based output helpers used by both the UI and non-UI modules

Import from the submodules directly (core.paths / core.console). This __init__
deliberately re-exports nothing so the leaf modules stay importable without
pulling in the rest of the application.
"""
