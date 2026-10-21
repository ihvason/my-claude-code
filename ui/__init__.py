"""
Terminal UI layer: slash commands (commands) and the input area (input_ui).

Import from the submodules directly (ui.commands / ui.input_ui). This __init__
deliberately re-exports nothing: the shared rendering primitives now live in
core.console, and re-exporting commands here would pull the whole agent stack
into any importer.
"""
