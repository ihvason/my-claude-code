"""
Terminal UI layer: rendering primitives (render), slash commands (commands), and
the input area (input_ui, only present in some versions).

Import from the submodules directly (ui.render / ui.commands / ...). This
__init__ deliberately re-exports nothing: render is the lowest-level primitive
that even non-UI modules (such as permissions and agent.hooks) depend on, so
re-exporting commands here could form a circular import through them.
"""
