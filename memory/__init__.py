"""
Long-term memory subpackage: prompt instructions (instructions), file storage
(store), on-demand recall (recall), and background extraction plus dream
consolidation (background).

Import from the submodules directly (memory.store / memory.recall / ...). This
__init__ deliberately re-exports nothing: store is the lowest-level module that
even permissions and agent.core depend on, so re-exporting background here could
form a circular import through them.
"""
