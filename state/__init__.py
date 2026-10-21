"""
Persistent session state: transcripts and metadata (sessions), task store (tasks),
rewind checkpoints (file_history), and background job registry (jobs).

Import from the submodules directly (state.sessions / state.tasks / ...). This
__init__ deliberately re-exports nothing: the stores depend only on core.paths,
and re-exporting them here would pull every store into any importer.
"""
