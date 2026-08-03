"""Figure-generation subpackage — the only matplotlib importer in the repo.

One module per plot kind: ``curves`` (line plots), ``bars`` (bar charts),
``maps`` (attention/alignment matrices), ``heatmap`` (side-by-side attention
and dependency matrices — the ``ha`` comparison variant), ``scoreboard``
(the battery table/heatmap), and ``style`` (shared rc params, palette, and the
machine-relative timing helper).  Every function is pure:
it takes already-computed data and returns or saves a ``Figure`` — it never
trains, runs a mechanism, or touches the network.

Provided scaffold — students never modify this subpackage.
"""
