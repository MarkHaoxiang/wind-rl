"""Architecture-benchmark proxies: rank ``ModelConfig`` variants on fast tasks.

Two proxy tasks from ``docs/research/2026-07-19-geometric-architectures.md`` S5:
the supervised critic value-regression proxy (:mod:`.critic`, backed by the
random-policy return dataset in :mod:`.dataset`) and the fixed-budget MAPPO
policy proxy (:mod:`.policy`, a thin adapter onto the shared sweep loop).
"""
