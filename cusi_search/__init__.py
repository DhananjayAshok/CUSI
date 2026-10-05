"""cusi_search: the pre-exploration tree search (skill_discovery.md §1.4).

    tree.py       Node / Expansion / Tree: every observation and action of every expansion is kept
    selection.py  energy sampling over all nodes, E(n) = Q(n) + c * P(n) * sqrt(N_total) / (1 + N(n))
    expanders.py  RandomExpander, VLMExplorer (goal-free)
    prior.py      VLMPrior: P(n) in [0, 1], one call per node
    search.py     TreeSearch: select -> restore -> expand K steps -> add children -> update statistics

The statistics (Q, N, N_total, yield and its propagation), the stopping rule, the env per scene
and logging are passed in as objects, so this package never imports cusi_search.toy.
"""
