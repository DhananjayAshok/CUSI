"""CUSI: one package for the project's code.

    utils/      parameters, logging, model wrappers, embeddings, statistics, reply parsers
    envs/       the three benchmarks as text-action gym envs (GameBoy, AndroidWorld, WebVoyager),
                and each env's discrete action vocabulary
    agents/     the agent core shared by every pipeline: records, the model handle (AgentVLM), the
                native executors (GameBoyRL, M3A, WebVoyager), env specs, and the supervisors
    state/      encoders, novelty archive and curiosity scorers (inference), incl. the world model
    practice/   the practice pipeline (propose -> attempt -> guidance -> practice -> clean -> dataset)
    explore/    curiosity PPO and world-model / embedder training
    search/     pre-exploration tree search (library)
    eval/       test-set evaluation through the envs, and episode artifacts

Dependencies point one way: utils <- envs <- agents <- (practice, eval, explore, search);
utils <- state <- (explore, search).
"""
