#!/usr/bin/env bash
# GameBoy parity check for the curiosity expander (plans/next_plan.md §E.7, step 5): one root, local reward buffer,
# image-only (embedding) novelty, <steps> env steps, against GameBoyRL's cleanrl ppo_curiosity (embedbuffer, curiosity
# reset every episode) from the same init state. Both print their episode returns; the summary compares the last 100.
# Runs on the node; the curiosity policy uses the first visible GPU if any. Run from the CUSI root
# (Slurm wrapper: slurm/curiosity_parity.sh).
#
#   bash scripts/slurm/curiosity_parity.sh --policy_text_encoder <policy_text_encoder> [--init_state <value>] \
#       [--scene <value>] [--steps <value>] [--k_steps <value>] [--num_envs <value>] [--run <value>] [--seed <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict CURIOSITY_PARITY_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array CURIOSITY_PARITY_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
JOB="${SLURM_JOB_ID:-local$$}"
LOG_DIR="$results_dir/logs/curiosity_parity_$JOB"
mkdir -p "$LOG_DIR"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  logs: $LOG_DIR"

ENV_ID="gameboy_worlds-pokemon_red-default-${ARGS["init_state"]}-low_level-${ARGS["k_steps"]}-False"
( cd GameBoyRL/cleanrl && python -u cleanrl/ppo_curiosity.py --exp_name parity --seed "${ARGS["seed"]}" \
    --env-id "$ENV_ID" --total-timesteps "${ARGS["steps"]}" --no-track --observation_embedder random_patch \
    --similarity_metric cosine --curiosity-module embedbuffer --ocr_alpha 0.0 --reset-curiosity-module \
    > "$LOG_DIR/gameboyrl.out" 2>&1 ) &
GBRL=$!

python -u run_search.py --env gameboy --run_name "${ARGS["run"]}" --scenes "${ARGS["scene"]}" --max_expansions 1 \
    --roots_only --k_steps "${ARGS["k_steps"]}" --image_embedder random_patch --text_embedder none \
    --novelty_scorer embedding --expander curiosity --reward_buffer local --rl_total_steps "${ARGS["steps"]}" \
    --rl_max_minutes 240 --rl_num_envs "${ARGS["num_envs"]}" --policy_vision_encoder mobilenet_v3_small \
    --policy_text_encoder "${ARGS["policy_text_encoder"]}" --seed "${ARGS["seed"]}" --overwrite \
    > "$LOG_DIR/cusi.out" 2>&1
CUSI_STATUS=$?
wait $GBRL
GBRL_STATUS=$?
RUN_DIR=$(python -m cusi.utils.paths search_run --env gameboy --run "${ARGS["run"]}")
python - "$LOG_DIR/gameboyrl.out" "$RUN_DIR" <<'PY'
import json, re, sys, numpy as np, glob
g = [float(m) for m in re.findall(r"episodic_return=(?:tensor\(\[?)?([-0-9.e]+)", open(sys.argv[1]).read())]
rows = [json.loads(l) for l in open(f"{sys.argv[2]}/expansions.jsonl")]
info = rows[0]["expander_info"]
c = np.load(f"{sys.argv[2]}/{info['returns']}")
def summ(x):
    x = np.asarray(x)
    return {"episodes": len(x), "first100": round(float(x[:100].mean()), 4) if len(x) else None,
            "last100": round(float(x[-100:].mean()), 4) if len(x) else None, "max": round(float(x.max()), 4) if len(x) else None}
print("GameBoyRL ppo_curiosity:", summ(g))
print("CUSI curiosity expander:", summ(c), "seconds", info["seconds"], "best episode", info["best_episode"])
PY
echo "status: cusi $CUSI_STATUS gameboyrl $GBRL_STATUS"
