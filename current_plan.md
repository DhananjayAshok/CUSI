# Current plan

## Running now

### Full-benchmark eval: 7 models × 3 envs

Each run is `slurm/eval.sh --env <env> --model <m> --run full_<short>_r0`. It's pushed with
`sync_data.py push --set evals` once all three of the model's envs complete; watchers do this automatically.

| model | GameBoy (99) | Web (429) | Android (116) | pushed |
|---|---|---|---|---|
| google/gemma-4-31b-it | 38.4% | 65.7% | 68.4% | yes |
| google/gemma-4-26b-a4b-it | 25.3% | 37.8% | 47.8% | yes |
| Qwen/Qwen3-VL-4B-Instruct | 4.0% | 23.5% | 34.8% | yes |
| Qwen/Qwen3-VL-2B-Instruct | running (rerun at 64k context) | 4.0% | 9.6% | — |
| Qwen/Qwen3-VL-8B-Instruct | queued | 34.7% | queued (vLLM start failed on a NAS stale-file error) | — |
| Qwen/Qwen3-VL-32B-Instruct | queued | 42.0% | 57.0% | — |
| meta-models/Muse-Glimmer-30B | queued | running | queued (rerun: the M3A set-action parser fix, 515d677) | — |

Notes:
- The non-Gemma GameBoy runs use 4 GPUs and `--max-model-len 65536`. At 32k, Qwen's native-resolution images
  overflowed the supervisor's multi-frame prompts.
- Every model has 1–2 Android tasks unscored by env faults, not the agent. OsmAndMarker fails for every model
  with `no such table: map_markers`.

### Exploration trees: GameBoy and Web (step 1 below)

| job | env | tree | run | resources |
|---|---|---|---|---|
| 285242 | gameboy | random | `tree_random_v1` | 1 Quadro |
| 285243 | gameboy | curiosity | `tree_curiosity_v1` | 1 Quadro |
| 285244 | gameboy | VLM (Gemma-4-31B) | `tree_vlm_v1` | 2 A6000, queued behind the evals |
| 285245 | web | random | `tree_random_v1` | 1 Quadro |
| 285246 | web | curiosity | `tree_curiosity_v1` | 1 Quadro |
| 285247 | web | VLM (Gemma-4-31B) | `tree_vlm_v1` | 2 A6000, queued behind the evals |

Submitted with `tmp_submit_trees.sh <env>...`. Android is submitted the same way once the evals finish.

- **GameBoy random:** done in 26 s. 40 expansions, 400 steps, 76 nodes, 245 archived screens.
  - Random and VLM trees give only ~400 frames each to the SigLIP and world-model training. GameBoy random could
    cheaply be extended with a larger `--max_expansions`; still undecided.
- **Web random:** 285245 crashed at iteration 12 and was resubmitted as 285248, which resumes.
  - Cause: `StaleElementReferenceException` in `WebVoyagerEnv._observe`. The page changed while
    `get_web_element_rect` was labelling it, and only the label cleanup catches that error.
  - **To do:** retry the labelling a few times on a stale element. Do it once no running job imports
    `cusi/envs/webvoyager.py` (Muse's Web eval, the Web trees).

### Pipeline test: passed (job 285241, 2026-10-20)

A small live GameBoy run on a Quadro GPU went search → SigLIP fine-tune → world model → wm_eval. Settings: 8 random
expansions, a 10-step fine-tune, a 5-epoch world model. Every step ran, and `predictions.png` rendered through the
fine-tuned SigLIP's decoder. The numbers mean nothing at this size: decoded frames are blurry, and world-model cosine
was 0.74 vs the copy baseline's 0.93.

## Plan: exploration trees → SigLIP observation embedder → world model

For each env (GameBoy, Web, Android):

1. **Three exploration trees**, each viewable via the `viewer.html` written into its run dir:
   - **VLM exploration:** `--expander vlm --model google/gemma-4-31b-it`
   - **Random exploration:** `--expander random`
   - **Curiosity exploration:** `--expander curiosity`, with novelty from the random projection;
     policy encoders `--policy_vision_encoder mobilenet_v3_small --policy_text_encoder sentence-transformers/all-MiniLM-L6-v2`.

   All three are scored the same way: `--image_embedder random_patch --text_embedder trigram --novelty_scorer combination`.
   Defaults are all scenes and 40 expansions × 10 steps for VLM and random. Curiosity is capped by time, about 6 h per env.
   Runs go through `slurm/search.sh`; they resume, and a larger `--max_expansions` extends them.
2. **SigLIP observation embedder:** fine-tune `google/siglip2-base-patch16-224` on the three trees pooled, by
   reconstruction (vision tower plus a decoder from the pooled vector to pixels). About 2,000 steps instead of the default 20.
   ```
   sbatch slurm/explore_train.sh --env <env> --command train_embedder \
       --extra "--embedder siglip --encoder_model google/siglip2-base-patch16-224 \
                --run_names search:<vlm>,search:<random>,search:<curiosity> --name trees --steps 2000 --max_minutes 120"
   ```
   Output: `storage_dir/explore/<env>/embedder/siglip_trees/`, holding the vision weights and the decoder.
3. **World model on that embedder:** the same pooled trees, re-embedded with the fine-tuned SigLIP, split by episode.
   ```
   sbatch slurm/explore_train.sh --env <env> --command world_model \
       --extra "--run_names search:<vlm>,search:<random>,search:<curiosity> --name trees \
                --image_embedder siglip --encoder_model google/siglip2-base-patch16-224 \
                --embedder_load_path <storage_dir>/explore/<env>/embedder/siglip_trees"
   sbatch slurm/explore_train.sh --env <env> --command wm_eval --extra "--name trees"
   ```
   wm_eval writes `predictions.png` and `eval.json` (cosine vs the "nothing changes" baseline) into the world-model dir.

Order: GameBoy and Web first. Android curiosity needs an exclusive A6000 node and is slow (~10 s/step), so it waits until
the evals free the nodes.

## Fixes made for this plan

1. **World model on search runs.** It read `action_space.json` from the first run dir, which nothing writes any more.
   It now builds the action vocabulary from the env name and saves it with the model.
2. **wm_eval on a SigLIP world model.** It fed the stored random-projection embeddings to the model and looked
   for `decoder/<name>`. The world model now records how to rebuild its embedder (`embedder_config` in its meta).
   wm_eval re-embeds with it and uses the fine-tuned SigLIP's own decoder.
3. **New Slurm pair** `slurm/explore_train.sh` and `scripts/slurm/explore_train.sh` for `train_embedder`, `decoder`,
   `world_model` and `wm_eval`.

## Known caveats (accepted for now)

- **Text masking.** The search encoder paints over text boxes before embedding. SigLIP fine-tuning and world-model
  re-embedding use the raw frames. That's consistent within this pipeline, but the fine-tuned SigLIP should not be dropped
  into a search with `mask_text` on unless it's retrained on masked frames.
- **Web/Android actions are element indices** (`click[12]`), so the world model's action input means different things on
  different pages. Expect a small margin over the copy baseline there.
- **Curiosity cost.** A curiosity session always commits its best episode, even a dull one. It adds one near-zero node
  and costs the whole RL session, and there's no early stop.
