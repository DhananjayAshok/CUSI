# CUSI plan: shared vLLM, a cross-env practice pipeline, curiosity exploration + world model

Written 2026-10-04. Builds on the common env contract in `cusi_envs/`
(`TextActionEnv`: obs = `{frame, texts, actions, goal}`, text actions, `test` / `free_play`
modes) implemented by `AndroidPlayEnv`, `GameBoyPlayEnv` and `WebVoyagerPlayEnv`.

## Decisions so far

| # | Question | Decision |
|---|---|---|
| 1 | Model | `google/gemma-4-26b-a4b-it` (Gemma 4 26B MoE). Not downloaded yet. |
| 2 | vLLM | A separate, shared vLLM environment **outside CUSI** (used by every project on this system), with the newest vLLM / torch / transformers that work together, plus a script that starts a server and returns. |
| 3 | GPUs | Whatever is available (plan: 2x RTX A6000, below). |
| 4 | Models per role | One model for every role (executor, checker/judge, proposer, guidance, cleaning). |
| 5 | Scenes | Two per environment. No artificial step caps: each stage keeps its normal step budget. |
| 6 | Scale | Small: 2 scenes x 3 proposed tasks per environment, then guidance + practice on the successes. |
| 7 | Element lists in prompts | Whatever each native agent does (see Part 1): M3A and WebVoyager include their element lists; GameBoy has none yet (OCR TODO). |
| 8 | Image downscaling for the VLM | No. |
| 9 | Locations | Code in new CUSI packages; outputs under `storage_dir`. |
| - | Executor design | **Option B: each environment's native agent** (GameBoyRL executor, M3A, WebVoyager), sharing everything else. |
| 10 | Exploration policy | A **VLM text policy** trained with PPO (not a CNN). |
| 11 | World model | **Same architecture as GameBoyRL**. |
| 12 | Observation embedder | A **frozen pretrained vision encoder** for all three environments (GameBoy included). |
| 13 | Text novelty | Yes: novelty of `obs["texts"]` (element lists) as the analogue of GameBoyRL's OCR novelty. |

**TODO (GameBoy OCR):** GameBoy will get an OCR text channel (`obs["texts"]["ocr"]`) by a
method still to be decided (TODO comment in `cusi_envs/gameboy.py`). Until then GameBoy has no
text channel: its executor prompts have no element/text block, and its curiosity uses frame
novelty only. Everything below plugs the OCR text in where marked **[GB-OCR]** once it exists.

---

## Part 0: Shared vLLM infrastructure

**Location (outside CUSI, shared across projects):** `/nas/eclairnas01/users/ashokd/vllm/`

- `.venv/`: a uv project with the newest **vLLM** release, the **torch** it is built against
  (vLLM pins torch; "newest possible" means vLLM's pin unless built from source), and the newest
  **transformers** compatible with both (Gemma 4 needs a recent transformers). Python 3.12.
- `hf_cache/`: `HF_HOME` for model weights, shared by all projects. Gemma 4 26B MoE is about
  52 GB in bf16.
- `serve_vllm.sh` / `stop_vllm.sh`: modelled on `GameBoyRL/scripts/core/serve_vllm.sh`:
  - activates the vLLM venv **inside a subshell**;
  - `setsid nohup vllm serve <model> ...` in its own process group;
  - defaults `-tp` to the number of visible GPUs;
  - waits for `/health` (with a long timeout for first-load) and fails fast if the process dies;
  - writes the log and PID under `/nas/eclairnas01/users/ashokd/vllm/state/<host>_<port>.{log,pid}`;
  - **returns to the caller**, whose environment is untouched (the venv was only active in the
    subshell). `stop_vllm.sh` kills the whole process group.
- `setup.sh`: idempotent installer (uv venv, `uv pip install vllm transformers`, a smoke test
  that imports them and reports versions).

**In CUSI:** `scripts/serve_vllm.sh` calls the shared `serve_vllm.sh` with CUSI's model and port
and leaves CUSI's venv active; CUSI talks to it through `vLLM_base_url` (cusi_utils `vLLMModel`).

**Hardware:** `medium-lg` A6000 nodes (`eclairlg[02-07]`, 4x RTX A6000 48 GB each). Plan:
`--gres=gpu:rtxa6000:2`, `-tp 2`. The Quadro RTX 8000 nodes are Turing (no bf16), so they are a
fallback only (fp16). The same Slurm job runs vLLM, the environments (Android emulators need
KVM; to confirm on the A6000 nodes) and the pipeline.

**Verification:** server healthy; a text and an image request through `cusi_utils`
(`build_model(model_backend="vllm")`); throughput noted.

---

## Part 1: Practice pipeline for all three environments (Option B)

### What GameBoyRL does (the reference)

`propose_tasks_zeroshot` (VLM proposes tasks from a start state's first frame + action menu)
→ `attempt_tasks` (the executor tries each task; a checker describes the frames and judges
success from the last 8; on failure a critique hint is derived and it retries, up to 5 tries)
→ `infer_guidance` (8-frame slices of each successful trajectory become a summary, a goal
condition and steps, then consolidated) → `practice_tasks` (each successful task re-run N times
from a start perturbed by random actions, with the guidance as the executor's hint and the goal
condition for the judge only; one retry with a critique hint; **every executor VLM call (prompt,
images, response) is pickled**) → `clean_practice` (paraphrase tasks; ACCEPT/REJECT each call)
→ `create_dataset` (strip hint blocks from inputs, cut at the judge's safe success point, drop
unparseable or illegal actions, train/val split with paraphrase augmentation).

The second task source, `infer_tasks` from curiosity trajectories, comes in Part 2.

### What changes for three environments

**Package:** `cusi_practice/` (GameBoyRL untouched). **Outputs:** `storage_dir/practice/<env>/...`

1. **Env contract additions (`cusi_envs`):**
   - `sample_action() -> str`: a random valid action for the start-state perturbation
     (GameBoy: a button; Android: tap/scroll/back/home; Web: click/scroll).
   - `env_description`: one sentence used in the shared prompts in place of "a GameBoy game".
   - `AndroidPlayEnv`: also return the raw screenshot (`info["raw_frame"]`), since M3A shows both
     the raw and the set-of-mark screenshot.
   - Environments run in **free_play** mode: the proposed task lives in the executor's prompt.
2. **Native executors behind one interface.** A shared skeleton (step loop, error feedback,
   4-consecutive-invalid limit, step budget, per-call logging, a hint/guidance slot), with prompt
   building taken from each benchmark's own code (imported, not rewritten):

   | | GameBoy | AndroidWorld (M3A) | WebVoyager |
   |---|---|---|---|
   | Prompt code | GameBoyRL `STEP_PROMPT`, `single` action + `actions` history (as GameBoyRL practice) | M3A `_action_selection_prompt` + `_summarize_prompt` | `SYSTEM_PROMPT`, `format_msg`, `clip_message_and_obs` |
   | Calls per step | 1 | 2 (action: raw + SoM screenshots + element list; summary: before/after) | 1, over the whole chat |
   | Text in prompt | none yet **[GB-OCR]** | the numbered UI-element list | the web element list |
   | Memory | last k actions | its own step summaries | chat history, last 3 screenshots (`run.sh` default) |
   | Guidance slot | the existing hint block | M3A's `additional_guidelines` | a marked block in the first user message |
   | Leg ends on | done-check (attempts only) / budget | the `status` action / budget | the `ANSWER` action / budget |

   The judge decides success everywhere; an agent's own "done" only ends the leg. Hint and
   guidance blocks are wrapped in markers so `create_dataset` can strip them, making training
   inputs identical to what the evaluation agent sees.
3. **Shared stages, ported once:** propose, attempt (+ critique-hint retries), guidance,
   practice (+ one retry), clean (paraphrase + accept/reject), dataset. The checker, critique and
   guidance work from frames (plus WebVoyager's `ANSWER` text, which the judge also sees).
4. **Records:** GameBoyRL's call-record fields (tag, images, prompt or chat, response, tokens),
   with generic step records (frame before/after, `parsed_action`, `valid`, `error`, `reward`),
   so pickles load without any environment's classes. Images are passed as **PIL RGB** (GameBoyRL's
   numpy-to-image converter keeps channel 0 only, which would grey out colour screenshots).
5. **Dataset format:** one row per model call = **a chat (role/content messages with image
   paths) + the target response**. GameBoy rows are a single user turn; Android rows carry the
   call type (action / summary, both kept: M3A writes its own summaries at test time); Web rows
   are the chat up to that call. The legality filter becomes "the environment accepted the
   action" (stored `valid`), replacing GameBoyRL's hard-coded `VALID_ACTIONS`.
6. **Concurrency:** thread pool as in GameBoyRL. GameBoy: several in-process instances; Web: a
   few browsers; Android: one emulator per worker (private AVD copy + distinct ports, which
   `scripts/android_emulator.sh` already supports), 1-2 at this scale.
7. **Scenes (2 each):**
   - GameBoy: two Pokémon Red training `init_state`s.
   - Android: two AndroidWorld task scenes in free_play (one Contacts, one Settings task).
   - Web: arXiv and GitHub start pages (WebVoyager tasks `ArXiv--*`, `GitHub--*`).

### Small-scale test

One Slurm job on an A6000 node: start vLLM, then for each environment run all six stages on
2 scenes x 3 proposed tasks with the normal step budgets, then guidance + practice on whatever
succeeded. Report counts per stage (proposed / attempted / succeeded / practice runs /
accepted calls / dataset rows) and a few example rows per environment. An environment with no
successful attempts has nothing to practise; that is reported as such rather than forced.

---

## Part 2: Curiosity exploration (VLM policy + PPO) and world model, for all three environments

### What GameBoyRL does (the reference)

`ppo_curiosity.py`: cleanrl PPO on one environment (2 stacked 144x160 grayscale frames, small
CNN actor-critic, discrete buttons, 30-step episodes); reward = environment reward + raw
intrinsic novelty. `EmbedBuffer`: embed the newest frame (fixed random patch projection or a
trained patch autoencoder), novelty = 1 - max cosine similarity to seen embeddings, with dedup
and KMeans compaction, optionally preloaded as an "already seen" prior; `OCRBuffer` gives
novelty of text regions; `CombinationBuffer` mixes them (`ocr_alpha`). Every transition goes to
a replay buffer; z-score outliers give high-novelty trajectories, which are grouped and feed
`infer_tasks`. World model: embeddings of the two stacked frames + `nn.Embedding(action index)`
→ MLP → next-frame embedding (MSE), trained on the replay buffers; the world-model executor
decodes predicted embeddings to pixels and lets a VLM pick an action from the predicted frames.

### What we build

**Package:** `cusi_explore/` (cleanrl and GameBoyRL untouched). **Outputs:** `storage_dir/explore/<env>/...`

1. **Observation encoder (frozen, pretrained, all environments).** A pretrained vision encoder
   (to choose: e.g. a SigLIP 2 / CLIP-class image encoder) embeds every frame. Used for
   curiosity novelty, trajectory grouping and the world model's embedding space. Same encoder for
   GameBoy, Android and Web.
2. **Policy: a VLM text policy trained with PPO** (RL4VLM-style):
   - The policy **reuses the native executors' prompts** from Part 1, so what it learns is in
     each benchmark's evaluation format.
   - Action = the generated text; its log-probability = the sum over the action tokens; a value
     head on the final hidden state; LoRA on the policy VLM; a KL penalty to the reference model
     to keep its language intact; sampling at temperature > 0.
   - PPO as in `ppo_curiosity.py`: GAE (gamma 0.99, lambda 0.95), clipped objective on the
     sequence log-prob ratio, rollouts of `num_steps` per update, minibatches re-encoding the
     stored images. Fix the stale-reward-at-episode-end bug in the original loop.
   - Runs in-process (transformers + PEFT) on the GPU node; vLLM is not used for the trained
     policy.
3. **Curiosity:**
   - `EmbedBuffer` ported as is, on frozen-encoder embeddings (cosine novelty, dedup, KMeans
     compaction, save/load prior).
   - **Text novelty buffer** (replaces `OCRBuffer`): novelty of the element lines in
     `obs["texts"]` (Android UI elements, web elements; GameBoy **[GB-OCR]**).
   - `CombinationBuffer` mixing them with an alpha, as now.
   - Reward = intrinsic only in free_play (the environments give 0), as in the curiosity runs.
4. **Episodes:** free_play, truncated at a per-environment `max_steps` (GameBoy 30 as now), then
   `reset()` (GameBoy instant; Android snapshot reset ~10-30 s; Web fresh browser ~8 s).
5. **Replay buffer:** per step: frame (compressed on disk; Android frames are 7.7 MB raw),
   frozen-encoder embedding, texts, the generated text, the canonical `parsed_action`, the
   world-model action index (below), rewards, episode boundaries. Chunked files.
6. **Trajectory extraction + grouping:** port `save_outliers` (z-scored rewards, high-novelty
   backtraces) and `group_trajectories` (grouping by final-frame similarity, now in encoder space),
   then the curiosity task source for Part 1 (`infer_tasks` + `infer_guidance` → practice).
7. **World model: GameBoyRL's architecture.** `[emb(frame_{t-1}), emb(frame_t)]` → Linear(512),
   action index → `nn.Embedding(n_actions, 512)` → Linear(1024→512) → ReLU → LayerNorm →
   Linear(→emb_dim), L2-normalised; MSE to `emb(frame_{t+1})`; trained on the replay buffers;
   `action_space.json` pins the index meaning. Action indices come from a per-environment
   vocabulary over canonical actions: GameBoy buttons; Android/Web: action type x element index
   (< K), scroll directions, back/home/GoBack, with typing as one index per action type. Note: on
   Android/Web an element index means a different element on each screen, so the action
   embedding carries less information than on GameBoy; kept as decided.
8. **Using the world model:** the frozen encoder has no decoder, so the world-model executor's
   "show predicted frames" needs either (a) a decoder trained from encoder embeddings to pixels,
   or (b) retrieving the nearest real frame in the replay buffer to each predicted embedding.
   To decide (open question).

**Throughput:** GameBoy ~450 env steps/s, but a VLM policy call per step dominates (~0.1-1 s);
Android ~3-5 s/step plus resets; Web ~1-5 s/step with `fast_waits`. Single-environment PPO,
as in the original.

---

## Order of work

1. Part 0: shared vLLM env + serve/stop scripts; download Gemma 4 26B MoE; smoke test.
2. Part 1: env contract additions → shared skeleton + records + dataset format → GameBoy
   executor (check it reproduces GameBoyRL's prompts) → M3A executor → WebVoyager executor →
   shared stages → small-scale test.
3. Part 2: frozen encoder + curiosity buffers + replay → VLM-PPO loop (GameBoy first, the
   fastest) → trajectory extraction/grouping → world model → Android/Web runs.

## Open questions

1. **GameBoy practice:** port GameBoyRL's executor into `cusi_practice` (one pipeline, one data
   format; must match GameBoyRL's behaviour), or run GameBoyRL's own practice pipeline for
   GameBoy and convert its output? (Leaning: port, since everything else here is shared.)
2. **Android agent:** M3A only, or T3A (text-only) as well?
3. **Pretrained encoder:** which model (e.g. SigLIP 2 base vs larger)?
4. **Policy VLM for PPO:** which small VLM (it must fit training on the node alongside vLLM, or
   run in its own job)?
5. **World-model executor display:** trained decoder vs nearest-real-frame retrieval (Part 2, 8).
6. **Element-index cap K** for the Android/Web world-model action vocabulary.
