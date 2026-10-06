# Curiosity plan: a fuller port of GameBoyRL's curiosity into `cusi.explore`

Written 2026-10-04. Scope: get the **PPO curiosity script** right, with GameBoyRL's curiosity
modules and frame embedders working on all three environments (GameBoy, Android, Web). Plus
rough **random** and **human** debug scripts.

**Out of scope for now:** other RL algorithms, iterative rounds (buffer of agent k as the
prior of agent k+1), sweeps / best-k selection, many start states per run. Nothing below
should block adding them later.

---

## 1. What GameBoyRL does (the reference)

Files: `GameBoyRL/cleanrl/cleanrl/ppo_curiosity.py`,
`GameBoyRL/cleanrl/cleanrl_utils/port_gameboy_worlds/{curiosity,embedders}.py`,
`GameBoyRL/cleanrl/cleanrl_utils/train_observation_encoder.py`,
`GameBoyRL/cleanrl/cleanrl/{random,human}_curiosity.py`.

**Frame embedders** (`--observation_embedder`). Both work on the 144x160 grayscale screen and
split it into non-overlapping k x k patches (`extract_patches`, k = 8 -> 18 x 20 = 360 patches):
- `random_patch` (`PatchProjection`): one fixed random linear map (seed 42, N(0,1) weights)
  from each 64-pixel patch to `patch_dim` = 2 numbers; concatenate -> 720-d; L2-normalise.
  So cosine similarity between two frames compares them patch by patch.
- `cnn` (`CNNEmbedder`): a small per-patch conv autoencoder (patch BatchNorm -> 3 convs ->
  Linear -> Sigmoid -> LayerNorm, `patch_dim` = 4 -> 1440-d), trained on replay frames by
  `train_observation_encoder.py` (reconstruction), loaded with `--embedder_load_path`. Has a
  `decode()` back to pixels (used by the world-model executor).

**Curiosity modules** (`--curiosity_module`, built by `get_curiosity_module`):
- `embedbuffer` (`EmbedBuffer`): novelty of the new frame = 1 - max cosine similarity to the
  buffer (also `distance` and `hinge` metrics). New embeddings are added unless some buffer
  entry is within 0.001 in every dimension (dedup). At 10,000 entries, KMeans compacts the
  buffer to 5,000 centres. The first frame after a reset scores 0 (`first_add`). An optional
  prior (`--buffer_load_path`) is loaded on every reset; `iterative_save` merges this agent's
  buffer into `--buffer_save_path`.
- `combinationbuffer` (`CombinationBuffer`): `ocr_alpha * OCRBuffer + (1 - ocr_alpha) * EmbedBuffer`.
  `OCRBuffer` is **not text recognition**: GameBoyWorlds returns pixel crops of fixed screen
  regions in `info["ocr"]["ocr_regions"]` (e.g. `dialogue`, `menu`, each a fixed shape). Each
  crop is cut into 8 vertical strips; a strip is novel if no stored strip of that region
  matches it to within 0.001 per pixel; region score = fraction of novel strips; reward = max
  over regions (1.0 the first time a region appears). Buffer per region, random subsampling at
  10,000.
- `world_model` (`WorldModel`): reward = 1 - cos(predicted next embedding, real next
  embedding), using the same embedder. The model is loaded from `--buffer_load_path`
  (trained between rounds by `train_world_model.py`); it is **not trained during the PPO run**.
  Without a loaded model it is randomly initialised.
- `clusterbuffer`: not wanted here.

**Reset.** `reset_curiosity_module` (default True) resets the module every episode. Here:
always reset, no flag.

**Baselines.** `random_curiosity.py`: random actions, same curiosity module and replay
buffer, so the reward signal can be checked without a policy. `human_curiosity.py`: replays a
hard-coded `input_sequence` of actions (fill in the list, it asserts it is non-empty) and logs
the curiosity reward per step. It is a scripted replay, not interactive.

## 2. What CUSI has now (`cusi/explore/`)

- One embedder: frozen SigLIP 2 (`encoder.py`, required `--encoder_model`), one 768-d pooled
  vector per frame (frame squashed to 224 x 224).
- One curiosity module, always the combination: `EmbedBuffer` (ported) +
  `TextNoveltyBuffer` (share of on-screen element lines never seen this episode), mixed by
  `text_alpha` (forced to 0 on GameBoy).
- Always resets per episode; `--buffer_load_path` prior exists; no world-model curiosity.
- PPO (`ppo.py`) with the sequence-ratio problem (see section 7).

---

## 3. Target design

### 3.1 Frame source and canvas per environment

The embedders see the **unlabelled** screen, converted to a fixed-size **canvas** per
environment so the patch grid is fixed:

| env | source frame | native size | canvas (H x W) | patches (k = 8) |
|---|---|---|---|---|
| GameBoy | `obs["frame"]` (grey, 3x repeated) | 144 x 160 | 144 x 160 (native, as GameBoyRL) | 18 x 20 = 360 |
| Android | `info["raw_frame"]` (no set-of-mark labels) | 2400 x 1080 | 256 x 112, RGB | 32 x 14 = 448 |
| Web | **new**: unlabelled screenshot (see below) | about 768 x 1024 | 192 x 256, RGB | 24 x 32 = 768 |

- **Colour wherever the environment has it** (Android, Web: RGB canvas; GameBoy is grey
  natively, so 1 channel). The canvas sizes above are kept.
- **Web change needed:** `WebVoyagerPlayEnv` only returns the set-of-mark screenshot
  (D1.8: "there is no unlabelled one"). `get_web_element_rect` draws the labels, the env takes
  the screenshot, then removes the labels; a second screenshot after the removal gives the raw
  frame (about 0.1 s more per step). Expose it as `info["raw_frame"]`, as Android does. The
  labels otherwise add patch novelty that is not content.
- The canvas spec lives in one place (a small per-env table in `cusi/state/`, see 3.2),
  used by every embedder, the world model and the decoder.

### 3.2 Shared state module (`cusi/state/`): what curiosity and pre-exploration share

Two consumers need the same machinery:
- **curiosity PPO** (this plan): novelty rewards for policy training;
- **pre-exploration search** (`plans/skill_discovery.md` §1.4): novelty of states found during
  rollouts, whether a step becomes a node, and visit counts over cells.

What they share is **using embeddings** (not training them) and **scoring novelty**. That goes
in one top-level package, `cusi/state/`. Everything about *lifetime and shaping* stays with
each consumer: when an archive is created, reset or kept, and reward penalties and
normalisation.

`cusi/state/` has three layers, all **inference-only** (no optimizers, no training loops),
with batched APIs (PPO needs throughput).

#### 3.2.1 Layer 1: encoders (`cusi/state/encoders/`)

- **Input:** the observation plus info. The frame is the **unlabelled** frame on the per-env
  canvas of 3.1 (`info["raw_frame"]` on Android, and on Web once added).
- **Text input (provisional):** `obs["texts"]` flattened to `key: text` lines, bounding boxes
  ignored. The text modality across envs still needs proper shaping (section 4; GameBoy has no
  text channel yet).
- **Image embedders**, one interface: `preprocess(frame) -> canvas`,
  `embed(frames) -> (N, D)` L2-normalised, `output_dim`, `load(path)`, optional `decode`:
  - **`random_patch`**: `PatchProjection` generalised to any canvas and channel count (k = 8,
    patch of k x k x C pixels -> `patch_dim` = 2, fixed seed 42). Needs no training.
  - **`cnn`**: `CNNEmbedder` generalised to any canvas and channel count, loaded from a trained
    checkpoint (3.2b). Keeps `decode()`.
  - **`siglip`**: SigLIP 2 (model id required, never defaulted), **always one pooled vector**
    per frame (768-d, L2-normalised). Either **zero-shot** (frozen weights) or **fine-tuned**,
    loaded from a 3.2b checkpoint.
  - Room for other learned image models later, behind the same interface.
- **Text embedders:**
  - **`dense`**: a sentence-embedding model (model id a required flag), L2-normalised vector.
  - **`tfidf`**: hashed bag of words, document frequencies kept by the archive, cosine. No
    model.
  - **`overlap`**: Jaccard overlap of token (or line) sets, the cheapest option.
  - **`none`**: no text part, e.g. GameBoy for now.
- **`StateEncoder(image=..., text=..., w_image=...)`:**
  `encode(obs_list, info_list) -> [StateEmbedding]`. A `StateEmbedding` holds the image
  vector, the text representation, and both embedder names.
- **Similarity:**
  `similarity(a, b) = w_image * sim_image(a, b) + (1 - w_image) * sim_text(a, b)`.
  - `sim_image` metric: cosine / distance / hinge, as GameBoyRL.
  - `w_image` is forced to 1 when text is `none`.
- **Checkpoint metadata** (embedder name, env, canvas, base model id) is checked on load. A
  checkpoint trained for another env or canvas is an error.

#### 3.2.2 Layer 2: archive (`cusi/state/archive.py`)

`NoveltyArchive(metric, dedup_threshold, max_size, compact_to)` is GameBoyRL's EmbedBuffer
storage logic generalised to `StateEmbedding`:
- `add`, `max_similarity`, `nearest`;
- 0.001 dedup;
- KMeans compaction (10,000 → 5,000);
- `save`/`load`;
- `copy()` / `restore(snapshot)`, so a consumer can "reset to the prior" without the archive
  knowing what an episode is;
- **cells**: `cell_of(embedding)` assigns a state to an existing cell when its similarity
  passes a threshold, else opens a new one, with per-cell counts. PPO gets it for free; the
  search uses it for visit counts.

#### 3.2.3 Layer 3: novelty scorers (`cusi/state/scorers/`)

One interface for all scorers:
`score(prev, action, next, archive) -> (value, components)`, on a transition.
- It is **pure with respect to lifetime:** it reads and, when asked, adds to the archive it is
  handed. It never decides when to reset.
- `components` (frame / region / ...) are returned for logging (3.7).

The scorers:
- **`embedding`**: `1 - max similarity(next, archive)` with the chosen metric (cosine /
  distance / hinge), as EmbedBuffer.
- **`region`**: novelty of text regions.
  - **GameBoy:** **GameBoyRL's `OCRBuffer`, ported unchanged**, on `info["text_regions"]` (the
    GameBoyWorlds region crops `GameBoyPlayEnv` already passes through). GameBoy gets this
    now, without waiting for [GB-OCR], which is about text for the executor's prompt.
  - **Android / Web:** section 4 decides between this and the text part of the encoder.
- **`combination`**: `region_alpha * region + (1 - region_alpha) * embedding`.
- **`world_model`**: `1 - cos(predicted next embedding, real next embedding)`.
  - Model: the existing `world_model.py` architecture (frame stack 2 + action index from
    `action_vocab.py`), **loaded** from `--world_model_load_path`, which is required with this
    scorer (a randomly initialised world model gives a meaningless score).
  - On load, it is checked to have been trained with the same encoder and action vocabulary
    (`action_space.json`, embedder name).
  - Archive resets don't affect it.

#### 3.2.4 Factories, flags, ownership

- **One factory each:** `build_encoder(config)` and `build_scorer(config)`, with **the same
  flag names** in `run_explore.py ppo` and the search entry point:
  - `--image_embedder`, `--encoder_model`, `--embedder_load_path`;
  - `--text_embedder`, `--text_embedder_model`, `--w_image`;
  - `--similarity_metric`, `--novelty_scorer`, `--region_alpha`, `--world_model_load_path`.

  So novelty numbers from both consumers are comparable.
- **Ownership:**
  - **This plan owns** the *list* of encoders and scorers and their behaviour.
  - **`plans/skill_discovery.md` §1.4 owns** how the search *uses* them.
  - **The interface** (`StateEncoder`, `StateEmbedding`, `NoveltyArchive`, the scorer
    signature) changes only with a note in both plans.
  - After the migration below, both tracks only *add* encoders and scorers.
- **Migration first:**
  - Move `cusi/explore/encoder.py` and the current buffers into `cusi/state/` *without changing
    behaviour*.
  - Gate it with a regression test: curiosity rewards on a fixed saved replay match the current
    code (exactly, apart from the stale-reward fix already in CUSI).
  - Only then generalise.

### 3.2b Training the learned embedders and world models (stays in `cusi.explore`)

Training is curiosity-side only. It writes checkpoints in the format 3.2.1 loads (weights +
metadata).
- **`cnn`:** a port of `train_observation_encoder.py` on replay frames
  (`run_explore.py train_embedder --embedder cnn --run_names ...`).
- **`siglip` fine-tuning per environment:**
  - **Objective:** an **observation reconstruction loss**, the same as the `cnn` autoencoder:
    SigLIP's pooled vector -> the pixel decoder (`decoder.py`) -> the canvas, L1 + MSE, encoder
    and decoder trained together (`train_embedder --embedder siglip`).
  - Low learning rate (or LoRA on the vision tower), so it specialises without forgetting.
  - Note: the current decoder's reconstructions from the frozen pooled vector are blurry with
    no legible text (end report). Fine-tuning end-to-end is what should improve that, but a
    single 768-d vector is a tight bottleneck for a full screen.
- **World models:** trained separately and loaded by the `world_model` scorer. How to train
  them well needs its own plan; the current one is no better than "nothing changes"
  (section 7).

### 3.3 Curiosity modules (`cusi/explore/curiosity/`, `--curiosity_module`): thin wrappers

A curiosity module = a `cusi.state` scorer + the module's own archive + PPO-side lifetime and
shaping. GameBoyRL's interface is kept on the outside: `get_reward(obs, action, next_obs, info)
-> float`, `reset()` (called every episode, always), `save()`, built by
`get_curiosity_module(...)` as in GameBoyRL.
- **`embedbuffer`** = the `embedding` scorer. **`combinationbuffer`** = the `combination`
  scorer. **`world_model`** = the `world_model` scorer.
- **Lifetime:**
  - `reset()` restores the archive to its prior (`--buffer_load_path`, or empty) via
    `copy()`/`restore()`;
  - `first_add` semantics as GameBoyRL;
  - `save()` writes the archive.
- **Shaping, curiosity-only (not in `cusi.state`):**
  - `--invalid_action_penalty`;
  - `--normalize_curiosity_reward`.

### 3.4 PPO flags (`run_explore.py ppo`)

The encoder and scorer flags are the shared `cusi.state` flags (3.2.4), named the same as in
the search entry point.

Required (never defaults, never in config): `--curiosity_module {embedbuffer,combinationbuffer,world_model}`,
`--image_embedder {random_patch,cnn,siglip}` (was `--observation_embedder`),
`--text_embedder {dense,tfidf,overlap,none}`; `--encoder_model` when the image embedder is
`siglip`; `--text_embedder_model` when the text embedder is `dense`;
`--world_model_load_path` when the module is `world_model`.
Optional: `--embedder_load_path` (trained `cnn` / fine-tuned `siglip`),
`--w_image` (image vs text weight in the similarity, 3.2.1),
`--similarity_metric {cosine,distance,hinge}` (default cosine, as GameBoyRL),
`--region_alpha` (replaces `text_alpha`), `--buffer_load_path` (prior, as now),
`--normalize_curiosity_reward` (default false),
`--invalid_action_penalty` (default 0.1: an action the environment rejects gets reward
-0.1 on top of its curiosity reward, which is 0 for the unchanged screen; 0 turns it off),
`--lora_r` (default 128), `--lora_alpha` (default 256).
Removed: `--ratio_mode` (per-token PPO only, section 3.6), `--text_alpha` (now `--region_alpha`),
`--observation_embedder` (now `--image_embedder`).
All recorded in the run's `config.json`.

### 3.5 Policy model: Qwen3.5-0.8B

`Qwen/Qwen3.5-0.8B` (March 2026, natively multimodal, hybrid Gated DeltaNet + attention
layers) is the policy model for now, replacing Qwen3-VL-2B. It stays a **required flag**
(`--policy_model Qwen/Qwen3.5-0.8B`), per the never-default-a-model rule; it is the model used
in the usage examples once the code supports it. Code changes:
- `cusi/explore/policy.py` loads `Qwen3VLForConditionalGeneration` explicitly; switch to the
  generic `AutoModelForImageTextToText` / `AutoProcessor` (and check the chat template and image
  token handling for Qwen3.5).
- `LORA_TARGETS` names Qwen3-VL's projections (`q_proj`, ..., `down_proj`); the DeltaNet layers
  have different module names, so derive the targets from the loaded model's linear layers.
- Thinking mode off (non-thinking chat template), matching short action-only replies.
- First step before anything else (section 6, step 0): check that our `transformers==5.6.2`
  and PEFT load it, run `generate` with an image, take LoRA on the DeltaNet layers, and that the
  fast kernels for those layers (flash-linear-attention) are installed, measuring tokens/s
  against Qwen3-VL-2B. If any of this fails, fall back to Qwen3-VL-2B and record why.

### 3.6 PPO objective (fixed, no flag)

Standard per-token PPO for text policies, replacing the sequence-level ratio:
- ratio per generated action token, `r_t = exp(logp_new(t) - logp_old(t))`;
- clipped surrogate per token with the step's advantage on every token, clip 0.2;
- averaged over the action's tokens, then over the minibatch;
- KL penalty to the reference model per token (as now), value loss as now;
- learning rate back to 1e-5 (2e-6 was a workaround for the sequence ratio); the 0.1
  target-KL early stop stays as a safety net; approx KL and clip fraction logged per token.
`--ratio_mode` and the `token_mean` variant are removed.

LoRA: rank 128, alpha 256 by default (flags), on the targets derived from the model (3.5).
The per-iteration cost is dominated by the frozen base model's forward/backward passes, so a
high rank costs little extra time; it mainly adds adapter and optimizer memory (small at 0.8B).
A higher-capacity adapter can move the policy further per update, so watch approx KL / clip
fraction and the KL to the reference in the first runs.

### 3.7 Replay

Keep frames (compressed), texts, actions, rewards, episode boundaries, plus the per-step
curiosity reward components (frame / region) for debugging. Store the active embedder's
embedding **and** its name; world-model, decoder and grouping code re-embed from frames when
they need a different embedder (the current code assumes stored SigLIP vectors).

---

## 4. The OCR question for Android and Web

GameBoyRL's OCR channel is "pixel novelty of the screen's text regions". Android and Web have
no fixed text regions, but they do have something GameBoy does not: the accessibility tree
(Android) and the DOM (Web) give every on-screen element's **exact text and bounding box**.
Options:

- **A. Text from the tree / DOM (recommended).** Novelty of the element text lines
  (`obs["texts"]`), what `TextNoveltyBuffer` already does: exact text, no extra compute, and
  the closest thing to "new words on screen". Keep D2.16's fix (Web's element list is
  tab-separated on one line). Dedup on normalised text (strip index numbers, as now).
- **B. Element crops (closest to the GameBoy mechanism).** Crop each element's bounding box
  from the raw frame, resize to a fixed strip size, key the buffer by element role/class, and
  use GameBoyRL's strip-matching `OCRBuffer`. Same mechanism as GameBoy, but exact pixel
  matching is brittle on anti-aliased, animated and live web content, and crops vary in shape.
- **C. Real OCR** (e.g. PaddleOCR / RapidOCR) on the raw frame -> text lines -> option A's
  buffer. Uniform across all three environments (it could also serve [GB-OCR]), but adds
  about 0.1-0.5 s per frame and OCR noise, and duplicates text the tree already gives exactly.

**Decision: A for Android/Web, GameBoyRL's `OCRBuffer` for GameBoy.**

TODO (OCR channel for Android/Web): option A may have to change (to B or C), e.g. if
tree/DOM text misses text drawn into images or canvases, or if a channel uniform with GameBoy
is wanted. Put this TODO in the code next to the Android/Web region-novelty buffer.

---

## 5. Random and human debug scripts

One click file, `debug_curiosity.py`, rough and outside the main framework:

- `random`: run an environment with `env.sample_action()` (added in Part 1), any curiosity
  module + embedder, for N steps; log the reward per step and per component; write the same
  replay format. Doubles as a cheap data source for training the `cnn` embedder, SigLIP
  fine-tuning and the world model (no VLM needed).
- `human`: replay a list of action strings from a file (GameBoyRL style), or, with
  `--interactive`, read actions from stdin one at a time, printing the reward after each step and saving
  each frame with its reward written on it, to check that the reward does what we expect
  (e.g. entering a new room gives a spike, walking in place gives ~0).

---

## 6. Order of work

0. Qwen3.5-0.8B check (3.5): loads, generates with an image, takes LoRA, speed vs Qwen3-VL-2B.
1. `cusi/state/` migration (3.2.4): move `cusi/explore/encoder.py` + the current buffers
   into the three layers with no behaviour change; regression test on a fixed replay.
2. Generalise in `cusi/state/`: canvas table + `random_patch` (all envs), Web raw screenshot,
   text embedders (`tfidf`, `overlap`, `none` first; `dense` after), `NoveltyArchive` with
   cells, scorers `embedding` / `region` (GameBoy `OCRBuffer` port + Android/Web option A) /
   `combination`, the shared factories and flags.
   Then the curiosity wrappers (3.3): `embedbuffer`, `combinationbuffer`, always-reset.
3. `debug_curiosity.py random` / `human`; check on each env that the rewards look sensible
   (sizes, spikes on new screens, ~0 when nothing changes) before any PPO.
4. PPO fixes (3.6 per-token objective, short replies, invalid-action penalty, LoRA r128/a256,
   reward-normalisation flag) and the new flags; a short GameBoy run per curiosity module.
5. `cnn` embedder in `cusi.state` + `train_embedder` in `cusi.explore` (on random-run replay).
6. `world_model` scorer in `cusi.state` (loading a trained world model only).
7. `siglip` pooled embedder in `cusi.state` + optional reconstruction fine-tuning (3.2b).
8. Android and Web PPO dev runs.

---

## 7. Issues from the first run that matter here (plans/decisions.md / plans/end_report.md)

- **PPO ratio (D2.6, D2.7, D2.15; end report item 1).** The sequence-level ratio over ~150
  tokens either clips almost everything or, with the token-mean variant, effectively turns
  clipping off (token-mean ratio ~ sequence ratio^(1/150), so a +-0.1 clip allows ~1.1^150).
  **Decided:** per-token PPO, no flag (section 3.6).
- **Invalid actions (end report: 58% valid on Web, 77% on Android).** An invalid action was a
  no-op step with 0 novelty and no other cost. **Decided:** `--invalid_action_penalty`, default
  0.1. Action-only replies may also raise validity; watch it in the first runs.
- **Dev runs not comparable (end report).** The GameBoy runs changed scene, ratio and learning
  rate together; Android/Web used the old settings. **Decided:** after the fixes, one
  comparison run per question on the same scene, changing one thing at a time.
- **Throughput (end report).** GameBoy runs at ~0.11 steps/s (~9 s/step) though the emulator
  does ~450 steps/s: one unbatched `generate` of a ~150-token reply per step. Every module
  above is cheap next to that (patch embedders: < 1 ms). This decides how long any curiosity
  comparison takes (Q6).
- **Reward scale (end report; left for now).** Frame novelty with frozen SigLIP on GameBoy averages ~0.015
  (screens are close in SigLIP space), while Android/Web text novelty is ~0.3 per step, so a
  0.5/0.5 mix is dominated by text. Patch embedders will change the frame scale; check scales
  in the random runs (step 3) before choosing `region_alpha`. Reward normalisation: Q7.
- **World model ≈ copy baseline (end report).** With SigLIP embeddings the world model does
  no better than "nothing changes", even on training data. World-model curiosity built on it
  would give a reward that mostly measures frame change. Patch embedders may help (more of the
  embedding changes with the screen); measure before relying on it.
- **First transition scores 0 (D2.9)** and the buffer resets every episode: kept.
- **Android raw screenshot (D2.9, D1.8)**: already available as `info["raw_frame"]`; Web needs
  the change in 3.1.
- **Web text bug (D2.16)**: keep the element-splitting fix in option A.
- **Web stale observations**: a failed action returns the previous observation
  (`info["stale"]`), so novelty is 0 for it, which is right.
- **Live websites**: a Web reset restores the browser, not page content, so a saved prior
  (buffer) can go stale between runs.
- **Action vocabulary and K (D2.10, decision 18)**: the world model's action index; K = 64 /
  128 held in the dev runs.
- **Element-index meaning (plan Part 2, item 7)**: on Android/Web an index means a different
  element on every screen, so world-model curiosity there has weaker action information than
  on GameBoy.
- **Android resets take 10-30 s (plan) and can fail (D1.18)**; Chrome starts are serialised
  (D1.15). Relevant if several environments run in parallel (Q6).
- **Never default a model (CLAUDE.md)**: `--encoder_model` stays required for `siglip`, and
  `--curiosity_module` / `--image_embedder` / `--text_embedder` are required flags too.
- **Reward normalisation (GameBoyRL)**: `gameboy_worlds_make_env` wraps the env in
  `gym.wrappers.NormalizeReward(gamma)`, but that only scales the **environment's** reward
  (0 in curiosity runs). `ppo_curiosity.py` adds the curiosity reward after `env.step`, outside
  the wrapper, so GameBoyRL's intrinsic reward is raw, as ours is now.

---

## 8. Decisions on the open questions (2026-10-04)

1. Android/Web OCR channel: **A** (tree/DOM text), with a TODO that it may change.
2. Canvas: **colour where available**; sizes in 3.1 are fine.
3. World-model curiosity: **load only, no training during PPO**; training world models gets its own plan.
4. SigLIP fine-tuning: **observation reconstruction loss** (through the decoder); SigLIP always gives one pooled vector.
5. Web raw screenshot: **yes**, second screenshot per step.
6. Throughput: **short policy replies, in scope; no batching / parallel environments for now.**
   Each PPO step currently generates ~150 tokens (reasoning + action), ~9 s/step on GameBoy.
   The policy prompt asks for the action only (no reasoning block), with `max_new_tokens` sized
   to the longest valid action (~15-30 tokens; Android/Web actions with typed text are the
   longest). Cutting `max_new_tokens` alone would truncate replies before the action, so the
   requested format must change. Trade-off: the policy's prompt then differs from the native
   agents' format (plan Part 2 reused their prompts so the policy learns the evaluation
   format); a `--reply_format {action_only,native}` flag keeps both. **Policy model:
   `Qwen/Qwen3.5-0.8B`** (replaces Qwen3-VL-2B; see 3.5). Other candidates kept for later:
   LFM2.5-VL-1.6B, and GUI-trained 2B models (MAI-UI-2B, GUI-Owl-1.5-2B) for Android/Web. Batching (several GameBoy instances
   per `generate`; 2-4 emulators/browsers for Android/Web) is deferred.
7. Reward normalisation: GameBoyRL does **not** normalise the intrinsic reward (section 7). Add
   `--normalize_curiosity_reward` (default false): divide the curiosity reward by a running std
   of the discounted curiosity return (as `NormalizeReward` does for env rewards).
8. `--curiosity_module` / `--image_embedder` (was `--observation_embedder`): **required flags**,
   never in config.
9. Human script: action file **and** `--interactive` stdin.


---

## 9. First build: decisions (2026-10-04)

<!-- TOY-BEGIN: throwaway decisions for the first build; delete this block to strip them. -->

**Goal:** everything *runs* end to end; correctness and tuning come later.
- No full tests, no long runs.
- Each component is exercised on **one env**.
- Exception: PPO itself gets a short smoke run on **all three** envs, to make sure it runs.

**Steps (section 6):**
- **Steps 0–4 in full:**
  - the Qwen3.5-0.8B check;
  - the `cusi.state` migration with its regression test on a fixed replay;
  - the generalisation;
  - the debug scripts;
  - the PPO fixes.
- **Steps 5–7 built, tested on a small scale only:**
  - `cnn` trained for a few minutes on a random-run replay;
  - `world_model` scorer loading a tiny trained model;
  - SigLIP fine-tune for a handful of steps.
- **Step 8:** short PPO smoke runs on Android and Web.

**Envs per component:**

| Component | Env |
|---|---|
| Migration regression test, `region` (OCRBuffer) scorer, `cnn`, `world_model` | GameBoy |
| `debug_curiosity.py random` / `human` | GameBoy (random), Android (human, a few actions) |
| Text embedders (`overlap`, `tfidf`, `dense`), option A text novelty, Web raw screenshot | Web |
| PPO smoke runs | GameBoy, Android, Web (a few iterations each) |

**Models (flag values in the scripts, never code defaults). Start small, scale up only if a
broken output format breaks the run:**
- **Policy:** `Qwen/Qwen3.5-0.8B` (step 0 decides), falling back to the cached
  `Qwen/Qwen3-VL-2B-Instruct`.
- **SigLIP:** `google/siglip2-base-patch16-224` (cached).
- **Dense text:** `sentence-transformers/all-MiniLM-L6-v2` (cached), via plain `transformers` +
  mean pooling.

**Shared with `plans/skill_discovery.md` §1.5:** the `cusi.state` migration goes first, then the
search's minimum pieces (`random_patch`, `overlap`/`tfidf`, archive with cells, `embedding`
scorer), then both plans continue on separate files.

### First build results (2026-10-04)

Everything in steps 0–8 runs. Logs: `$results_dir/logs/`; outputs: `storage_dir/explore/<env>/`.

| Step | What ran | Result |
|---|---|---|
| 0 | `tests/policy_model_check.py` (jobs 282983, 282988) | Qwen3.5-0.8B loads via `AutoModelForImageTextToText`, LoRA r128/a256 on 12 derived targets incl. DeltaNet `in_proj_*`/`out_proj` (86.6M params), backward ok, act vs evaluate per-token log-probs identical (48-token reply). 0.47 s/act vs 0.35 s for Qwen3-VL-2B (2-token replies). **Kept Qwen3.5-0.8B.** |
| 0 | `uv add causal-conv1d` | **Skipped**: no nvcc/CUDA toolkit on the login node, torch 2.11+cu130 has no prebuilt wheel. Qwen3.5 runs with fla's triton delta-rule kernels on the host and the torch conv fallback. In the container (no C compiler) triton can't JIT, so `policy.py` disables fla there (torch implementation). |
| 1 | `tests/cusi_state_regression_test.py` | Exact (max diff 0) vs the pre-migration code on 1588 GameBoy + 342 Web replay steps (cosine/distance/hinge), KMeans compaction, SigLIP embeddings. |
| 2 | `tests/cusi_state_unit_test.py`; `tests/web_state_check.py` (arXiv, container) | `random_patch` == GameBoyRL PatchProjection; Web `raw_frame` present every step (labels change 1–10% of pixels); option-A region novelty 1.0 on new pages / 0 on repeats; overlap / tfidf / dense (MiniLM) similarities sensible, self-similarity 1. |
| 3 | `debug_curiosity.py random` GameBoy, 3000 steps (17 s), `combinationbuffer` + `random_patch` | frame novelty mean 0.042 (SigLIP was ~0.015), exactly 0 when the screen is unchanged; OCR region 1.0 when a menu/dialogue appears, 0 for 89% of steps. |
| 3 | `debug_curiosity.py human` Android (job 282984, 8 actions) and GameBoy `--interactive` | new app → region 0.75 / home screen 0.52; repeats → 0; invalid click → −0.1. Annotated frames in `debug/<run>/frames/`. |
| 5 | `train_embedder --embedder cnn` on the random replay (CPU, 4.6 min, 6 epochs) | val MSE 0.89 → 0.58; decoded pixel MAE 41 (toy). Loads via `--image_embedder cnn --embedder_load_path`; frame novelty mean 0.11. |
| 6 | `world_model --image_embedder random_patch` on the random replay (30 epochs, CPU) | val cos 0.949 vs copy baseline 0.935 (changed frames 0.949 vs 0.924): unlike SigLIP, it beats "nothing changes". `world_model` scorer loads it, checks embedder + action vocab (mismatch refused); novelty mean 0.056 in a random run. |
| 7 | `train_embedder --embedder siglip`, 10 steps CPU | loss 0.54 → 0.38; `vision_model.pt` + meta load in `SiglipEmbedder`; frame novelty 0.047 (zero-shot 0.015). |
| 4/8 | PPO smoke, 2 iterations each (`explore.sh`): GameBoy embedbuffer / combinationbuffer / world_model (282987, 282989, 282994), Web (282996), Android (282997) | All run end to end. GameBoy ~0.8–1 step/s with action-only replies (was ~0.11), 2 tokens per action once the policy settles. Web 0.14–0.18 step/s, 37–50% valid, 7–9 tokens. Android 0.06 step/s (emulator-bound), 100% valid, 9–10 tokens (with the prefill below; 25–50% and 79–96 tokens without). |

Fixes made on the way:
- `world_model` scorer: a rejected action (unchanged screen) now scores 0. The random-agent replay has no "invalid" transitions, so the model's arbitrary prediction for that index rewarded invalid actions (novelty 0.57 vs 0.06).
- `action_only` replies get a prefilled start (`RESPONSE_PREFIX`): `Action:` on Web, `{"action_type": "` on Android. Qwen3.5-0.8B followed the native prompts' "Thought:" / "Reason:" formats over the appended instruction: 0% valid on Web, 79–96 tokens on Android.

Open issues (for the real runs):
- **Update size.** With LoRA r128/a256 and lr 1e-5, the first Adam step moves the policy far: approx KL 0.3–1.2 per token right after the first minibatch, so every iteration early-stops on target_kl 0.1, and kl_ref reaches ~4.6 by iteration 2. The objective is right (log-probs match exactly before the update). Try a lower lr (or rank) first.
- **Collapse.** In one GameBoy run (combinationbuffer + `--normalize_curiosity_reward`), iteration 2 collapsed to 16-token invalid replies (valid 0%).
- **first_add.** After a reset into an empty archive, the first two frames score 0 (GameBoyRL's semantics, kept). On Android this zeroes the first, often most novel, step.
- **Untested.** Dense text in PPO (it was only checked live on Web); `siglip` in PPO beyond the regression; world-model training on Android/Web replays.

<!-- TOY-END -->
