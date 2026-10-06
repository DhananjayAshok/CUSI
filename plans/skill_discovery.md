# Skill discovery

Shared design for how CUSI discovers what can be done in an environment and turns it into
skills, across GameBoy, Android (AndroidWorld) and Web (WebVoyager). Sections are added as
they come up.

---

## 1. Exploration procedure

Several self-improvement methods start with a **pre-exploration phase**: the agent interacts
with the environment *before* any task synthesis, attempts on tasks, filtering or training.
Its output (trajectories, visited states, transitions, screenshots) is what later stages
build on. Examples:
- **PAE (GameBoyRL):** the zero-shot attempt.
- **GameBoyRL / CUSI:** curiosity-driven RL exploration.
- **OS-Genesis:** interaction-driven exploration before reverse task synthesis.
- **AutoPlay:** an explorer phase before task generation.

**Goal:** one exploration procedure that covers these methods, with their minor differences
(who acts, how novelty is judged, budgets, resets, what gets recorded) smoothed out into
parameters. That way the downstream stages of every method can run on the same exploration
output, and comparisons between methods aren't confounded by how each one explored.

Sources: the code (GameBoyRL in this repo, the clones in `../SetupAttempt/repos/`) and the
papers. *Paper only* = no public code, so the description comes from the paper. AW =
AndroidWorld.

### 1.1 Methods with a separate pre-exploration phase

**PAE zero-shot propose → attempt (GameBoyRL; ported in `cusi.practice`).** *Goal-conditioned
exploration: the agent explores by attempting tasks it proposed itself.*
- **Propose** sees one frame only: `env.reset()` on each train init_state, the start
  screenshot plus the verbalised action list (`vlm_scripts/propose_tasks_zeroshot.py:51-72`).
  "Propose an exhaustive list of distinct tasks… grounded in what is actually visible or
  reachable". The list is not capped; CUSI caps it at `n_tasks` per scene and adds the env's
  text elements.
- **Attempt:**
  - The same VLM, with the goal, through the `single_actions` executor.
  - 50 steps; up to 5 tries, each from a fresh `env.reset()`.
  - After a failure, a critique hint for the next try.
  - 16 in parallel.
- **Judged** by a two-stage checker: a goal-free DESCRIBE over 10-frame slices, then a binary
  JUDGE on the description plus the last 8 frames, which also returns the "safe success point".
- **Output:**
  - all tries in `all_trajectories.csv`;
  - successes in `success_trajectories.{json,pkl}` (final successful try only);
  - CUSI also keeps the failed legs.
- **Consumers:** PAE guidance/practice and the info docs. The successful attempt trajectories
  are the demonstrations.
- GameBoyRL once fed curiosity output to propose as a prior. It was "near-inert (~3%)" and
  removed, so propose never sees exploration data.

**Curiosity-driven RL exploration (GameBoyRL; ported in `cusi.explore`).**
- **Actor:** cleanrl PPO with a small CNN on 2 stacked grayscale frames and discrete button
  actions; `num_envs=1`. CUSI uses a Qwen3-VL LoRA text policy with PPO + KL to the
  reference model.
- **Novelty** is an episodic archive, not RND or ICM: α·OCR novelty + (1−α)·frame novelty.
  - **Frame term:** 1 − max cosine to an archive of frozen random-projection patch embeddings
    (720-d). KMeans-compacted above 10k entries. CUSI uses SigLIP 2.
  - **OCR term:** the fraction of new strips in dialogue/menu text regions. CUSI uses text-line
    novelty.
  - **Extrinsic reward:** 0.
- **Schedule:** agent_0, then 3 more agents, each starting from the previous agent's archive
  as "already seen"; each run is swept over α ∈ {0, 0.75}.
- **Budget:** 100k steps per run, 30-step episodes, always reset to the same named init_state.
- **Recorded:** a replay buffer of frames, actions and intrinsic rewards.
- **Post-processing:**
  - z-score every transition's reward over the buffer.
  - Each transition with z > 2.5 is back-traced up to 30 steps.
  - Keep trajectories with z ≥ 6, then merge groups whose final frames have cosine ≥ 0.95.
  - Snip loops.
- **To tasks:** per cluster, ≤3 trajectories go through INFER (last 8 frames → task; "NO TASK"
  allowed) → REFINE (valid/invalid + rewrite) → DISTILL (one task string).
- **Exploration trajectories are themselves the demonstrations.** These tasks are never
  attempted; the RL policy is thrown away.
- GameBoyRL quirks: a stale reward on episode-end steps (fixed in CUSI), and the "seen" prior
  comes from eval rollouts, not training.

**GameBoyRL world-model buffers.**
- A uniform-random agent (10k steps per init_state), plus the curiosity PPO without grouping.
- Raw buffers train the observation embedder and the world model. No tasks.

**OS-Genesis** (Android, also a WebArena variant).
- **Actor:** a rule-based weighted random walk over the a11y tree, with no goal.
  - Weights: element 10, scroll up/down 1, back 1.
  - Click vs long-press 9:1.
  - For editable fields, gpt-4o-mini writes "text a user might input".
- **No novelty signal.** The only memory is a per-app blacklist of elements whose action left
  the a11y identifier set unchanged.
- **Start states:**
  - Loops over the 116 AW task classes; for each, `generate_random_params()` +
    `initialize_task()` seeds the app data.
  - Then go home → open that task's app (hardcoded task→app map).
- **Budget:** 10-step episodes, reset home after each, infinite relaunch loop, one emulator.
- **Recorded:** only transitions that changed the screen
  (before/after screenshots, element list, action, element).
- **To tasks:** each *single transition* → GPT-4o reverse synthesis (before/after with a red box
  on the element → sub-instruction → high-level instruction).
- **After exploration:**
  - Each task is re-executed by an M3A GPT-4o agent from the same AW-initialised state.
  - A trajectory reward model scores it 1–5, and the SFT data is sampled by score.

**OpenMobile** (Android).
- **Actor:** the OS-Genesis walker, nearly verbatim.
  - Scroll/back weight 3.
  - Deterministic seeds (`task_random_seed=222`, not the eval's 30).
  - `tear_down` after each episode.
  - One pass over the 116 AW tasks, 10 steps each.
- The paper deliberately keeps exploration simple: "the critical factor lies not in exploration
  efficiency, but in how … data is organized."
- **Novelty:** none during exploration. Afterwards:
  - pHash clustering of screens (similarity ≥ 0.95);
  - a transition graph;
  - an LLM labels elements as functionality vs data.
- **To tasks:** per anchor *state*, with context:
  - short-term: the screen, 1 predecessor and ≤3 successors;
  - long-term: 30 retrieved same-app functionalities.
  - The prompt asks for 1–3 tasks starting from home whose data is visible in the given
    screens.
  - An LLM judge plus refinement keeps ≤100 tasks per app.
- **After exploration:** rollout from the original AW task initialisation with the synthesised
  goal. Success = the agent's own `terminate(success)`. Then SFT.
- The paper measures overlap with AW test instructions: 3.5% have similarity > 0.7.

**AutoPlay** (Android + Ubuntu, *paper only*).
- **Actor:** an MLLM (GPT-4o planner/reflector + UI-TARS grounding), the same system as the
  executor, with a generic goal: "Explore the APP_NAME app exhaustively to access all features,
  functionalities and data stored on the app."
- **Novelty:** only the prompt plus per-app episodic memory. After each turn the trajectory is
  summarised to `{action summary, data stored}`, and earlier summaries go into later turns.
  No state hashing.
- **Start states:** app homepage "with data populated". How data is populated on AW is not
  stated.
- **Budget:**
  - AW: 20 apps × 3 turns. OSWorld: 13 apps × 5 turns.
  - Steps per turn not stated.
- **To tasks:** each *whole trajectory* × 4 guideline prompts (Feature-Use,
  Feature-Composition, Information-Retrieval, Feature-Repetition) → ≤50 templated tasks per
  pair, each starting from the trajectory's first state.
- **After exploration:** execute → GPT-4o verifier → SFT (~8k of 20k tasks succeed) → GRPO.
- **Ablations:** exploration beats both hand-written app descriptions and iterative
  exploration.

**MobileGUI-RL** (Android, *paper only*).
- **Actor:** random walks "not purely random", preferring unexplored UI elements and avoiding
  repetitive loops. Start states and budget not stated.
- **To tasks:** GPT-4o reverse-engineers tasks from the trajectories.
- **Filter:** an LLM *text world model* simulates the GUI from the home screen, and the base
  agent must reach `terminate(success)` within T_max. 1,251 → 436 tasks.
- **Curriculum:** easy → hard by simulated step count.
- **Reward:** a VLM judge. No AW checkers.

**mobile-use proactive exploration** (Android). *Non-parametric: the output is a text summary,
not tasks.*
- **Actor:** a VLM (Qwen2.5-VL-72B, temperature 1) told to familiarise itself with the app's
  functions and "explore the process of creating different type content".
  - Every 3 steps a critic gives advice: do something different, don't leave the app.
  - When before == after pixel-exactly, "pages are almost identical, try again".
- **Start state:** the AW tasks of that app initialised with `create_suite(seed=30)`, the
  benchmark's own seed. **It explores the exact test-task data states.**
- **Budget:** one continuous 100-step episode per app, no resets.
- **Output:** a rolling knowledge summary per app; at test time an LLM extracts a ≤100-token
  snippet into the prompt.

**SynthAgent** (Web). *VLM explorer with a Go-Explore-style return to frontier pages.*
Task synthesis runs inside the same loop (`synthagent.py:283-397`, whenever 8 transitions are
buffered), but the tasks never feed back into exploration: the explorer only reads its own
page visit counts and element pools. The only coupling is the stop rule (`synth_until_tasks`).
So it belongs here, not with the interleaved methods.
- **Actor:** a goal-free LLM groups the page's elements into categories and proposes an action,
  value and low-level instruction for each ("fully explore the current page state…"), plus
  weighted random sampling.
- **Coverage:**
  - **URL frontier:** a page joins it when an action leads to a new in-site page. A page is
    picked with weight `1/(visits+1)+0.1`, and its count then doubles.
  - **Element weights:** unexplored 4 vs explored 1.
  - **Pools:** a (url, action) pool of unexplored elements; no-op and off-site actions are
    blacklisted.
  - **No-op test:** a state is unchanged if URL, a11y tree and pixels all match.
- **Resets:** before *every* action, close tabs and reopen the chosen frontier URL. Every
  transition is one step deep; depth comes only from the frontier.
- **Budget:** 128 iterations, or until N unique tasks.
- **To tasks:** each *single transition* → the OS-Genesis-style reverse prompt, plus the site
  description and example tasks.
- **Downstream:**
  - The executor refines the task when it gives up (STOP).
  - Trajectories are scored keep/refine/drop.
  - SFT.

### 1.2 Methods that interleave exploration with task discovery or learning

Only methods where what is learned or created changes what gets explored next.

**GSAR** (Android, *paper only*). Explore → tasks → execute, iterated, before RL.
- **Explore:** random interaction per app.
- **To tasks:** a VLM writes tasks *per explored state* (app name, screenshot, UI components,
  action history). Each task's start state is that state.
- **Execute:** GPT-4o executes the tasks, which *mutates app data*. This "environment evolution"
  matters because freshly installed apps are nearly empty. Mismatched task/trajectory pairs are
  filtered out OS-Genesis-style.
- **Complexify:** successful tasks are made harder (inherit / merge / rewrite), and the next
  iteration starts from the evolved environment.
- **Budget:** 6 iterations for AW apps. Other apps get a single random exploration, and its
  number of unique pages sets 1–6 iterations.
- **Reward:** goal-state anchors from successful final screenshots.
- Training uses 42 AW tasks while evaluating on all of AW.

**SE-GA** (Android).
- **3 rounds**, each: interact with the environment (through its memory module) → relabel
  trajectories by *hindsight goal-shifting* (a failed prefix gets an alternative sub-goal it did
  achieve) → offline policy update.
- **Weak exploration:** the paper claims "model-free, interaction-driven traversal" but doesn't
  specify the explorer, novelty or budget. The case studies show the agent acting on given
  instructions. Most data comes from static datasets.

### 1.2b Methods with no exploration (task-driven only), for completeness

| Method | Where tasks come from |
|---|---|
| MobileRL | AW's own parameter generator (2k tasks, test instances removed); RL curriculum drops tasks that keep failing; AW checkers as reward |
| UI-Voyager | A seed generator perturbing AW templates (>7k); rejection sampling on AW checkers; fork-point relabeling between successful and failed rollouts |
| UI-Mem | 256 human dataset instructions (AMEX / AndroidLab / UI-Genie) + GPT-4o augmentation; memory-guided GRPO rollouts |
| WebRL | WebArena-Lite training tasks + GPT-4o self-instruct from *failed* tasks; the critic difficulty filter is in the code but never called |
| WebEvolver / WebCoT | OpenWebVoyager query files; the world model "dreams" trajectories for the same queries; WebCoT is post-processing only |
| EvoSkill-GUI | Benchmark tasks; a skill is generated from the instruction + initial screen only, then evolved on ground-truth success |
| Mobile-Agent-E | Given task lists; tips/shortcuts are updated after each task (the reflector also sees upcoming tasks) |
| WebCoach | WebVoyager tasks; memory fills from attempts only |

### 1.3 Differences across methods

| Axis | Variants seen |
|---|---|
| **Who acts** | Uniform/weighted random walk (OS-Genesis, OpenMobile, GSAR, GameBoyRL WM buffers) · heuristic random walk preferring unexplored elements (MobileGUI-RL) · goal-free LLM explorer with an "explore everything" prompt (AutoPlay, mobile-use, SynthAgent) · goal-conditioned LLM on self-proposed tasks (PAE zero-shot) · RL policy with intrinsic reward (curiosity) |
| **Novelty / coverage signal** | None · no-op blacklist (OS-Genesis, OpenMobile, SynthAgent) · visit-count frontier over pages (SynthAgent) · unexplored-element weights (SynthAgent, MobileGUI-RL) · summary memory in the prompt (AutoPlay, mobile-use) · embedding-archive novelty as reward (curiosity) |
| **State identity** | a11y identifier set (OS-Genesis/OpenMobile) · URL + a11y + pixels (SynthAgent) · exact pixels (mobile-use) · pHash ≥ 0.95, after the fact (OpenMobile) · embedding cosine (curiosity) · none (AutoPlay) |
| **Episode shape** | 1 step from a reset (SynthAgent) · 10 steps (OS-Genesis/OpenMobile) · 30 (curiosity) · 50 × ≤5 tries (PAE) · 100 continuous, no reset (mobile-use) · "turns" of unstated length (AutoPlay) |
| **Start states / app data** | AW `initialize_task` with random params (OS-Genesis), seed 222 (OpenMobile), or **the test seed 30** (mobile-use) · "homepage with data populated" (AutoPlay) · data built up by executing tasks (GSAR) · named save states (GameBoy) · homepage URL (SynthAgent) |
| **What exploration passes on** | Single transitions (OS-Genesis, SynthAgent, MobileGUI-RL) · states with neighbour/memory context (OpenMobile, GSAR) · whole trajectories (AutoPlay) · clusters of novel trajectories (curiosity) · task + successful attempt (PAE) · a text summary (mobile-use) |
| **Selection after exploration** | None · pHash dedup + LLM element labelling (OpenMobile) · z-score outliers + cosine clustering (curiosity) · judge on attempts (PAE) |
| **Is the exploration trajectory used as training data?** | Yes: curiosity (clusters become demonstrations), PAE (successful attempts) · No: re-executed from the same start (OS-Genesis, OpenMobile, AutoPlay, SynthAgent, GSAR) |
| **Budget** | A few hundred steps per app (AutoPlay 3 turns, mobile-use 100) · ~1.2k episodes of 10 steps (OS-Genesis/OpenMobile, one pass over AW tasks) · 8 × 100k steps per init_state (curiosity) |

**Leakage to keep in mind for AW:** every Android code path seeds app data through AW's
`initialize_task`. mobile-use uses the benchmark's own seed. OS-Genesis and OpenMobile map AW
*test-task* classes to apps.

### 1.4 A unified exploration procedure

**Constraints.**
- **One procedure for all three envs** (GameBoyWorlds, AndroidWorld, WebVoyager).
- **No domain-specific state information** (no URLs, page-visit counts, a11y ids, app names
  to identify states). It assumes only what all three share: an observation with an image and
  text. The text is a flat string or a dict of named regions with text, optionally with
  bounding boxes.
  - **Provisional text format:** flatten `obs["texts"]` to `key: text` lines and ignore
    bounding boxes.
  - **TODO:** the text modality across envs needs proper shaping (GameBoy has no text channel
    yet; Android/Web region and box structure).
- **Never goal-conditioned.** The explorer is never given a task. PAE's zero-shot attempt
  (attempting self-proposed tasks) is therefore *not* a pre-exploration method under this
  design.

**Shape: everything is a tree search.**
- Roots are the scenes' start states: GameBoyWorlds init_states, AndroidWorld scenes,
  WebVoyager start pages.
- One iteration:
  1. **select** a node from the whole tree;
  2. **restore** its state;
  3. **expand** it by running an exploration policy for several steps;
  4. **add** new child nodes;
  5. **update** the statistics.
- "Always restart from the scene start" (OS-Genesis, OpenMobile, AutoPlay, GameBoyRL
  curiosity) is the special case where selection only ever picks roots.

#### Env support: saved states (done)

All three envs now have `state_id = env.save_state()`, `env.load_state(state_id=...)` (returns
nothing; the observation is in `env.current_obs`; errors via `log_error`) and
`env.delete_state(state_id=...)`. `close()` deletes every saved state, and a loaded state
starts a new episode (step 0). Test: `tests/env_saved_state_test.py`.

| Env | How | Save / load time | Fidelity | Notes |
|---|---|---|---|---|
| GameBoy | GameBoyWorlds' custom states (`save/load/delete_custom_state`); the scene's init_state is put back after a load | 0.07 s / 0.03 s | exact: identical frame, deterministic replay | Files go to the game's `states/` dir as `custom_<id>.state` |
| Android | Emulator snapshots (disk + RAM, clock pinned) in the emulator's private AVD copy | ~3.9 s / ~9 s | exact restore of the device | **3.6 GB per snapshot**: only ~20 fit in 78 GB of `/tmp`. Snapshots are lost if the emulator is restarted after a hang |
| Web | URL + all cookies (CDP) + localStorage/sessionStorage + scroll, in a fresh browser, after waiting for the layout to settle | ~0 s / ~17 s | approximate: ~2% of pixels differ, same element text | Live sites; loses typed text, open menus/modals, URL-less in-page state |

Consequences:
- **Android needs lazy nodes.** Most nodes store only "nearest saved ancestor + the actions
  since", and get a real snapshot only when selected for expansion, with old snapshots evicted
  under a disk budget.
- **Android replay is not exactly deterministic:** the same action after a load produced a
  different screen once in testing. Lazy nodes therefore check the replayed observation
  against the stored one.
- **Web loads are slow.** The layout wait is the first thing to trim.

#### A. State comparison: the shared `cusi/state/` module

Embeddings (used, not trained) and novelty scoring are shared with curiosity PPO and specified
in **`plans/curiosity_plan.md` §3.2**. In short:
- **Layer 1, encoders:** `image_embedding` (`random_patch` / `cnn` / `siglip` zero-shot or
  fine-tuned / other learned models) and `text_embedding` (`dense` / `tfidf` / `overlap` /
  `none`) → `StateEmbedding`, plus a weighted similarity.
- **Layer 2, archive:** `NoveltyArchive`: add, dedup, compaction, `copy`/`restore`, and
  **cells** with per-cell counts.
- **Layer 3, scorers:** `embedding` / `region` / `combination` / `world_model`, on a
  transition, against an archive they are handed.

All three are inference-only. Training learned embedders and world models stays in
`cusi.explore` (curiosity_plan §3.2b).

**Ownership.**
- **curiosity_plan §3.2 owns** the list of encoders and scorers and their behaviour.
- **This section owns how the search uses them.**
- Interface changes need a note in both plans.
- The search builds encoder and scorer through the same factories and **flag names**
  (`--image_embedder`, `--text_embedder`, `--w_image`, `--novelty_scorer`, ...), so its
  novelty numbers are comparable with curiosity's.

**How the search uses it (search-side lifetime and policy, not in `cusi.state`):**
- **One archive for the whole search,** never reset. Curiosity, by contrast, resets its
  archive to the prior every episode.
- **Node creation:** a rollout step becomes a child node when its scorer value passes a
  threshold, or at the segment end.
- **Expansion yield:** the summed scorer values and the number of new cells an expansion
  produced.
- **Visit counts:** `archive.cell_of(...)` counts per cell, used as `N(n)` in the energy (C).

> **Open question:** should visit counts be over *clusters* of cells instead (e.g. cluster
> the cell embeddings periodically and count per cluster)? That gives a coarser "region of the
> environment" count that doesn't fragment as the archive grows.

#### B. Tree

- **Node:**
  - state handle (a saved `state_id`, or lazy: ancestor + actions);
  - observation, `StateEmbedding` and cell;
  - parent;
  - the segment that reached it (see below);
  - statistics: times expanded, expansion yield, VLM prior;
  - flags: unrestorable, challenge page.
- **Expansion:**
  1. restore the selected node;
  2. run the exploration policy for K steps (K a flag);
  3. create children at the segment end and at intermediate steps whose novelty passes a
     threshold;
  4. propagate the **yield** (e.g. new cells found, summed novelty) up the path to the root.
- **Every observation and action is stored.** Every expansion's full segment
  (obs, action, info per step, including steps that did not become nodes) is kept with the
  node it started from and passed up into the tree's log. Nothing seen during exploration is
  dropped. Downstream stages can then use whichever unit they need:
  - single transitions (OS-Genesis, SynthAgent);
  - a state with neighbours (OpenMobile, GSAR);
  - whole root→node trajectories (AutoPlay);
  - clusters of novel trajectories (curiosity's INFER/REFINE/DISTILL).
- Any node can also be a **start state for practice**, since it can be restored.

#### C. Selection: energy-based sampling over all nodes

Each iteration:
1. Collect **all** existing nodes.
2. Compute an energy for each:
   `E(n) = Q(n) + c * P(n) * sqrt(N_total) / (1 + N(n))`
   - `Q(n)`: mean expansion yield from n;
   - `N(n)`: visit count of n's cell (or cluster, per the open question);
   - `N_total`: total visits;
   - `P(n)`: the VLM prior;
   - `c`: exploration constant.
3. Normalise to a distribution, `p(n) ∝ exp(E(n) / τ)`. τ → 0 is greedy PUCT; large τ is
   uniform.
4. **Sample** the node to expand from p.

Root-only selection (the restart special case) is a flag.

**VLM prior (set up now, minimal).**
- When a node is created, a VLM sees its observation (frame + flattened text) and
  `env.env_description`, and answers how promising it is to explore from here (a 0–10 score
  with one line of reasoning).
- The score is normalised to `P(n) ∈ [0, 1]`.
- One call per node, never per selection.
- No tree summary in the prompt for now.

#### D. Exploration policies (expanders)

All act through the env's text actions (`env.step(text)`).
- **Random (env-specific).**
  - `env.sample_action()`, which all three envs already have.
  - Optional cheap upgrade: re-sample when the next observation is near-identical by the
    shared similarity (a no-op filter without domain state).
- **VLM explorer (one policy for all envs, minimal now).**
  - Prompt: `env.env_description` + `env.actions_text` + the current observation + the last
    few actions, with a goal-free instruction ("explore: do something you have not done yet;
    no task is given").
  - Parsed by the env's own action parsing.
  - To be replaced by the standardised agent when it lands.
- **Curiosity PPO: ON HOLD.** Not wired in until the curiosity work in `plans/curiosity_plan.md` is
  settled, so the two tracks don't overwrite each other. The shared part (scoring novelty)
  is already common through `cusi.state`. What would remain is search-side only: starting PPO
  episodes from selected nodes, and feeding it the search's archive.

#### Existing methods as configurations

| Method | Expander | K | Selection |
|---|---|---|---|
| OS-Genesis / OpenMobile | random | 10 | roots only |
| AutoPlay | VLM explorer (+ memory) | long | roots only |
| GameBoyRL curiosity | PPO (on hold) | 30 | roots only |
| SynthAgent | VLM explorer | 1 | count-weighted over all nodes |
| Go-Explore | random | any | cell counts |

#### Order of work
1. Saved states in the three envs: **done**.
2. `cusi/state/` (curiosity_plan §3.2, built in that plan's steps 1–2, first the
   behaviour-preserving migration): encoders, similarity, archive with cells, scorers. The
   search needs at least `random_patch` + `tfidf`/`overlap` + the `embedding` scorer to start.
3. Tree + log format + energy-sampling selection, with the random expander (cheapest
   end-to-end test, GameBoy first).
4. VLM explorer + VLM prior.
5. Lazy nodes and a snapshot disk budget for Android; trimming web load time.

#### Not covered by this plan yet (needed before a search can be run and inspected)

Noted here, not properly designed yet; each needs its own decision. For the first build,
items 1–5, 7 and 8 get **throwaway toy versions** only to check that the pieces run together
(§1.5). The real designs are still open.

**Running a search**
1. **Entry point and job pair.** A `run_preexplore.py` (env, expander, budget, `cusi.state`
   flags) and a `slurm/` + `scripts/slurm/` pair. When the VLM explorer or prior is used, it
   starts vLLM via `scripts/serve_vllm.sh`, and `--model_name` is required (as
   `practice_small`).
2. **Stopping rule and budgets.** Total expansions, steps, wall time, and a disk budget for
   saved states. Nothing above says when a search ends.
3. **Envs per scene.** An env instance is fixed to one scene, and saved-state ids only work on
   the env that made them. The tree must record which env instance holds each node's state
   (one env per root scene, as practice's `EnvPool`). On Android all scenes share one
   emulator: snapshots work across scenes, scene objects don't.
4. **Exact definitions of the statistics.**
   - `N(n)`: every observed step whose state falls in n's cell, or only expansions from n?
   - `Q(n)` for a never-expanded node: prior only?
   - How yield propagates up the path: sum, mean, discounted?

   The energy is undefined until these are fixed.
5. **Threshold calibration.** The cell similarity threshold and the "step becomes a node"
   novelty threshold depend on the embedder and the env. They need a calibration pass, e.g.
   score distributions from a random run (curiosity's `debug_curiosity.py random`), or the
   tree ends up with one node or thousands.
6. **Parallelism.** Sequential, one env per scene, is assumed. Several workers expanding at
   once (Web: 2; GameBoy: many; Android: 1 per emulator) would need concurrent selection
   (e.g. virtual visits).

**Inspecting a search**

7. **On-disk tree format.**
   - nodes: stats, parent, cell, VLM prior;
   - segments: compressed frames, texts, actions, info;
   - a selection log: which node was picked, with its energy and probability, per iteration.

   Ideally resumable.
8. **Viewer.** For example, a static HTML page per run:
   - the tree with a thumbnail per node;
   - node stats and the VLM prior's score and reasoning;
   - click a node to step through the segment that reached it;
   - the energy/probability distribution over time.

**Env-specific gaps**

9. **Android:** lazy nodes + a snapshot disk budget (order of work step 5) are required
   before any run with more than ~20 nodes (3.6 GB per snapshot).
10. **Web:**
    - the challenge-page detector (`plans/web_voyager_plan.md` §1), or the tree fills with
      Cloudflare/captcha pages that score as highly novel;
    - the unlabelled screenshot (`plans/curiosity_plan.md` §3.1), or set-of-mark labels add fake
      novelty;
    - restore time (~17 s).
11. **Durable states.** Saved states die with the process: `close()` deletes them, and
    Android snapshots live in `/tmp`. Fine for watching a run. Using nodes afterwards (e.g. as
    practice start states) needs a "keep states" option and a way to rebuild a node, e.g. by
    replaying its root→node path.

**Scheduling dependency**

12. The search needs `cusi.state`, which the curiosity plan builds after its migration step.
    To avoid waiting on the curiosity track, the first `cusi.state` pieces to build are the
    ones the search needs: `random_patch`, `overlap`/`tfidf`, the archive with cells, and the
    `embedding` scorer.

With these added, the first run would be GameBoy + random expander + `random_patch` + the
viewer: cheap and exact, to check the tree growth and selection behaviour before adding the
VLM, Android or Web.

### 1.5 First build: decisions and toy scaffolding (2026-10-04)

*(The throwaway toy scaffolding that was here was deleted on 2026-10-06.)*
