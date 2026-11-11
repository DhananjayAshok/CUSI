# Reproduction spec for 9 self-improving GUI-agent methods (PAE, OpenWebVoyager, WebEvolver, AppVLM, MobileRL, UI-Voyager, TTI, OS-Themis, MobileForge)

## What was read
- **Papers.** All 9 papers, read in full including appendices, from arXiv HTML. Every paper's identity matched the requested arXiv ID.
- **Official code.** Read the training, collection and config files, not just READMEs. I read PAE and verified one TTI claim myself; the other methods were read by parallel sub-agents and their findings are merged here.
- **Locations.**
  - Scratch root: SCR = /tmp/claude-8561/-nas-eclairnas01-users-ashokd-projects-CUSI/e2e65eef-1ffb-40d4-a120-b91e3f4724c7/scratchpad
  - Papers: SCR/papers/<id>.c.txt
  - Repos: SCR/repos/<Repo>/
- **Repos found:**
  - amazon-science/PAE
  - MinorJerry/OpenWebVoyager
  - Tencent/SelfEvolvingAgent (the WebEvolver/ subfolder is WebEvolver)
  - THUDM/MobileRL
  - ui-voyager/UI-Voyager
  - test-time-interaction/TTI
  - OS-Copilot/OS-Themis
  - kwai/MobileForge
  - **AppVLM has no code.** GitHub and HF were searched. The only "appvlm" repo, yuanjielu-64/appvlm, is an unrelated robotics project.
- **Nothing in the CUSI working directory was modified.**

**Markers used below:** UNVERIFIED = I could not confirm it. FLAG = the paper and the code disagree.

---

# 1. PAE: Proposer-Agent-Evaluator (arXiv 2412.13194; github.com/amazon-science/PAE)

## A. Algorithm
Paper Alg. 1 (App. A). Code: pae/algorithms/onpolicy_train_loop.py, pae/algorithms/utils.py, pae/algorithms/filteredbc/trainer.py.

```
Before training: the proposer generates a FIXED task pool, 100k tasks for WebVoyager (WV) / 30k for WebArena (WA). (Table 6)
policy <- SFT checkpoint            # LLaVA-1.6-7B/34B SFT'd on 11,220 Claude-3-Sonnet trajectories (App. E) -> REMOVE for us
D <- [] (train_trajectories, cumulative; capacity 500000 = effectively unlimited)   onpolicy_train_loop.py:57-60, 172, 190
for global iteration in 1..epochs:                                        # code epochs=200 (rl_llava_7b.yaml:20)
    collect rollout_size trajectories (code 256 per host; paper "4096" -- see D)
       each env: task = random.choice(tasks)   (webgym.py:552, random_task=True), uniform with replacement
       reset: fresh browser at task["web"] = site homepage                  # online reset per attempt
       up to max_iter=10 steps; ends on ANSWER or timeout
       reward: if last action is ANSWER -> Claude/Qwen judge (0/1); else 0 with no judge call   (utils_eval.py:487-492)
    D += all trajectories (successes and failures stored)
    for actor_epochs (code 1, paper 4):
        F = filter_trajectories(D): keep reward>0; group by task; if a task has >1 success,
            keep only trajectories with len <= 20th percentile of that task's success lengths   (utils.py:52-72, top_p=0.2)
        sample actor_trajectories=1024 trajectories from F without replacement (trainer.py:76), flatten to steps, 1 pass
        loss = -mean over samples of [ mean over tokens of log π(Thought+Action | prompt) ]   (trainer.py:37; interface.py:133-159)
    checkpoint every save_freq=3; eval every eval_freq=3 on the test set
```

- **Objective:** Filtered BC, i.e. plain NLL on the thoughts and actions of successful trajectories.
  - No advantage, no KL, no negatives.
  - Token log-probs are divided by temperature (1.0), and the per-sequence mean includes padded positions in the denominator.
  - `max_grad_norm` is passed, but the clip call is commented out (trainer.py:84-88). Grad clipping is effectively off. FLAG against Table 6's "0.01".
- **Iteration:** one collection followed by one update. The buffer is cumulative. The model continues from the previous iterate (LoRA weights carry over).
- **Stopping:** a fixed number of iterations. Table 1 reports "a running average of the evaluation results at each iteration". There is no checkpoint selection.

## B. Task proposal and exploration
- **Proposer:** Claude-3-Sonnet (main) or Qwen2VL-7B/72B (ablation). It runs once, offline, before RL.
- **Inputs:**
  - WebVoyager: the website name only, plus ONE fixed example task (Apple–40) as format. The same prompt is used for all 13 WebVoyager sites (App. C).
  - Variants:
    - WA-Map adds 3 hand-written in-domain examples "to set the right difficulty" (Fig. 15).
    - WA Reddit/OneStopMarket add screenshots (Fig. 16): "answer how many screenshots are provided and describe in detail the functions…", "start from the home page", "VERY BASIC functions… within 5 steps", "do not propose modifications".
    - "PAE (User Demos)" appends 10 screenshots per site (3 per the App. C text, FLAG). The screenshots come from author-collected demos of 10 tasks per site (App. B).
- **Prompt:** scripts/propose_tasks_from_names_webvoyager.py:101-116 and paper Fig. 14. Summary:
  - "…please give 25 more example tasks… in the domain of {web_name}. You should imagine tasks that are likely proposed by a most likely user… output thoughts, then 'Output:' then JSONL"
  - Rules:
    - diverse, covering a wide range of difficulty;
    - objective and unambiguous;
    - no dependence on current time or location;
    - "by looking at the last three screenshots and the answer… it should be possible to tell… whether the task was completed";
    - 3–7 minimum steps;
    - humans should succeed 100%;
    - no sign-in.
- **Sampling:** temperature 1.0, top_p 0.9, max_tokens 2048, 25 tasks per call.
- **No dedup or filtering.** Lines that fail to parse are dropped.
- **Code bugs:** in the script, "Coursera" maps to espn.com, and "EPSN" ≠ "ESPN", so those tasks are silently dropped (:18-34).
- **No curriculum.** Tasks are sampled uniformly.
- **Start state:** always the site homepage via online env reset. Exploration is just the task attempts; there is no separate exploration phase.
- **Particular?** Moderately.
  - Proposals are made blind, from the site name, before any interaction. Every attempt resets to the landing page.
  - With pre-collected trees, a faithful PAE ignores the trees for proposal and resets per task.
  - The paper's own "User Demos" variant, screenshots as context, is the faithful hook for the trees. Feed sampled tree screenshots per site/app/game as the "user demos" context, using the Fig. 16 style prompt. Tasks must still start from env reset, matching Fig. 16 rule 4: "start from the home page… instead of the shown screenshots".
  - A mid-tree start state is a deviation.

## C. Judge
- **Inputs:** task, the ANSWER text, and the last 3 screenshots of the trajectory (Table 6 "seen screenshots 3"; ablated {1,3}).
  - Code: utils_eval.py:469-566. The user message is `TASK:… Result Response:… <num> screenshots at the end:` + images + "Your verdict:".
  - Settings: system = evaluator prompt, temp 0, top_p 0.7.
- **Output:** free-text reasoning, then SUCCESS / NOT SUCCESS. Reward = 0 if "NOT SUCCESS" appears. The reward is binary at trajectory level.
- **Prompt (Fig. 13):** WebVoyager auto-eval prompt plus:
  - "YOU SHOULD EXPECT THAT THERE IS A HIGH CHANCE THAT THE AGENT WILL MAKE UP AN ANSWER NOT VERIFIED BY THE SCREENSHOT";
  - "completes EXACTLY what the task asks".
- **Trajectories without ANSWER get reward 0.** This means every success must end with ANSWER, even for operational tasks.
- **Note:** the shipped scripts/prompts/evaluator_prompt.txt is the OpenStreetMap-specific variant.
- **Ablations (Fig. 3):**
  - A step-based judge (BC on steps judged correct) does worse than SFT.
  - A function-based judge (LLM writes verifiers) does worse because it hallucinates criteria.
- **Human alignment (Fig. 4):** 1.7% system-level and 8.6% instance-level misalignment.

## D. Hyperparameters
| Item | Paper (Table 6) | Code |
|---|---|---|
| Policy | LLaVA-1.6 Mistral-7B, Yi-34B | `liuhaotian/llava-v1.6-mistral-7b`; 34B 4-bit (README) |
| FT | not stated | **LoRA r=256, α=256, dropout 0.05, all linear layers of the LM, vision frozen** (llava_agent.py:56-63; `use_lora: True`, `train_vision: False`) |
| lr | 2e-5 | 2e-5 (default.yaml), Adam (trainer.py:26) |
| Batch / grad accumulation | 8 / 32 | per-GPU batch 1 (onpolicy_train_loop.py:76) × 8 GPUs (DeepSpeed ZeRO-2 + CPU offload, bf16; accelerate config_zero2.yaml) × grad_accum 32 = 256 steps per optimizer step |
| Rollouts per iteration | 4096 (WV), 2048 (WA) | `rollout_size: 256` single host (rl_llava_7b.yaml:22); 512 per worker in distributed mode (worker_llava.yaml). 4096 is presumably the sum over host + workers. FLAG/UNVERIFIED |
| Actor epochs per iteration | 4 (WV), 2 (WA) | 1 (rl_llava_7b), 2 (distributed). FLAG |
| Trajectories per update | — | 1024 sampled from the filtered buffer |
| Rollout temperature | 1.0 | 1.0, max_new_tokens 256 |
| Max steps | 10 | `max_iter: 10` |
| Proposed tasks | 100k (WV), 30k (WA) | — |
| Grad norm | 0.01 | not applied (commented out). FLAG |
| Iterations | not stated | `epochs: 200` (single host) / 20 (distributed) |

## E. Tricks
1. **Length filter:** keep only the shortest 20% of successes per task. This is a PAE trick that TTI explicitly removed.
2. **Chain-of-thought.** "Thought:… Action:…" is trained on. Removing reasoning hurts generalisation to held-out human tasks (Fig. 3).
3. **Set-of-mark actions.** Click [n], Type [n]; text (auto-Enter), Scroll [n|WINDOW]; up/down, Wait, GoBack, Google, ANSWER; text.
4. **Policy input** (llava_utils/interface.py:40-67):
   - ONE current screenshot (336px LLaVA, no anyres);
   - task ≤512 chars;
   - element text list ("buttons") ≤512 chars;
   - text history of the last 3 Thought/Action strings, ≤512 chars.
5. **Judge sees** the last 3 screenshots plus the answer.

## F. Results
- **Table 1, WebVoyager (557 tasks, 13 sites, Claude-3-Sonnet judged):**

| Model | SFT | PAE |
|---|---|---|
| LLaVA-7B | 14.9 | 22.3 |
| LLaVA-34B | 22.2 | 33.0 |

  - LLaVA-7B PAE with a Qwen2VL-7B proposer and judge: 21.7.
  - Comparisons: Qwen2VL-72B 22.6; Claude-3.5-Sonnet 50.5.
- **Table 2, WebArena-Easy:**
  - LLaVA-7B: SFT 18.0 → PAE 24.6.
  - With Qwen72B proposer/judge: 26.0. With Qwen7B: 23.1. With User Demos: 25.7.
- **Table 3, unseen websites (85 sites, 500 tasks):**
  - 7B: 9.1 → 16.3.
  - 34B: 16.1 → 21.4.
- **Ablations:**
  - Fig. 3: reasoning on/off; outcome vs step vs function judge.
  - Fig. 6: name-only vs user-demo context. Demos matter for OpenStreetMap: many name-only tasks there are infeasible.

## G. Released vs missing
- **Released:**
  - Code: env, filtered BC, proposer script, prompts.
  - HF gated dataset `yifeizhou/pae-data`:
    - checkpoints for 7B/34B SFT and PAE;
    - tasks `webvoyager_pae_byqwen7b.jsonl`, `webarena_pae_byqwen7b.jsonl`, `webarena_easy.jsonl`, test sets;
    - example WA data.
  - Models `yifeizhou/pae-llava-7b`, `-34b`, `-7b-webarena`.
- **Missing:**
  - The Claude-proposed task file referenced in the config (`webvoyager_pae.jsonl`). Only the Qwen-7B-proposed pool is released, and it is gated, so I could not inspect it.
  - SFT data.
  - The WA-Easy rewritten-task generation code and evaluator scripts. Partially released.
- **To guess:** which of the code vs paper epochs and rollout sizes produced Table 1.

## H. Feasibility for us
- **Cheap:** one rollout per task, offline NLL, LoRA (PAE itself used LoRA r=256), alternating vLLM serving and FSDP training.
- **Scale:** the paper's scale (4096 rollouts per iteration over many iterations) is out of reach. Use 256–512 rollouts per iteration for 10–20 iterations.
- **Risk without the SFT warm-up:** PAE needed SFT because LLaVA was at ~0%. Gemma-31B zero-shot is likely non-trivial, so dropping SFT is acceptable but must be declared.

---

# 2. OpenWebVoyager (arXiv 2410.19609; github.com/MinorJerry/OpenWebVoyager)

## A. Algorithm
Paper Alg. 1 (App. B); code WebVoyager/exploration_feedback.py:337-574.

```
Phase 0: all queries synthesized up front (GPT-4o self-instruct): Q_IL (1516), Q_SI^1..3 (480 each = 48 sites × 10)
Phase 1 IL [REMOVE: GPT-4o teacher]: π_b = SFT(idefics2-8b, 1165 finished GPT-4o trajectories)
         successful AND unsuccessful ones kept; only unfinished ones dropped
for j in 1..3:
   D_j = {}
   for q in Q_SI^j:
      for attempt in 1..5:  reset fresh Chrome at q.web (homepage)
                            τ = rollout(π_{j-1}, T=1.2, ≤15 steps)
                            r = GPT-4o judge(task, ANSWER, last 3 screenshots)
                            if r: D_j += τ; break        # ≤1 kept trajectory per query, early stop
   π_j = SFT(init=π_{j-1}, D_IL ∪ D_j)                   # NOT cumulative over earlier D's (Table 6 ablation)
```

- **Iteration:** 300 optimizer steps × global batch 64 (≈2–3 epochs). The LR schedule is fresh each round and starts from the previous checkpoint. There is no checkpoint selection.
- **Loss:** token NLL on thought+action (Eq. 6/7).
  - FLAG: the Megatron script does not pass `--answer-loss-only` (pretrain_megatron_idefics2.py:95-110), so the loss as released also covers user and accessibility-tree tokens.
  - Even with the flag, every earlier assistant turn in the sample is trained.

## B. Task proposal and exploration
- **Method:** self-instruct with GPT-4o from the website name/URL plus existing seed queries (Mind2Web train, WebVoyager) as in-context examples, deduplicated with all-mpnet-base-v2. The threshold is UNVERIFIED.
- **Pre-generated:** query IDs run continuously across rounds, so all rounds' queries were generated in one batch and split into thirds. Nothing is adaptive.
- **Released:** only the resulting files (WebVoyager/data_for_training/), not the generator code or prompt.
- **Start state:** homepage reset for every attempt. Exploration and attempts are the same thing (T=1.2).
- **Leakage:** 8 IL queries are textually identical to WebVoyager test queries (found by the agent's script).
- **Particular?**
  - Proposals come from site name plus seeds, not from observed states. There is a landing-page reset per attempt, and queries are fixed in advance.
  - The method has no slot for exploration data. A faithful run ignores our trees; using them is a separately labelled variant.

## C. Judge
- **Code:** WebVoyager/evaluation/auto_eval.py; GPT-4o, T=0.
- **Inputs:** task, ANSWER text, and the last 3 screenshots during exploration (15 at test time, FLAG).
- **Output:** SUCCESS / NOT SUCCESS. A trajectory with no ANSWER scores 0.
- **Not shown to the judge:** thoughts, actions, accessibility tree.

## D. Hyperparameters
Source is code unless noted.

| Item | Value |
|---|---|
| Model | idefics2-8b, full FT (all modules) |
| LR | 1e-5 → 1e-6 cosine, warmup 9 (not in paper) |
| Optimizer | Adam, weight decay 0, clip 1.0 |
| Batch | global 64 |
| Steps | 300 per phase |
| Sequence length | 8192; ≤3 images at 980px |
| Parallelism | TP=8 on 8×A100 |
| Rounds | 3 |
| Queries per round | 480 |
| Attempts per query | ≤5 |
| Rollout sampling | T=1.2, max_new_tokens 256; T is hardcoded in the server, so eval also ran at 1.2 (UNVERIFIED) |
| Max steps | 15 |

- **Observation:** accessibility tree (current only) plus an **unmarked** 1024×768 screenshot (no set-of-mark), last 3 screenshots, full thought/action history.
- **Data volumes (Table 3, successes / turns):**
  - iter1: 152 / 943
  - iter2: 205 / 1324
  - iter3: 207 / 1333

## E. Tricks
1. ≤5 attempts per query with early stop, so ≤1 trajectory per query. This biases toward short trajectories.
2. D_IL is re-mixed every round as an anchor; self-generated data is non-cumulative.
3. IL keeps unsuccessful but finished GPT-4o trajectories.
4. Restart action (→ google.com). Its use grows across rounds (Table 5).
5. Exploration temperature 1.2.
6. Difficulty-guided sampling (dgs): more samples for sites with S@5 < 40%. Details UNVERIFIED.

## F. Results
- **Table 1, WebVoyager:**

| Stage | Success |
|---|---|
| IL | 19.9 |
| iter-1 | 22.6 |
| iter-2 | 22.7 |
| iter-3 | 25.8 |
| dgs | 25.5 |
| dgs + GPT-4o data | 27.4 |

- **Table 2, Mind2Web:**

| Split | IL | iter-3 |
|---|---|---|
| Cross-task | 6.3 | 19.6 |
| Cross-web | 6.6 | 10.4 |

- **Table 3, exploration stats:**

| Round | F@1 | S@1 | S@5 |
|---|---|---|---|
| iter-1 | 74.6 | 10.4 | 31.7 |
| iter-3 | 91.5 | 18.8 | 43.1 |

  Answers are often hallucinated.
- **Table 6 (the only true ablation):** cumulative self-generated data 20.8 vs non-cumulative 23.3 (120 queries).

## G. Released vs missing
- **Released:** environment, loop, judge, all query files, conversion scripts, Megatron training, HF checkpoints IL/opt-1/2/3.
- **Missing:** self-instruct code and prompt, the collected trajectories (despite the README), dgs details, amazon_cookies.pkl, the answer-loss flag actually used, eval temperature.

## H. Feasibility
- **Cheap.** Per round: 480 queries × ~3.5–4 attempts ≈ 1.8k trajectories ≈ 20–28k policy calls.
- **Steps:** without D_IL each round is only ~1–1.3k step samples. Scale to 2–3 epochs, not 300 steps.

---

# 3. WebEvolver (arXiv 2504.21024; code: github.com/Tencent/SelfEvolvingAgent/WebEvolver)
The README confirms this repo is WebEvolver. It is Cognitive Kernel v2 plus data scripts. **No training code is released.**

## A. Algorithm

### Arm 1: plain self-improvement
This is OpenWebVoyager's loop without GPT-4o IL. The judge is the policy's own base model.
```
Q = OWV's 1,516 IL queries; M0 = Llama-3.3-70B zero-shot
for i in 1..2:  for q in Q: 1 rollout (T=0 default, base_model_connection.py:123), ≤16 steps (MAX_STEPS; code default 12, FLAG)
                S_i = {τ: zero-shot Llama-3.3-70B judge says SUCCESS}
                D_i = D_{i-1} ∪ S_i                                 # cumulative (§3.2)
                M_i = SFT(Llama-3.3-70B?, D_i)                      # restart from base vs continue: UNVERIFIED
```
- **FLAG:** the released converter (data/get_sft_data_from_trajectory.py:567-569) applies no judge. It keeps any trajectory ending in a non-N/A `stop [answer]`.
- **Iterations:** iter-1 is used downstream because it performed better.

### Arm 2: co-evolving world model (WM) and synthetic trajectories
- **WM training data (iteration i):** every (o_{t-1}, a_{t-1}, o_t) transition from that iteration's trajectories. The code uses all parsed trajectories; Eq. 8 says the rejection-sampled ones (FLAG).
- **WM context:** "Initial operation: CallWeb(...)" plus objective; earlier steps as actions with "The previous observations are omitted."; the last observation and the action.
- **WM target:** `THOUGHT: <rationale distilled from base Llama given the true next page>` followed by `OBSERVATION: <full next accessibility tree>`.
  - Rationale prompt: data/world_model_sft_data_construction.py:28-58.
- **WM model:** a separate SFT of Llama-3.3-70B.
- **Synthesis:**
  - Models: WM-iter2 with policy-iter1.
  - Runs only on queries never solved. One synthetic rollout per query, depth ≤7, T=0.7 for both policy and WM.
  - At depth 6 the WM is prompted to "generate a next webpage containing the information… directly needed for answering".
  - The code imagines o_0 from the CallWeb string; the paper says synthesis starts from a real o_0 (FLAG).
  - Synthetic trajectories are filtered by the zero-shot Llama judge.
- **Final policy:** WebEvolver = SFT(base, D_1 ∪ D_w).

### Arm 3: WMLA (world-model lookahead) at inference
Code: cognitive_kernel/call_web.py:305-642.
- **When:** active from step 2 (steps 0–1 are greedy).
- **Candidates:** k=3.
  - a1 greedy (T=0).
  - a2 and a3 sampled with "Please generate actions different from {a1}…". Temperature 1.0 in code vs 0.7 in the paper (FLAG).
  - Duplicates are removed.
- **Simulation of each candidate:** depth d ∈ {1,2,3}, best d=2. The policy acts on imagined pages at the intermediate steps.
  - `stop` is not simulated; the critic judges the answer directly.
  - `restart` uses the real first page.
- **Critic:** GPT-4o, prompt at call_web.py:159-173. Inputs: objective, real current tree, candidate thought+action, simulated observations. Output: Score ∈ {0, 0.5, 1}.
- **Decision:** execute the argmax; ties go to the greedy candidate.

## B. Task proposal and exploration
- **No proposal of its own.** It uses OpenWebVoyager's fixed 1,516 IL queries:
  - 460 general Google search;
  - 516 Mind2Web;
  - 165 Mind2Web human-extended;
  - 300 WebVoyager GPT-4o self-instructed;
  - 75 WebVoyager human-extended.
- **Start state:** reset to the target URL.
- **No exploration phase.** "Diversity" comes from WM hallucination plus synthesis on failed queries.
- **Not particular about proposal.** Our trees can supply extra WM training transitions. That is an addition and should be run as an ablation; the faithful WM uses agent trajectories only.

## C. Judges
- **Rejection judge:** zero-shot Llama-3.3-70B (self/base).
  - Inputs: task, answer, and the accessibility trees of the whole trajectory (generate_webvoyager_mind2web_new.py:66-141; a "last k trees" variant also exists, which one was used is UNVERIFIED).
  - Output: SUCCESS / NOT SUCCESS.
- **WMLA critic:** GPT-4o, scores {0, 0.5, 1}. Hard-wired to `gpt_connection` in cognitive_kernel.py:864.
- **Test evaluation:** GPT-4o.

## D. Hyperparameters
- **Models:** policy, WM and judge are all Llama-3.3-70B.
- **No training hyperparameters are given anywhere** (LoRA vs full FT, lr, epochs, batch).
- **Rollouts:** 1 per query, T=0. History is the current observation only plus all thoughts and actions (k=1).
- **Released HF data `CognitiveKernel/WebEvolver`:**
  - Policy SFT: 3,912 examples (≈670 successful trajectories plus 723 planning examples).
  - WM SFT: 9,553 examples. Median target 2.5k characters, max 17k.
- **Observation is text only** (Playwright accessibility tree, "Text-only WebVoyager").

## E. Tricks
1. SFT data filters:
   - drop no-change steps;
   - drop wait steps;
   - drop "log in" trajectories;
   - normalise action strings.
2. The planning step (CallWeb) is trained.
3. WM rationale prefix.
4. Synthesis only on failed queries.
5. Depth-6 "answer page" nudge.
6. Policy-iter1 chosen for synthesis.
7. WMLA dedup and the "different from" prompt.

## F. Results
- **Table 1 (text-only WebVoyager 473 tasks / Mind2Web-Live 53 tasks):**

| Model | WV | M2W |
|---|---|---|
| Llama zero-shot | 32.98 | 18.86 |
| self-improve iter1 | 38.68 | 15.09 |
| self-improve iter2 | 38.23 | 16.98 |
| self-improve iter3 | 38.65 | 16.98 |
| no-WM synthetic baseline | 38.98 | 18.86 |
| **WebEvolver** | **42.49** | **22.64** |
| + WebDreamer | 44.61 | 22.64 |
| + WMLA d=1 | 46.24 | 28.30 |
| + WMLA d=2 | **51.37** | 24.53 |

- **Table 2:** WM fidelity degrades past depth 2.
- **Table 3:** GAIA-web / SimpleQA.
- **Table 4, branching k at d=2:** k=2: 48.62; k=3: 51.37; k=5: 50.73.
- **Not ablated:** WM rationale on/off; GPT-4o critic vs self-critic.

## G. Released vs missing
- **Released:** code (sampling, converters, WM data construction, synthesis, WMLA) and the SFT data.
- **Missing:**
  - all training code and hyperparameters;
  - the judge script;
  - checkpoints;
  - raw trajectories;
  - per-iteration splits;
  - the D_w size and mixing ratio.
- **Several shipped scripts are broken:** swapped service arguments in traj_synthesizing_world_model.py:189-190; an undefined `ss()`; missing imports.

## H. Feasibility and modality issue
- **Training is cheap.** The policy and the WM can be two LoRA adapters on one Gemma base (vLLM multi-LoRA).
- **WMLA cost:** at d=2, about 15 calls per real step.
- **The core issue is modality.** The WM must generate **text** observations.
  - Web: set-of-mark element list or accessibility tree. Store this text alongside screenshots in our trees.
  - Android: UI hierarchy.
  - GameBoy: no tree is allowed, so the WM must use free-text screen descriptions. This is the paper's "Abstract Description" WM variant.
- **Consequences:**
  - Synthetic trajectories would train a screenshot policy on imagined text observations. That is a declared mismatch.
  - WMLA ports cleanly at d=1; d=2 is an extension.

---

# 4. AppVLM (arXiv 2502.06395): no code released
All of this is from the paper. Missing details are UNVERIFIED.

## A. Algorithm (Alg. 1, §3.3–3.4)
"Reinforce Fine-Tuning" is ReST-style rejection SFT, not REINFORCE.
```
AppVLM-base = SFT(PaliGemma-3b-pt-896, AndroidControl human demos)           # stage dropped in our setting (human data)
for i = 0..N (3 or 4; FLAG: text says 3, Table 3 shows RFT_1..RFT_4):
    D_on ∪= Collect(AndroidWorld, π_i)
       - each of 82 AW tasks queued "several times" at "high temperature"
       - re-queue tasks solved < τ=10 times; repeat several rounds
       - keep successes only
       - drop steps whose screenshot didn't change
       - oversample tasks with < τ successes
    RFT_{i+1} = SFT(RFT_i, D_on)            # continue from previous iterate; cumulative buffer
AppVLM = SFT(AppVLM-base, D_on)             # final retrain from base: +2.8 AW vs RFT_4 (Table 4)
```
- **Loss:** token NLL on the JSON action. No CoT, KL, advantage, or offline-data mixing.
- **Epochs:** 3 per stage.

## B. Task proposal
- **No proposal.** Tasks are the AndroidWorld templates themselves: 82 of 116, after removing verification, Q&A and drawing tasks (App. A.3). Parameters are randomized.
- **Start state:** AndroidWorld task init/reset.
- **Exploration:** the same thing as the attempts.
- **No stated train/test split.** The same templates are used for training and evaluation; only the parameters differ. This leakage caveat should be reported.
- **Particular?** Not about proposal; it has no slot for trees. It needs a fixed task set with a checker and resettable task init.

## C. Judge
- **AndroidWorld's rule checker, run after EVERY step.** Success terminates the episode.
- **No terminate/answer action exists.** This is why Q&A and verification tasks are dropped.
- **Consequence for (b):** we must add a terminate action plus a step cap and judge at the end. That is a declared deviation.

## D. Hyperparameters
| Item | Value |
|---|---|
| Model | PaliGemma-3B at 896², full FT |
| Optimizer | AdamW, lr 3e-6 every stage, decayed to 0 |
| Epochs | 3 |
| Effective batch | 64 |
| History | last 5 actions as text (element descriptions replace indices) |
| Screenshots | 1 per step, set-of-mark boxes from the accessibility tree |

- **Not given:** temperature, rollouts per task, max steps, D_on size, compute.

## E. Tricks
1. Drop unchanged-screen steps (removes no-ops and most waits).
2. τ=10 rebalancing in both collection and training.
3. Final SFT from base.
4. Action JSON: open-app, click(idx), long-press, input-text, scroll×4, home, back, wait. No `keyboard-enter`, no terminate.

## F. Results
- **Table 2, AndroidWorld (82 tasks):**

| Agent | Success |
|---|---|
| AppVLM-base | 11.4 |
| **AppVLM** | **37.8** (0.91 s/step) |
| T3A | 41.9 |
| M3A | 36.6 |

- **Table 3, AW across RFT iterations:** 17.9 → 23.2 → 30.5 → 35.0.
- **Table 4 ablations:**
  - No final retrain from base: 35.0.
  - No AndroidControl SFT: 22.4.
- **Table 1:** AndroidControl drops after RFT (73.9 → 69.0).

## G. Released vs missing
- **Released:** nothing.
- **To guess:** prompt, temperature, rollouts per task, max steps, oversampling rule, iteration count.

## H. Feasibility
- **Cheap:** offline SFT. About 82 × 8 × 15–20 steps ≈ 10–13k calls per collection round.

---

# 5. MobileRL / AdaGRPO (arXiv 2509.18119; github.com/THUDM/MobileRL)
**The repo has evaluation and inference code only.** THUDM/AgentRL, its trainer, has generic GRPO but no AdaGRPO components. All of AdaGRPO comes from the paper.

## A. Algorithm
- **Stage 0, reasoning-free SFT** on 97.9k steps: AndroidControl plus human annotation plus selected self-rollouts. Action only, 2 epochs, cosine 1e-5→1e-6.
- **Stage 1, reasoning SFT** on 23.6k steps. Reasoning is bootstrapped from an unnamed "off-the-shelf instruction model" and refined iteratively.
- **Both SFT stages use external, human or other-model data → drop both.**
- **Stage 2, AdaGRPO**, for about 100 RL steps:
```
sample tasks with FCF weights; each: fresh emulator reset + task init
G=16 rollouts, T=1.0, ≤50 turns; r_i ∈ {0,1} from the AW rule checker
SPA (shortest-path reward adjustment, α=1): R_i = r_i·(1 − α(T_i − T_min)/T_i) = r_i·T_min/T_i, T_min = shortest success in the group
A_i = (R_i − mean_G)/std_G, broadcast to every turn/token of τ_i
negative pruning: keep pos:neg ≤ 1:2 (drop lowest-advantage negatives)
AdaPR (adaptive positive replay): buffer |B|=256, insert top κ=0.25 by advantage; minibatch = on-policy ∪ ≤γM (γ=1) top-advantage replays
loss = token-level PPO-clip (ε=0.2), 1/T_i turn weighting, + β·KL (β=1e-3, reference = SFT init); 1 PPO epoch, lr 1e-6, clip 1.0
FCF (failure curriculum filtering): a task with all-zero rewards for 2 consecutive "epochs" gets a 3-epoch cooldown with w = exp(−f), then is removed permanently
```
- **Ambiguities (UNVERIFIED):**
  - what "epoch" means in FCF;
  - κ/γ semantics;
  - whether replayed samples keep their old advantage;
  - whether batch 256 counts prompts or trajectories.

## B. Task proposal
- **AndroidWorld:** the 116 benchmark templates with randomized parameters and initial states. Instances overlapping the test set are removed, leaving 2,000 curated train instances (App. D.4.1).
  - The test suite is fixed at seed 30 (inference/src/android_world/task.py:25-53).
  - The train instance list is not released.
- **AndroidLab:** 1,103 tasks generated by an unnamed model from app names, then manually verified.
- **Start state:** always a fresh Docker emulator plus the task's own init.
- **Exploration:** no separate phase; it is just the G rollouts.
- **Particular?** Not about proposal, but it requires online group rollouts: 16 from the same s0 per task per step, with a verifiable reward.
  - Trees can supply restorable s0 states and tasks; that is a deviation.
  - Trees cannot replace the rollouts.

## C. Reward
- **AndroidWorld:** the rule checker `is_successful`. On `finish(message)` the message is passed as the AW answer.
- **AndroidLab:** a fine-tuned GLM-4.1V-9B reward model.
  - Trained on labels from 3 proprietary VLMs (86% accuracy).
  - Inputs: task, per-step XML, actions, and one concatenated image of all screenshots.
  - Output: `<analysis>…</analysis><ans>0|1</ans>`. The prompt is quoted in App. D.3.2.
  - The reward model is not released.
- **The best 9B model used AW-only training with rule rewards.**
- **For (b):** the policy answers the D.3.2 prompt itself, untrained.

## D. Hyperparameters (Table 4)
| Item | Value |
|---|---|
| Models | Qwen2.5-VL-7B, GLM-4.1V-9B; full FT, FSDP |
| lr | 1e-6 |
| Clip | 0.2 |
| KL | 1e-3 |
| G | 16 |
| Batch | 256 |
| Rollout | T=1.0, max new tokens 4096, prompt 16k |
| Max turns | 50 |
| Images | ≤5M pixels (code default 500k pixels, FLAG; Table 6: 75.8 vs 80.2) |
| SPA | α=1 |
| AdaPR | buffer 256, γ=1, κ=0.25 |
| Steps | ~100 |
| Compute | 4 CPU hosts × 1 TB RAM, ≤256 parallel emulators, >10 min/step, 25–40 h per 100 steps (FLAG: §3.3 vs App. E) |

- **Action space:** `do(Tap|Long Press, element=[bbox])`, Type, Swipe, Launch, Back, Home, Wait, `finish(message)`.
- **Output format:** `<think>…</think><answer>`.
- **Observation:** screenshot plus compressed XML. History keeps all past assistant text plus the previous screenshot.

## E. Tricks
- SPA, negative pruning, AdaPR, FCF, full-resolution images, reasoning-free SFT before reasoning SFT.

## F. Results
- **Table 1:**

| Model | AW | AndroidLab |
|---|---|---|
| MobileRL-7B | 72.0 | 42.5 |
| MobileRL-9B | 80.2 | 53.6 |

- **Fig. 3a stages (GLM, AW):** base 7.7 → reasoning-free SFT 48.1 → reasoning SFT 66.2 → AdaGRPO 80.2.
- **Fig. 3c AdaGRPO ablation (Qwen, AW):**

| Variant | AW |
|---|---|
| Full | 71.1 |
| w/o AdaPR | 63.6 |
| w/o SPA | 69.1 |
| w/o both | 58.5 (collapses at ~30 steps) |
| w/o FCF | 64.8 |
| SFT only | 56.8 |

## G. Released vs missing
- **Released:** eval and inference code, Docker images, prompts, the `xuyifan/MobileRL-9B` checkpoint.
- **Missing:** all training code (SFT, AdaGRPO components), all training data and task lists, the reward model, the 7B checkpoint.

## H. Feasibility
- **This is the most expensive method.** One faithful step is ≈16 tasks × 16 rollouts × ~15 turns ≈ 3.8k 31B calls, at roughly 8–16 concurrent emulators. That is about 45–60 minutes per step, so 4–5 days per 100 steps.
- **Recommended scale-down:** G=8, 8–16 tasks per step, 30 turns, 50–100 steps.
- **KL reference is free with LoRA** (disable the adapter).

---

# 6. UI-Voyager (arXiv 2603.24533; github.com/ui-voyager/UI-Voyager)
- **Released:** fork-point detection and the SFT sample builder (cores/), the eval harness, the final checkpoint MarsXL/UI-Voyager, and one eval run.
- **Not released:** any training code, the seed-task generator, `run_group_sample.sh`, and all hyperparameters (the paper has no hyperparameter table).

## A. Algorithm

### Stage 1: RFT (rejection fine-tuning)
- **Base model:** Qwen3-VL-4B.
- **Round 0 data:** generated by "various scales of Qwen3-VL", i.e. stronger teachers. **Remove this.**
- **Each round:**
  1. Generate new tasks from AndroidWorld templates (perturbed parameters).
  2. Roll out the previous model at T=0.7.
  3. Keep a trajectory iff the agent terminated itself AND `task.is_successful` > 0.5 (androidworld/eval/runner.py:226-230).
  4. Run step-level SFT on response tokens.
- **Rounds:** 3; round 3 reaches Pass@1 73.2%.
- **UNVERIFIED:** whether the data is cumulative, and whether each round starts from θ0 or θ_{r-1}.

### Stage 2: GRSD (group-relative self-distillation)
```
init = RFT round-3
repeat: per task (template, seed): G rollouts from an identical init (README: 8 repeats × 3 seeds; random.seed before initialize_task)
        skip groups that are all-success or all-fail (fork_algorithm.py:352-354)
        teacher = shortest success; pair it with every failure ("shortest_base")
        M = ForkPointDetect(τ+, τ−, θ=0.95)
        for each (j, i*) in M: sample = failed step j's context (failed trajectory's text action history + its screenshot)
                               with target = the success step i*'s full response (Thought + Action + tool_call)
        SFT on these samples (token-mean NLL; response only). No GRPO / KL / advantage.
        "Mixed data" with successful-trajectory samples is mentioned (Fig. 2, README); the ratio is UNVERIFIED
```

### Fork detection
Code: cores/fork_algorithm.py:27-210.
- **Alignment:** monotone.
- **Transition advance:** skip failed step j if, for some i, SIM(o+_i, o−_j) ≥ θ AND SIM(o+_{i+1}, o−_{j+1}) ≥ θ. The first match only is taken.
- **Candidate i for step j** must satisfy all of:
  - SIM(o+_i, o−_j) ≥ θ;
  - a different parsed tool_call (code-only filter, FLAG);
  - a different next screen.
- **Choice:** i* = the candidate with maximum similarity.

### SIM
Code: cores/fork_utils.py:34-201. Pixel-based only; no accessibility tree.
1. Crop the top 25/878 of the image (status bar).
2. 16×16 mean-hash; if the hash agreement is < 0.80, reject.
3. Otherwise compute SSIM on a grayscale 480×270 thumbnail.

## B. Task proposal
- **No model proposal.** Tasks are AndroidWorld templates plus a "seed task generator" that perturbs parameters (time, quantities, files). The generator is unreleased.
- **Pool:** >7000 train tasks "following MobileRL".
- **Start state:** task init.
- **Exploration:** the same as the attempts.
- **Particular?** Not about proposal. It needs G rollouts from an identical start state plus a checker; our restorable states make that exact.
  - Matching against tree paths with tree-node identity instead of SSIM is a non-faithful extension.
- **Risks in our environments:**
  - GameBoy: identical frames can hide different state, giving false fork matches.
  - Web: dynamic content lowers SSIM, so forks are missed.

## C. Judge
- **AndroidWorld checker AND self-termination.** A run that hits the step cap is a failure.
- **No model judge.**
- **Self-judge risk in (b):** a false-positive "teacher" gets grafted onto many failed contexts, which amplifies the error.

## D. Hyperparameters
Mostly not reported.

| Item | Value | Source |
|---|---|---|
| Sampling | T=0.7, top_p 0.8 | eval config androidworld/eval/configs/UI-Voyager.yaml |
| Max steps | 10×complexity | runner.py:264 |
| History | text history of the last 30 actions; current screenshot only | yaml |
| θ | 0.95 | code |
| G | 8 | README |
| lr, epochs, LoRA vs full FT | unknown | — |

## E. Tricks
- Shortest-success teacher.
- The failed-prefix context is kept: the model learns the correct action conditioned on a wrong history.
- Thought is distilled along with the action.
- Status-bar crop.
- Same-effect filter.
- RFT warm start; GRPO from base is very sample-inefficient.

## F. Results
- **Table 2:** UI-Voyager-4B 81.0 AW Pass@1 (64-seed average) vs Qwen3-VL-4B 45.3.
  - FLAG: the released eval config runs a single combination; the paper says 64 seeds.
- **Fig. 4:** RFT 37 → 64.0 (round 1) → 73.2 (round 3). GRPO/PPO from base need ~175 steps to reach RFT round 1.
- **Fig. 8:** from the same RFT model, GRSD 81 vs GRPO/PPO ~76.
- **Fig. 9:** GRSD wins on the 10 hardest tasks.

## G. Released vs missing
- See the opening paragraph of this section.
- **To guess:** all training hyperparameters, GRSD iteration count, mixing ratio.

## H. Feasibility
- **Updates are SFT only, so cheap.** Rollouts dominate: ≈8×150×15 ≈ 18k calls per GRSD iteration.
- **Fork detection is CPU-only.**

---

# 7. TTI: Test-Time Interaction (arXiv 2506.07976; github.com/test-time-interaction/TTI)
TTI is a fork of PAE.

## A. Algorithm (code: scripts/webvoyager_train.sh:21-41 + PAE-style loop)
```
for it in 1..N (WV: 15 in code vs 12 in paper App. E.3, FLAG; WA: 10):
   h = 10 (it=1), 20 (it=2,3), 30 (it≥4)       # train horizon; eval h=30
     # paper Eq. 5.3 says h = h_min·i (10, 20, 30); the code and App. D.6/E.3 use 10, 20, 20, 30, …
   512 rollouts: task ~ uniform with replacement from the pool; reset to site homepage; 1 rollout per task; T=1.0, top_p 0.95
   reward: ANSWER-terminated only -> judge; timeouts get 0
   buffer += all rollouts (cumulative)
   F = successes, minus those whose final action is "ANSWER [N/A]"; NO length filter (top_p=1, vs PAE 0.2)
   sample ≤512 trajectories without replacement, weights 0.5·recency + 0.5·inverse site frequency (filteredbc/trainer.py:533-580)
   1 epoch per-step token-mean NLL on the full "Thought…Action" response; save; continue (LR schedule continues)
```

## B. Task proposal
- **Paper (§6, App. E.1):** 128K WebVoyager tasks and 11K WebArena tasks, "generated by the base model".
  - Generated by PAE-style seed prompting and also by an exploratory agent that explores live and emits `Tasks: GENERATE [task;answer;;…]` alongside Click/Type/Scroll actions.
  - Proposer inputs: accessibility tree, set-of-mark screenshot, seed examples, URL.
  - Rules: realistic, 3–20 steps, self-contained, not dependent on time or location, checkable from the last 3 screenshots plus the answer.
- **Code:**
  - **No proposer code.** Only `tasks/webvoyager_train_data.jsonl` is released: 128,072 lines, 104,672 unique, 13 sites, heavily skewed, no reference answers.
  - Its size (24.1 MB) is nearly identical to PAE's Qwen7B task file (24.0 MB), suggesting reuse. UNVERIFIED, because the PAE file is gated.
  - **The WebArena training file is 733 REAL WebArena tasks with ground-truth evaluators, and all 733 are in the 812-task test file.** The 11K synthetic WebArena tasks are not released.
- **Start state:** homepage reset. Exploration for proposal is separate from attempts.
- **Particular?** Mildly.
  - The E.1 explore-and-propose prompt is a natural fit for proposing from our tree states (an offline explorer instead of a live one, declared).
  - Attempts must start from env reset.

## C. Judge
- **Code (verified myself):** for WebVoyager the judge is **the policy being trained**, via `agent.get_action(batch_eval_msg)` in tti/environment/env_utils.py:259, with the same sampling settings (T=1.0).
  - The paper (§6) says Gemma-3-27B (FLAG).
  - Inputs: evaluator prompt, task, optional reference answer, final URL, final accessibility tree, answer, last 3 screenshots. No action history (the paper says action history is included, FLAG).
  - Output: SUCCESS / NOT SUCCESS. Called only on ANSWER-terminated trajectories.
- **WebArena:** the ground-truth evaluators, with GPT-4 fuzzy match (contradicts §6.3, which says a prompted verifier was used, FLAG).
- **Consequence:** our condition (b) is literally what TTI's released WebVoyager code does.

## D. Hyperparameters
| Item | Value |
|---|---|
| Model | Gemma-3-12B-it, full FT (vision tower frozen; the LoRA flag is dead code) |
| Engine | DeepSpeed ZeRO-3 with CPU offload |
| Optimizer | AdamW, weight decay 0.01 |
| lr | 4e-6 (WV), 1e-6 (WA) |
| Schedule | WarmupCosine, 2000 total steps, warmup 20 |
| Grad clip | 0.01 |
| Micro-batch | 4/GPU × GA 2 × 4 GPUs = 32 steps (paper says 6 GPUs → 48, FLAG) |
| Rollout | 512 per iteration; T=1.0, top_p 0.95, max_new 1024, max len 16k |
| History | 3 screenshots (paper says 4, FLAG); all past assistant text kept |
| check-again | `min_try=1`, i.e. off during training |
| Compute | vLLM TP=4 then train on the same GPUs; README "≥4×H100" |

## E. Tricks
1. Horizon curriculum. Fixed h=30 collapses: 45.2 vs 64.8.
2. No length filter.
3. Recency- and site-balanced replay.
4. Drop N/A answers.
5. Structured CoT: observation summary, plan, ≥3 possible actions.
6. Bing action as an escape.
7. Small lr and clip.
8. WebArena per-domain human hints (prior knowledge).
9. Claude handles PDFs (proprietary component; drop it).

## F. Results
- **Table 3, WebVoyager (427 tasks):**

| Variant | Success |
|---|---|
| Gemma-3-12B base | 55.8 |
| fixed h=10 | 59.1 |
| fixed h=30 | 45.2 |
| **TTI** | **64.8** |

- **Table 4, WebArena (812 tasks):**

| Variant | Success |
|---|---|
| zero-shot | 18.3 |
| h=10 | 23.8 |
| h=30 | 19.0 |
| **TTI** | **26.1** |

- **Table 2, schedule comparison (WebArena subset):** additive 29.5 vs multiplicative 32.25.
- **Table 6, check-again at inference:** 23.81 → 26.14.

## G. Released vs missing
- **Released:** full training and eval code, the 128K pool, the WebArena real-task files, checkpoints `sjunhongs/tti_webvoyager` and `sjunhongs/tti_webarena`.
- **Missing:** proposer and explorer code, the 11K synthetic WebArena tasks, the 27B judge setup.

## H. Feasibility
- **Cheap per iteration:** one rollout per task, no reference model. Roughly 5–15k calls per iteration.

---

# 8. OS-Themis (arXiv 2603.19191; github.com/OS-Copilot/OS-Themis)
- **What it is:** a trajectory critic. Only the critic is released; the RL code is not.
- **Pipeline** (trajectory_reward/completion_*.py, prompts in PROMPT/):
  1. **Selector** (text only: goal plus the policy's own think/operation/action history) picks milestone steps with assessment goals.
  2. **Verifier** checks each milestone using before/after screenshots. Output: success / failure / uncertain plus feedback. It loops with the Selector for ≤6 rounds.
  3. **Reviewer** (text only) lists issues. The Selector may request re-verification.
  4. **Judge** (text only, over the whole log) outputs `final_decision`. Reward = 1 iff "completed".
  - All four roles use one model at T=0.
  - **Code vs paper (FLAG):** in code the Reviewer never sees the verdicts, and review is effectively one round.
  - **Only the Verifier sees images.** The policy must emit reasoning plus an action description.
- **RL (paper only):**
  - Multi-turn GRPO in Verl: n=4, T=1.0, ≤50 steps, 20-step update window, format penalty −1.
  - No KL in the main runs (0.005 in the scaling run).
  - lr 1e-6, 4 episodes over 96 tasks.
  - Tasks are synthesized by Qwen3-VL-235B templates (a stronger teacher, so a deviation for us).
  - Start state: env reset.
- **Results:**
  - Table 2, AndroidWorld: Qwen3-VL-8B base 47.6 → 53.4 with an 8B critic (= frozen initial policy) → 54.7 with a 235B critic. 4B: 45.3 → 50.9 / 51.3.
  - Table 1, OGRBench: 88.0 accuracy with 235B.
  - Table 3 ablations: without the Judge, recall falls to 5%.
- **Self-judging:** the "8B critic for an 8B policy" setting is a frozen copy of the initial policy acting as critic. So (b) with a frozen initial Gemma is faithful to a tested configuration. Using the currently updating policy as critic is untested.
- **Under (a)** OS-Themis disappears and the method reduces to plain GRPO with oracle reward.
- **Cost:** ~165k prompt tokens and 14 calls per trajectory (Table 12). Per 384-trajectory episode on our hardware, roughly 3.5–6 hours of critic prefill (UNVERIFIED estimate).
- **Not particular about tasks.**

# 9. MobileForge (arXiv 2606.19930; github.com/kwai/MobileForge)
- **What it is:** offline, single pass, not online RL.

## Exploration
Code: explore/exploration_and_mining.py.
- Per app: an activity list from the APK, and an injected deterministic app data snapshot.
- A goal generator works from the screenshot, app name and activities ("tasks that start from the current screen… 10–30 steps").
- A GUI-explorer agent (M3A-style with RAG; the API model is unnamed, defaulting to gpt-4o) pursues each goal for ≤30 steps.
- DFS with branching factor 3 and depth ≤5. A parent state is restored by relaunching the app and replaying the action prefix.
- Output: 520 trajectories, 5,839 steps, 20 apps.

## Curriculum
Code: explore/curriculum_generator_refactored_add_aw_few-shot/.
- Per exploration trajectory, ONE VLM call. Inputs: app name, the exploration goal, ≤15 action-annotated screenshots, few-shot examples, 14 principles, and all tasks already generated for that app (for dedup).
- The call evaluates the source trajectory and generates 1–10 tasks, each needing ≥8 steps and "screenshot-grounded".
- Result: 3,249 tasks.
- **FLAG / contamination:** the code's few-shot examples are ALL 116 AndroidWorld *benchmark* tasks for that app. The principles say "MUST comprehensively cover all [AW] examination points" and "directly replicate these tasks". The paper's printed prompt omits this.
- No feasibility filter.

## Rollout
Code: rollout/run.py.
- Per task, K=4 **serial** attempts. Each attempt resets app data, re-injects the data and presses Home, so it starts from reset, not from the explored mid-state.
- Each attempt gets the concatenated hints from earlier attempts.
- Max steps: 2.5 × golden_steps + 1.

## Critic
Code: rollout/mobilegym_critic/evaluator.py.
- Per-step descriptions from before/after images (Qwen3-VL-8B).
- A final decision from task + step descriptions + the last 3 screenshots stitched into one image. Output: 0/1, per-step reasonable/unreasonable, feasibility. Gemini-2.5-Pro in the main results (inferred).
- A hint call on failure.
- **Self-judging:** the all-Qwen3-VL-8B critic (= the policy model) is ablated: trajectory accuracy 77.0 vs 89.75, but still 52/67/70 Pass@1/2/3 (Tables 9/10). So a self-judged condition is a tested configuration.

## Training (HiFPO)
- Keep tasks whose success rate across the 4 attempts is in [0, 0.9]. Compute this from all attempts (training/verl/utils/dataset.py:691-719), NOT via the data processor, which has an ordering bug that recomputes the success rate after best-attempt selection.
- From the best attempt, keep the reasonable steps.
- **Single-step GRPO:**
  - Prompt = the exact rollout prompt, including hints.
  - G=5, T=1.0.
  - Reward = 0.2·[action type match] + 0.8·[type match]·S_arg(pred, the policy's own logged action), where S_arg is the argument-match score.
  - Clip 0.2 / 0.3, low_var_kl 0.01, lr 1e-6, 4 epochs, batch 128.
  - Follow training/README.md; the script defaults conflict on epochs, reward weights and the hint removal flag.

## Results
- **Table 1, AndroidWorld Pass@1:**

| Policy | Base | 200 tasks | 400 tasks | 900 tasks |
|---|---|---|---|---|
| Qwen3-VL-8B | 40.5 | 47.4 | 52.6 | 50.9 |
| GUI-Owl-1.5-8B | 56.0 | — | — | 67.2 |

- **Table 8:** GRPO beats SFT on the same data (50.9 vs 47.4).
- **Table 6/7:** hints raise the rollout success rate from 52 to 77.
- **Table 11:** exploration-grounded tasks beat landing-screen tasks.

## Released vs missing
- **Released:** everything, including datasets and 6 checkpoints.
- **Missing:** the exploration and curriculum model identities, the subset seeds.

## Relevance to our trees
- MobileForge consumes exactly our kind of data: exploration paths with screenshots, turned into tasks that are then attempted from reset.
- A tree root→node path ≈ an exploration trajectory.
- Two requirements:
  1. The root must equal the env reset state, so that the proposed tasks are executable from reset.
  2. Random-walk paths lack an "original goal". That is a declared deviation.

---

# (1) Comparison table

| Method | Task proposal source | Exploration style | Judge input (original) | Objective | Iterations / data carry-over | Key hyperparameters (orig.) |
|---|---|---|---|---|---|---|
| PAE | Claude/Qwen from **website name only** (+1 format example); optional demo screenshots; 25 per call; 30k–100k fixed pool, no dedup | none; attempts from homepage reset | last 3 screenshots + ANSWER → SUCCESS/NOT | filtered BC (NLL on successes) + **shortest-20% length filter** | many iterations (code 200), cumulative buffer, continue weights | LLaVA 7B/34B, **LoRA r=256**, lr 2e-5, T=1.0, 10 steps, 256–4096 rollouts/iter, 1–4 epochs |
| OpenWebVoyager | GPT-4o self-instruct from site name + seed queries, mpnet dedup; 480/round, all made up front | none; ≤5 sequential attempts from homepage, stop at first success | GPT-4o: last 3 screenshots + answer | SFT on D_IL ∪ D_j | 3 rounds, **non-cumulative** self-data + IL anchor, continue weights | idefics2-8B full FT, lr 1e-5 cosine, batch 64, 300 steps/round, T=1.2, 15 steps |
| WebEvolver | none (OWV's fixed 1,516 queries) | none; WM-imagined trajectories on failed queries | zero-shot base Llama over all accessibility trees + answer; WMLA critic GPT-4o {0, .5, 1} | SFT on successes; WM = next-observation SFT; WMLA k=3, d=2 | 2 iterations, cumulative; final SFT(D_1 ∪ D_w) | Llama-3.3-70B; training hyperparameters **unreported**; text-only observations |
| AppVLM | **benchmark templates** (82 AW), random params | none; high-T attempts, re-queue tasks with <10 successes | AW rule checker every step (no terminate action) | SFT on successes (ReST) | 3–4 iterations, cumulative, continue; final retrain from base | PaliGemma-3B full FT, lr 3e-6, 3 epochs, batch 64; no code |
| MobileRL | **benchmark templates** (2,000 AW train instances); AndroidLab model-generated | none; G=16 group rollouts from reset | AW rule checker (AL: trained RM over all screenshots + XML) | **AdaGRPO**: GRPO + SPA + AdaPR + FCF + negative pruning, clip 0.2, KL 1e-3 | ~100 online steps; replay buffer 256 | 7–9B full FT, lr 1e-6, T=1.0, 50 turns, batch 256; no training code |
| UI-Voyager | **benchmark templates** + param perturbation (>7000) | none; G=8 rollouts from identical init | AW checker AND self-terminate | RFT (SFT on successes) then **GRSD**: SSIM fork-point SFT of success responses into failed contexts | 3 RFT rounds + GRSD iterations | Qwen3-VL-4B; θ=0.95; T=0.7; training hyperparameters unreleased |
| TTI | 128K fixed pool: seed-prompting + **explore-and-propose agent** (code not released); WA code uses real tasks | proposal exploration separate; attempts from homepage reset | **policy itself** (code) / Gemma-27B (paper): last 3 screenshots + final tree + URL + answer | filtered BC, **no** length filter, recency/site-weighted replay, **horizon curriculum 10→20→30** | 12–15 iterations × 512 rollouts, cumulative buffer, continue | Gemma-3-12B full FT, lr 4e-6, clip 0.01, T=1.0 |
| OS-Themis | Qwen-235B-synthesized templates (96 tasks) | none (RL) | multi-agent critic: text history + before/after screenshots of milestones | multi-turn GRPO n=4, no KL | 4 episodes, on-policy | lr 1e-6, 50 steps, 20-step window; RL code unreleased |
| MobileForge | **from exploration trajectories** (goal + ≤15 annotated screenshots), dedup vs existing tasks; AW test tasks as style few-shots (contamination) | DFS GUI-explorer (separate phase); attempts from reset | step descriptions + last 3 screenshots → 0/1 + per-step labels + hint | single-step GRPO G=5, reward = match own reasonable action; hint-conditioned | single pass; K=4 serial attempts with hints | 8B full FT, lr 1e-6, KL 0.01, clip 0.2/0.3, 4 epochs |

# (2) Which methods are particular about task proposal or exploration, and how

- **Particular about proposal inputs, in a way that conflicts with our trees:**
  - **PAE:** proposes blind from the site name before any interaction.
    - Its documented alternative is "user demo" screenshots. That is the legitimate entry point for tree screenshots, provided tasks still start from the home page (Fig. 16 rule 4).
    - All attempts are online resets to the landing page.
  - **OpenWebVoyager:** self-instruct from site name plus seed queries, all generated up front. Homepage reset per attempt. No slot for exploration data, so feeding in trees is an addition.
- **Bound to benchmark task templates (no proposal component at all):**
  - **AppVLM, MobileRL (AndroidWorld), UI-Voyager:** these need a fixed task set with a checker and resettable task initialisation.
  - Under (a) on AndroidWorld they are faithful as-is.
  - On GameBoy, WebVoyager and any (b) setting, the task source must come from us. This is unavoidable and must be declared. UI-Voyager and MobileRL additionally need group rollouts from an identical s0, which our restorable states make exact.
- **Fixed borrowed set, otherwise indifferent:**
  - **WebEvolver:** uses OpenWebVoyager's query set. Its world model can additionally consume tree transitions (an addition).
  - **OS-Themis:** a critic only, indifferent to task source; its own RL tasks came from a 235B teacher.
- **Exploration-grounded, where our trees fit naturally:**
  - **MobileForge:** tasks come from exploration trajectories, and its landing-screen baseline is worse (Table 11). It requires tasks to be executable from the reset state, which is satisfied if the tree root is the reset state. Random-walk paths lack its per-trajectory "goal".
  - **TTI:** its paper describes an explore-and-propose agent that pairs each task with an answer. Proposing from tree states with the E.1 prompt fits naturally.
- **Common to all nine:** task ATTEMPTS start from an env reset (homepage, app home, or task init). None starts attempts mid-exploration. Starting from mid-tree states is a deviation for every method. Exploration and task attempts are the same activity in every method except MobileForge and TTI's proposer.
- **Condition (a) on WebVoyager:** not possible as an oracle in any method, because WebVoyager's "checker" is itself an LLM judge. Declare whichever judge plays the oracle role.

# (3) Minimal faithful reproduction per method

**Deviations common to ALL methods (declare once):**
- Gemma-4-31B instead of the original backbone.
- LoRA with FSDP instead of full FT, so learning rates are re-tuned (typically ~1e-5 to 1e-4 instead of 1e-6 to 2e-5).
- Smaller rollout and batch scale.
- Our action and observation spaces: set-of-mark for web, Android tap/type/swipe, GameBoy buttons.
- A terminate/ANSWER action added wherever the original lacked one.
- Every stronger-teacher SFT or IL phase removed, so we start from zero-shot Gemma.
- Proposer, judge and critic roles played by the policy itself:
  - in (b): the frozen initial policy, or the current one; state which;
  - in (a): the benchmark checker.
- Benchmark train tasks for (a); tasks proposed from our trees for GameBoy/Web where no checker exists.
- vLLM instead of the original serving engine.

**PAE**
- (a) Benchmark train tasks; reset per task; filtered BC with the shortest-20% filter; cumulative buffer; T=1.0; 10-step cap (scale for GameBoy); judge = checker.
- (b) Gemma proposes 25 tasks per call with the Fig. 14 prompt from the site/app/game name. Optional "user-demo" variant with tree screenshots per Fig. 16. Gemma judge, Fig. 13 prompt, last 3 screenshots + answer.
- Declare: proposer and judge are Gemma; rollout and epoch counts; grad clip (the code doesn't clip).

**OpenWebVoyager**
- 3 rounds; ≤5 sequential attempts per query, stop at first success; T=1.2 (or 1.0, declare); 15 steps; SFT on this round's successes only, continuing the LoRA.
- (a) Queries are benchmark train tasks (self-instruct dropped).
- (b) Gemma self-instruct, 10 per site per round from seeds, mpnet dedup; Gemma judge with the auto_eval prompt (last 3 screenshots + answer).
- Declare: no IL anchor, or use D_1 as the anchor; loss mask choice; trees unused.

**WebEvolver**
- 2 iterations, 1 rollout per task at T=0, cumulative buffer.
- A WM LoRA predicting text observations (set-of-mark/UI text; GameBoy uses free-text descriptions), with a Gemma-distilled rationale.
- Synthesis on unsolved tasks from the REAL reset observation, depth 7, T=0.7; synthetic data filtered by the Gemma judge in both conditions.
- Final SFT(D_1 ∪ D_w). WMLA k=3, d=1 (d=2 as an extension), with a Gemma critic.
- Declare: query set; untrained-in-paper hyperparameters guessed; the text-observation modality mismatch; self critic instead of GPT-4o; tree transitions in the WM as an optional addition.

**AppVLM**
- AndroidWorld (a): 82-task subset, train seeds disjoint from eval; K≈8 rollouts per task at T≈1.0 plus top-ups to τ=10; drop unchanged screens; rebalance; continue LoRA for 3 epochs per iteration; 4 rounds; final fresh LoRA on all data.
- (b) and other environments: add terminate plus a self-judge.
- Declare: no AndroidControl SFT (the paper's AWO ablation shows a large drop for PaliGemma); guessed rollout parameters.

**MobileRL**
- GRPO with SPA (α=1), 1:2 negative pruning, AdaPR (buffer 256, top 25%, recompute log-probs), FCF ("epoch" = one sampling of a task, declared), clip 0.2, KL 1e-3 to LoRA-off, T=1.0.
- Scaled: G=8, 8–16 tasks per step, 30 turns, 50–100 steps.
- (a) AndroidWorld `create_suite(random=True)` minus test seed 30.
- (b) Gemma answering the D.3.2 reward prompt.
- Declare: no SFT stages; guessed κ/γ, batch and FCF semantics; image downscaling; tree-sourced tasks and start states outside AndroidWorld.

**UI-Voyager**
- 2–3 RFT rounds (T=0.7, success = judge AND self-terminate, step-level SFT), then GRSD: G=8 from an identical restored start state, shortest-success teacher, `fork_algorithm` verbatim (θ=0.95, hash 0.80, 480×270; crop adjusted per environment), SFT on fork samples plus successful steps (ratio declared), re-roll each iteration.
- (b) The self-judge decides both RFT filtering and the teacher; consider requiring self-terminate.
- Declare: no multi-scale round-0 generators; seed perturbation replaced by AndroidWorld seeds or tree tasks; SSIM on raw (non-set-of-mark) frames; hyperparameters guessed.

**TTI**
- Horizon schedule [10, 20, 20, 30, …] (rescaled for GameBoy); 512 (or 256) rollouts per iteration from reset; no length filter; drop N/A answers; recency + site-balanced replay of ≤512; 1 epoch; T=1.0/top_p 0.95; 3-image history; eval at h=30 on held-out tasks.
- (a) Benchmark train tasks with checkers. This mirrors the released WebArena config, but use train tasks disjoint from test, which TTI's own file is not.
- (b) Pool from the E.1 explore-and-propose prompt over tree states, seeded with benchmark train examples. Judge = the policy (matches the code; the paper says 27B).
- Declare: offline tree-based proposer; no Bing/PDF helper; no domain hints.

**OS-Themis**
- (a) = plain multi-turn GRPO (n=4, T=1.0, ≤50 steps, 20-step window, format penalty −1, no KL, 4 episodes) with oracle reward, as a control only.
- (b) Same GRPO with the released critic code as-is, served by a frozen initial Gemma for all 4 roles. The policy must emit think/operation/action text.
- Declare: task source (no 235B templates); guessed window and advantage handling; single review round (code) vs iterative (paper).

**MobileForge**
- Tree root→node paths (≤15 annotated screenshots plus goal if the node came from VLM exploration) → the code's curriculum prompt with principles generalised and dedup against existing tasks.
  - Few-shots: benchmark TRAIN tasks in (a); generic examples in (b). Never test tasks.
- K=4 serial attempts from reset with the hint block.
- Critic:
  - (a) checker for the outcome, Gemma for step labels and hints (declare the mix);
  - (b) frozen Gemma for all critic calls.
- Train: success rate from all attempts in [0, 0.9]; best attempt; positive steps; single-step GRPO G=5, reward 0.2/0.8, clip 0.2/0.3, KL 0.01, 4 epochs; 200-task scale.
- Declare: our trees replace GUI-explorer DFS + RAG; non-Android environments lack data injection, so the reset state must contain the entities seen during exploration.

**Feasibility ranking on 4×A6000 with 31B LoRA, cheapest first:**
1. AppVLM, OpenWebVoyager, PAE, TTI, WebEvolver training (single rollouts plus offline SFT).
2. MobileForge (offline single-step GRPO).
3. UI-Voyager (G=8 groups, SFT updates).
4. WebEvolver WMLA at inference (~9–15 calls per step).
5. OS-Themis (GRPO plus a ~14-call critic per trajectory).
6. MobileRL (G=16 online GRPO with ~100 steps; days per run even when scaled down).

All online methods must time-share the GPUs between vLLM (31B bf16 ≈ 62 GB, needs TP≥2) and FSDP LoRA training.

**Most important code-vs-paper flags** (detailed per method above):
- PAE: grad clipping disabled; rollouts and epochs differ from the paper.
- OpenWebVoyager: loss covers non-answer tokens.
- WebEvolver: no training code; judge filter missing from the converter; o_0 is imagined.
- MobileRL and UI-Voyager: no training code at all.
- TTI: self-judge in code; WebArena trains on real tasks that overlap the test set.
- OS-Themis: reviewer loop differs from the paper.
- MobileForge: AndroidWorld test tasks used as curriculum few-shots; a processor ordering bug in success-rate filtering.

The extra downloads the sub-agents made are in SCR/data_webevolver/ and SCR/uiv_extract/.
