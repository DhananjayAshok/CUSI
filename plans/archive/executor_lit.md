# Executor literature review and feature catalogue (design plan step 2a/2b)

- **Date:** 2026-10-13.
- **Purpose:** turn prior work (papers, agent harnesses, the `initial_browse.md` survey, GameBoyRL's own arms) into a
  catalogue of **features**. A feature is one switch we can flip on our executors (or supervisor) with the fixed model
  `google/gemma-4-31b-it`. The catalogue sets the screening order in `plans/archive/design_plan.md` §5. The failure taxonomy (§2c) has not
  been run yet, so no `failures:` sources appear.
- **Method:** web search plus the arXiv abstract/HTML pages, READMEs and blog write-ups of the harnesses, read on
  2026-10-13. Numbers marked **(verified)** were read from the paper text or tables during this review. Numbers marked
  **(abstract)** come from the abstract only. Numbers marked **(recalled)** are from memory of a well-known abstract and
  were not re-read. Where a number could not be found, the entry says "not reported" or "not verified".
- **Source labels** (as in `plans/archive/design_plan.md` §2b): `paper:<short name, year>`, `harness:<repo>`, `browse:<repo>`
  (a feature described in `initial_browse.md`), `failures:<bucket>` (none yet), `user` (from the seed list in
  `plans/archive/design_plan.md` §3), `ours` (this repo's state-space work, or our own earlier eval results in
  `storage/eval/ablation_tables.md`).
- **Failure-bucket codes** (design plan §2c), used in the catalogue:
  - `GRD` grounding miss;
  - `PLAN` wrong plan or misread task;
  - `EARLY` finishing too early or claiming false success;
  - `LOOP` loop or no-op actions;
  - `FMT` invalid action format;
  - `STEPS` running out of steps;
  - `READ` misreading the screen;
  - `ENV` env or harness error.
- **Evidence scale:**
  - **strong:** several controlled ablations, in more than one paper or setting, point the same way;
  - **medium:** one controlled ablation, or several uncontrolled comparisons that agree;
  - **weak:** anecdotal, a harness that ships the feature with no ablation, or results that disagree.

**How much to trust a reported gain.** `initial_browse.md` §4 and §6 explain why scores from different repos are not
comparable. The same applies here, with these rules of thumb:

1. **Bigger models and different judges.** Almost every reported gain uses GPT-4-class or Claude-class models, often
   with a different judge (WebVoyager's GPT-4V auto-eval, WebJudge, a self-reported flag). The WebVoyager auto-evaluator
   also agrees poorly with humans (paper:Illusion of Progress, 2025). Only like-for-like ablations inside one paper count
   as evidence. Cross-repo leaderboard deltas never do.
2. **Different action power.** Several Android and web "agents" add separate grounding, OCR or icon models
   (Mobile-Agent v1/v2/E, Agent S2), or are fine-tuned (UI-TARS, Aguvis, GUI-Owl, V-Droid). For these we keep only the
   prompting or harness idea. Their headline numbers carry no weight for us.
3. **Different budgets and scoring.** Some report pass@k, others score timeouts, use oracle retries (EvoSkill) or
   larger step budgets. Our step limits are fixed (§0), so a feature that only helps when given more steps is worth less
   to us.
4. **Our setup already has some features on.** The baselines already include WebVoyager's Set-of-Mark (SoM) and
   element list, M3A's reflective summary, and the GameBoy done-check and visual history. A paper's gain from adding one
   of these is not a gain we can get again.
5. **The most relevant evidence is the Gemma result.** The single most relevant data point is paper:TTI 2025, a
   prompting-only "check again" result on **Gemma 3 12B** (same family as our model).

**What our own results already show** (`ours`, from `storage/eval/ablation_tables.md`, gemma-4-31b-it, supervisor off
unless stated):
- **Baselines:** Web 53.4% (429 tasks), Android 61.3% (115), GameBoy Red 36.7% (49).
- **Plan-driving supervisors hurt Web and Android:**
  - `subgoal`: −12.1 points on Web, −24.8 on Android;
  - `revision`: −3.3 on Web, −8.3 on Android.
- **On GameBoy they are neutral to slightly positive:** +4.1 points each, not significant.
- **How episodes end:**
  - **Web:** 159 of 429 baseline episodes end `truncated` and 14 `max_steps`.
  - **GameBoy:** 30 of 49 end `truncated`.
  - **Android:** 19 of 115 end `truncated`.
  - "Out of steps" is therefore likely a large bucket. This is to be confirmed in §2c, because `truncated` must first be
    checked to mean the step budget.

---

## 1. Sources

Each entry gives: **idea**; **gain** (reported gain and its setting); **applies?** (applies / partly / no, with the
reason).

### 1.1 Web

- **paper:WebVoyager, 2024** (He et al., arXiv 2401.13919); also `harness:WebVoyager`, `browse:WebVoyager`.
  - **Idea:** a SoM screenshot plus a text element list, ReAct Thought/Action, and the 3 most recent screenshots kept.
  - **Gain (verified):**
    - Human-judged success: 59.1% for GPT-4V, 40.1% for text-only (a11y tree), 30.8% for GPT-4 All Tools. That is
      +19 points for vision on 643 tasks.
    - Auto-eval: 57.1% vs 44.3%.
    - Error analysis of 300 failures: navigation stuck 44.4%, visual grounding 24.8%, hallucination 21.8%, prompt
      misalignment 9.0%.
    - Black single-colour boxes beat multicoloured ones (no number given).
    - There is no ablation of the number of screenshots in context. That ablation exists only for the judge.
  - **Applies:** yes; it is our Web baseline.
- **paper:SeeAct, 2024** (Zheng et al., 2401.01614); also `browse:android_world` (the GW SeeAct port).
  - **Idea:** generate the intended action in natural language first, then ground it with a second, multiple-choice
    call.
  - **Gain (abstract):**
    - 51.1% task success on live sites when grounding is done by hand (oracle), with "a substantial gap" for automatic
      grounding.
    - "SoM … not effective for web agents."
    - The best grounding uses both HTML and visual information.
  - **Applies:** partly. Two calls per step, and it was measured on GPT-4V. The SoM finding is relevant to F01.
- **paper:Set-of-Mark, 2023** (Yang et al.).
  - **Idea:** numbered marks on image regions let a VLM refer to them by ID.
  - **Gain:** not reported for agents. It was measured on grounding and referring tasks with GPT-4V.
  - **Applies:** yes; it is already in the Web and Android baselines.
- **paper:WebArena, 2023** (Zhou et al.).
  - **Idea:** the a11y-tree text agent with CoT, plus a hint that unachievable tasks exist.
  - **Gain (recalled):** GPT-4 reaches 14.4% on WebArena.
  - **Applies:** partly. It is a text-only baseline design and a different benchmark.
- **paper:VisualWebArena, 2024** (Koh et al., 2401.13649).
  - **Idea:** a SoM screenshot plus an element list for a VLM.
  - **Gain (verified):** GPT-4V with image + captions + SoM scores 16.37%, against 15.05% with image + captions + a11y
    tree, and 7.25% for text-only GPT-4. That is +1.3 points for SoM over the tree.
  - **Failure notes:** agents undo completed tasks, give up early and loop between pages, and the authors call for
    state tracking.
  - **Applies:** yes (observation design and failure types).
- **paper:Agent-E, 2024** (Abuelsaad et al., 2407.13032).
  - **Idea:** a planner plus a browser-navigation sub-agent, DOM distillation into three task-dependent views,
    "change observation" (MutationObserver feedback after each action), and skills.
  - **Gain (verified):** 73.1–73.2% on WebVoyager with GPT-4-Turbo, against 57.1% for WebVoyager and 52.6% for WILBUR
    in the same table. About 25 LLM calls per task.
  - **No numeric ablations.** The contributions of the planner, DOM modes and change observation are shown only by
    example.
  - **Applies:** partly. A text-only DOM agent with a different action power, so the cross-agent delta carries little
    weight. Change observation and DOM views transfer.
- **paper:AgentOccam, 2024** (Yang et al., 2410.13825).
  - **Idea:** align the observation and action spaces to the LLM. Remove rarely useful actions, disable scrolling (show
    the whole page), simplify the observation, replay only pivotal history, and plan with branch/prune actions.
  - **Gain (verified):** incremental ablation on a 190-task WebArena dev subset with GPT-4-Turbo:

    | Configuration | Success |
    |---|---|
    | Vanilla | 14.2% |
    | Action-space reduction | 25.8% |
    | No scrolling | 30.0% |
    | Observation simplification | 34.7% |
    | Selective history | 36.8% |
    | Plan branch/prune | 44.2% |

  - **Applies:** partly. It is text-only WebArena; our sites are live and we have a screenshot. Observation pruning,
    action pruning and the plan field transfer.
- **harness:browser-use** (blog "SOTA technical report"); `browse:WebCoach` (browser-use 0.7.1).
  - **Idea:**
    - Strict JSON output `{thinking, evaluation_previous_goal, memory, next_goal, action[…]}`.
    - A todo list and "Step k of N".
    - `*` marks new elements.
    - Up to 10 actions per step.
    - About 19 actions, including `go_to_url`, `extract_structured_data` (an extra LLM call over the page content),
      `send_keys` and dropdowns.
    - Only `done` is allowed on the last step.
    - The run stops after 3 consecutive failures.
  - **Gain:** 89.1% on 586 WebVoyager tasks with GPT-4o. This is self-reported, on a modified task set (55 removed, dates
    changed), with the WebVoyager auto-judge. There is no per-feature ablation.
  - **Eclair (browse):** WebCoach 10/15 with the coach, 8/15 without, with gemma.
  - **Applies:** partly. Individual features transfer; the number does not.
- **harness:OpenHands** (BrowsingAgent / VisualBrowsingAgent, via its README).
  - **Idea:** BrowserGym observation (a11y tree, plus a SoM screenshot in the visual variant) with prior thoughts and
    actions.
  - **Gain:** not verified. No published number was found.
  - **Applies:** partly. It contributes no new feature beyond F01–F03.
- **paper:Tree search for LM agents, 2024** (Koh et al., 2407.01476).
  - **Idea:** best-first search over sampled actions with a VLM value function. The value comes from 20 sampled
    self-evaluations, averaged into a success / on-track / fail score. Actions come from 20 sampled CoTs, aggregated by
    count into the top 5.
  - **Gain (verified):**
    - VisualWebArena: 18.9% → 26.4% (+39.7% relative). On a 200-task subset: 24.5% → 32.0% at budget c=5, → 37.0% at
      c=20.
    - WebArena: 15.0% → 19.2%. GPT-4o throughout.
    - It needs state resets (replaying actions).
  - **Applies:** the search itself does not (live sites and Android cannot be reset cheaply; GameBoy save states need a
    ruling). The action-sampling aggregation and the self-evaluation value transfer as F37 and F38.
- **paper:WILBUR, 2024** (Lutz et al., 2404.05902).
  - **Idea:** retrieve task demonstrations with a trained ranker, plus intelligent backtracking after mistakes.
  - **Gain (abstract):** +8 points over text-only SOTA on WebVoyager. 52.6% overall per Agent-E's table.
  - **Applies:** partly. The trained ranker and demo memory are out of scope. Backtracking transfers (F35).
- **paper:Agent Workflow Memory, 2024** (Wang et al., 2409.07429).
  - **Idea:** induce reusable workflows from past (or online, self-judged) trajectories and add them to the prompt.
  - **Gain (abstract):** +24.6% relative on Mind2Web, +51.1% relative on WebArena. Online absolute gain of 8.9–14.0
    points.
  - **Applies:** partly. It needs cross-task memory, which breaks per-task independence on the test set unless the
    workflows come only from dev or exploration (F53).
- **paper:WebDreamer, 2024** (Gu et al., 2411.06559).
  - **Idea:** the LLM simulates the outcome of each candidate action (horizon 1), scores the simulated states, and the
    best action is executed.
  - **Gain (verified):**
    - VisualWebArena: reactive 17.6%, WebDreamer 23.6%, tree search 26.2%.
    - Online-Mind2Web: 26.0% → 37.0%.
    - Mind2Web-Live: 20.2% → 25.0%.
    - GPT-4o, about 4× faster than tree search. Longer horizons hurt.
  - **Applies:** yes (F39). It needs no reset; it costs about 2k extra calls per step.
- **paper:ExACT / R-MCTS, 2024** (Yu et al., 2410.02052).
  - **Idea:** MCTS with contrastive reflection and a multi-agent-debate value.
  - **Gain (abstract):** +6–30% relative on VisualWebArena over the previous SOTA (GPT-4o).
  - **Applies:** no for the search, which needs resets. Debate-style state evaluation is a variant of F38.
- **paper:TTI ("Thinking vs Doing"), 2025** (Shen et al., 2506.07976).
  - **Idea:** after the agent signals completion, re-prompt it to "pause and think again" (check-again), possibly 2–3
    times.
  - **Gain (verified):**
    - **Gemma 3 12B**, prompting only, WebArena (62 tasks, h=30): CoT 23.8% → at least 28% with check-again. The action
      changed in about 25% of re-checks.
    - Per-step "budget forcing" (longer reasoning) plateaus. Best-of-n gave under 3% at equal compute.
    - Too many re-checks hurt, and combined prompting tricks were not additive.
  - **Applies:** yes (F41). Same model family. Note that interaction scaling (more steps) is not available to us, since
    steps are fixed.
- **paper:RegionFocus, 2025** (Luo et al., 2505.00684).
  - **Idea:** test-time zoom into the relevant region for grounding, plus an "image-as-map" of past action landmarks.
  - **Gain (abstract):** "28+%" on ScreenSpot-Pro and "24+%" on WebVoyager, on top of UI-TARS-72B and Qwen2.5-VL-72B
    (coordinate grounders). It is not stated whether the gains are absolute or relative.
  - **Applies:** partly. Our grounding is by index, so zooming helps reading (`READ`) more than clicking (F07).
- **paper:Illusion of Progress, 2025** (Xue et al., 2504.01382).
  - **Idea:** a comparability critique. The WebVoyager auto-eval agrees poorly with humans, and published WebVoyager
    gains are inflated.
  - **Gain:** —.
  - **Applies:** yes, as a weighting rule. Our judge is the same model, so features that game the judge (for example
    confident answer wording) must be checked by hand (§5).
- **browse:WebEvolver** (Cognitive Kernel v2).
  - **Idea:** a text-only a11y tree (≤300 nodes); a Python planner calling a `CallWeb` sub-agent; a final-answer
    rewrite call; a world-model search (3 candidates, predicted next tree, critic) that the eval endpoint never uses.
  - **Gain:** eclair 11/15 with a text-only judge (browse). Not comparable.
  - **Applies:** partly (F02, F39, F49).
- **browse:SynthAgent.**
  - **Idea:** a raw screenshot plus an a11y tree plus an addressable-element list and a "MUST NOT choose" list; the last
    3 step summaries; 15 rules; GOTO / HOVER / PRESS / STOP actions.
  - **Gain:** eclair 10/15 (browse). Not comparable.
  - **Applies:** partly (F03, F12, F30).
- **browse:GUI-Owl-1.5 web.**
  - **Idea:** current and previous screenshot, CSS-SoM, label → coordinate clicks, forced answer on the last step.
  - **Gain:** not run (broken as shipped).
  - **Applies:** partly (F08, F11).
- **browse:web_policy** (playenv `WebVoyagerPolicy`).
  - **Idea:** WebVoyager's prompts in the free-play env, with a mid-episode Chrome restart that keeps history.
  - **Gain:** —.
  - **Applies:** nothing new beyond the baseline. The Chrome restart is harness robustness (`ENV`).

### 1.2 Android

- **paper:AndroidWorld (M3A/T3A), 2024** (Rawles et al., 2405.14573); also `browse:android_world`.
  - **Idea:**
    - M3A: raw + SoM screenshot, an element list with state flags, a second call that writes a reflective summary
      (intent, success, reason, next-step advice) from before/after screenshots, long GUIDANCE, and a `status`
      infeasible option.
    - T3A: text-only with bboxes.
  - **Gain (verified), AndroidWorld:**

    | Agent | Model | Input | Success |
    |---|---|---|---|
    | M3A | GPT-4T | text-only | 30.6% |
    | M3A | GPT-4T | SoM | 25.4% |
    | M3A | Gemini 1.5 Pro | text-only | 19.4% |
    | M3A | Gemini 1.5 Pro | SoM | 22.8% |
    | M3A-Simple (no guidelines, no reflection) | GPT-4T | — | 19.8% |
    | M3A (full) | GPT-4T | — | 30.6% |

    - The paper says multimodal input "generally does not outperform" text-only.
    - Failure classes: perceptual, reasoning, missing-knowledge and grounding errors.
  - **Applies:** yes; M3A is our Android baseline.
- **paper:AppAgent, 2023** (Zhang et al., 2312.13771).
  - **Idea:** numeric-label tapping, plus app documentation built by autonomous exploration or by watching demos.
  - **Gain (verified):** GPT-4V, 45 tasks:
    - raw action space 2.2%;
    - custom labelled action space with no docs 48.9%;
    - auto-exploration docs 73.3%;
    - demo docs 84.4%;
    - hand-written docs 95.6%.
  - **Applies:** partly. Docs need exploration of the same apps, which is a fairness question (F29 and §7 of the
    design plan).
- **paper:AppAgent v2, 2024.**
  - **Idea:** a flexible action space (parser plus visual features) and a knowledge base with RAG from exploration.
  - **Gain:** not verified.
  - **Applies:** partly (F29).
- **paper:Mobile-Agent v1, 2024.**
  - **Idea:** OCR and icon detection models locate elements; self-planning and self-reflection.
  - **Gain:** not verified.
  - **Applies:** no for the perception models. Reflection is covered by F34.
- **paper:Mobile-Agent-v2, 2024** (Wang et al., 2406.01014).
  - **Idea:** a planning agent (text-only task-progress summary), a decision agent, a reflection agent (checks each
    action's outcome) and a memory unit ("focus content").
  - **Gain (verified):** on the authors' own 88 instructions with GPT-4V/GPT-4, full system 88.6% (basic) and 61.4%
    (advanced) success. Removing components drops to 59.1–86.4% and 29.5–54.5%. The text says removing the **planning
    agent hurts most**.
  - Uses separate OCR, GroundingDINO and Qwen-VL perception models.
  - **Applies:** partly. Planning, reflection and memory transfer. Perception does not.
- **paper:Mobile-Agent-E, 2025** (Wang et al., 2501.11733).
  - **Idea:** Manager, Perceptor (separate models), Operator, Action Reflector and Notetaker, plus self-evolving Tips
    and Shortcuts learned across tasks.
  - **Gain (verified):** on Mobile-Eval-E (Satisfaction Score, GPT-4o): 75.1 against 53.0 for MA-v2; 86.9 with
    evolution. Evolution adds about +6.5 on average across backbones. There is no reflector ablation.
  - **Applies:** partly. The roles transfer. Cross-task evolution is F53 (a protocol issue).
- **paper:Mobile-Agent-v3 / GUI-Owl, 2025** (2508.15144); also `browse:MobileAgent`.
  - **Idea:** a Manager / Worker / Reflector / Notetaker (+RAG) framework around a fine-tuned GUI-Owl.
  - **Gain (abstract):** GUI-Owl-7B alone 66.4% on AndroidWorld; the framework 73.3%. These are not like-for-like model
    sizes.
  - **Browse detail:**
    - the Executor sees the last 5 actions and outcomes;
    - the ActionReflector classifies the outcome as A/B/C;
    - the Notetaker is gated by goal text;
    - goal-string hacks and auto-dismissal of Chrome's first-run dialogs.
  - **Applies:** partly. The roles transfer. The task-specific hacks are excluded (F51).
- **paper:Agent S, 2024** (Agashe et al., 2410.08164).
  - **Idea:** experience-augmented hierarchical planning (web search plus memory) and an agent-computer interface.
  - **Gain (abstract):** OSWorld +9.37 points (+83.6% relative).
  - **Applies:** partly. Hierarchical planning maps to F47. External search and experience memory are out (F53).
- **paper:Agent S2, 2025** (2504.00906).
  - **Idea:** Proactive Hierarchical Planning (re-plan after **every** subgoal, not only after a failure) and a mixture
    of grounding experts (separate models).
  - **Gain (verified):**
    - AndroidWorld 54.3%.
    - OSWorld: proactive planning gives +4.62 points at 15 steps and +6.15 at 50 steps over reactive planning; the
      mixture of grounding gives +3.1 and +4.6.
  - **Applies:** partly. Proactive planning transfers (F47). The mixture of grounding does not (separate models).
- **paper:Agent S3 / Behavior Judge (bBoN), 2025** (Gonzalez-Pumariega et al., 2510.02250).
  - **Idea:** run N full rollouts, turn each into a "behavior narrative" (per-step facts from before/after screenshots),
    and let a judge pick one by comparative multiple choice.
  - **Gain (verified):**
    - AndroidWorld 68.1% → 71.6% (N=3, GPT-5).
    - OSWorld GPT-5 62.6% → 69.9% (N=10).
    - Narratives 60.2% against screenshot-only 56.0% and plain summaries 55.0% for the judge.
  - **Applies:**
    - Trajectory-level best-of-N: partly (Web answers only, F42).
    - Narratives as supervisor input: yes (F46).
- **paper:GTA1, 2025** (2507.05791).
  - **Idea:** sample K=8 candidate actions per step; a judge (which can be the planner model itself) picks one.
  - **Gain (verified):** OSWorld (o3 planner/judge) 41.3% → 43.4% at 50 steps, and 43.4% → 45.2% at 100 steps. The
    paper reports no AndroidWorld test-time-scaling number.
  - **Applies:** yes (F40).
- **paper:UI-TARS, 2025** (2501.12326).
  - **Idea:** a native fine-tuned agent with "system-2" thought patterns (task decomposition, reflection, milestone
    recognition).
  - **Gain (abstract):** AndroidWorld 46.6% against GPT-4o 34.5%.
  - **Applies:** no for the model (it is trained). The thought template transfers (F19).
- **paper:Aguvis, 2024** (2412.04454).
  - **Idea:** a pure-vision fine-tuned agent with a structured inner monologue (observation description, thought,
    low-level instruction).
  - **Gain:** not verified.
  - **Applies:** no for the model. The monologue template transfers (F19).
- **paper:OS-Atlas, 2024** and **paper:CogAgent, 2023.**
  - **Idea:** a grounding foundation model and a high-resolution GUI VLM.
  - **Gain:** not relevant.
  - **Applies:** no; both are separate or fine-tuned models (F52).
- **paper:AutoDroid, 2023** (2308.15272).
  - **Idea:** functionality-aware UI text representation, plus app memory from automated exploration injected into the
    prompt.
  - **Gain (abstract):** action accuracy 90.9%, task success 71.3% on their 158 tasks; +36.4 and +39.7 points over
    GPT-4 baselines.
  - **Applies:** partly (F03, and F29 on fairness).
- **paper:DroidBot-GPT, 2023.**
  - **Idea:** translate the GUI state into a text prompt and let the LLM choose an action.
  - **Gain (recalled, not verified):** about 39% task completion.
  - **Applies:** no; nothing beyond T3A (F02).
- **paper:AndroidLab, 2024** (2410.24024).
  - **Idea:** compare XML mode with SoM mode, and ReAct with SeeAct prompting.
  - **Gain (verified):**
    - GPT-4o: XML 25.4%, SoM 31.2%. Gemini 1.5 Pro: XML 18.8%, SoM 16.7%.
    - ReAct: XML 25.4% → 33.3% (GPT-4o) and 18.8% → 31.2% (Gemini); in SoM mode about +0.7 and −0.7 points.
    - SeeAct-style two-stage prompting gives small or mixed changes.
  - **Applies:** yes. It is evidence that observation modality interacts with the model, and that explicit reasoning
    matters more for text observations.
- **paper:MobileUse, 2025** (2507.16853); `browse:mobile-use` is the related harness.
  - **Idea:** hierarchical reflection:
    - an action reflector (before/after);
    - a trajectory reflector, fired on repeated actions or screens;
    - a global reflector that checks completion when the agent tries to terminate;
    - reflection-on-demand, gated by action-token log-probability;
    - proactive exploration before tasks.
  - **Gain (verified):** Qwen2.5-VL-72B on AndroidWorld, cumulative ablation:

    | Configuration | Success |
    |---|---|
    | Base | 49.5% |
    | + action reflector | 55.2% |
    | + trajectory reflector | 56.1% |
    | + global reflector | 58.6% |
    | + on-demand gating | 61.6% |
    | + exploration | 62.9% |

  - **Applies:** yes. It is the best like-for-like reflection ablation on our benchmark, with a single open model
    (F32, F34, F41, F45).
- **paper:V-Droid, 2025** (2503.15937).
  - **Idea:** the LLM acts as a verifier, scoring every candidate action from a discretised action space.
  - **Gain (abstract):** AndroidWorld 59.5%. The verifier is trained.
  - **Applies:** partly. A prompted, same-model verifier over enumerated candidates is the transferable idea (F38, F23).
- **browse:mobile-use** (MultiAgent config).
  - **Idea:** Operator, Reflector, Progressor, a TrajectoryReflector every 5 steps, a GlobalReflector that can veto the
    first terminate, an AnswerAgent, and `clear_text` / `take_note` actions.
  - **Gain:** the repo's number is not comparable (forked harness).
  - **Applies:** partly (F27, F41, F45, F49).
- **browse:MobileRL.**
  - **Idea:** a screenshot plus a compressed XML tree with 0–999 boxes and `current_app`; 2 screenshots (previous
    plain, current annotated).
  - **Gain:** not comparable.
  - **Applies:** partly (F03, F08).
- **browse:GUIOwl v3 / v3.5.**
  - **Idea:** the model's own `<conclusion>` text as its only history (no second call), and 5 past screenshots as a
    multi-turn chat.
  - **Gain:** not comparable (fine-tuned, coordinate-native).
  - **Applies:** partly. Self-summary (F25) and past images (F08) transfer. Coordinates do not (F52).
- **browse:OpenMobile, browse:UI-Voyager and browse:OS-Genesis.**
  - **Idea:**
    - up to 30 text steps of history (UI-Voyager);
    - `answer` ends the episode;
    - swipe collapsed to a direction scroll;
    - a label → centre dictionary (OS-Genesis).
  - **Gain:** not comparable (fine-tuned models, modified harnesses).
  - **Applies:** partly. History depth (F26) transfers. The rest are harness semantics we keep fixed.
- **browse:EvoSkill-GUI.**
  - **Idea:** skill packages in the system prompt, plus up to 3 rollouts with a verifier that sees the oracle result.
  - **Gain:** not pass@1 (it uses oracle feedback).
  - **Applies:** no (oracle feedback). Skills are F53.

### 1.3 Games (GameBoy)

- **harness:GameBoyRL** (`GameBoyRL/execution`, read 2026-10-13).
  - **Action policies:**
    - `single`: Reasoning, then one action;
    - `scored`: score every listed action 1–5 with a one-line justification, take the argmax;
    - `sequence`: 1–5 actions per call, aborted on a failed high-level action.
  - **History policies:**
    - `none`;
    - `actions`: the last k actions tagged `[no change]`, plus a STUCK_HINT when any recent action changed nothing;
    - `visual`: one extra call per step describes the before/after change in one sentence; same STUCK_HINT.
  - **Done-check:** an extra call after each step, with before/after frames and the last 8 actions; it says "no" unless
    the task is completely accomplished.
  - **Supervisors:**
    - `revision`: critique, hint, retry;
    - `subgoal`: plan, per-step judge, regression check, plan-flaw replanning;
    - `info_subgoal`: a knowledge doc, either retrieved or written by the model from its priors ("parametric").
  - **Strategist-only Pokémon executors:**
    - `GridMoveExecutor`: the VLM classifies 16×16 screen cells into tile tags (ground, ledge, door, NPC, sign…),
      caches them by tile hash, then runs a BFS shortest path;
    - Dialogue, Menu and Combat executors keyed on `pokemon_core.agent_state`, which is RAM-derived and not exposed by
      our env;
    - `FreeExecutor`: one call that returns `Complete: yes/no` and `Press:` together.
  - **`world_model` executor:** a separately trained world model, so not applicable.
  - **Gain (ours):** GameBoyRL's README reports that self-improvement methods (world model, PAE, curiosity docs) do not
    help. Our run gives `single_visual` 36.7% on Red, and `subgoal` +4.1 points (not significant). There are no
    per-arm executor ablations on file.
  - **Applies:** yes. Everything except `world_model`, PAE fine-tuning and `agent_state`.
- **harness:ClaudePlaysPokemon** (secondary sources: Latent Space, LessWrong/GreaterWrong posts, press).
  - **Idea:**
    - a screenshot with a coordinate grid overlay;
    - RAM-derived state;
    - a `navigator` tool that walks to a visible grid coordinate;
    - a persistent knowledge base (notes);
    - context summarisation when context fills;
    - a critic instance.
  - **Gain:** not reported (a live stream with no ablation).
  - **Applies:** partly. The grid overlay, a navigator built on screen-only walkability, notes and summarisation
    transfer. The RAM state does not.
- **harness:GeminiPlaysPokemon, plus paper:Gemini 2.5 report, 2025** (appendix on Pokémon; secondary summaries).
  - **Idea:**
    - a grid overlay with text labels;
    - a text "minimap" of explored tiles (from RAM);
    - a `pathfinder` sub-agent (Gemini simulating a BFS);
    - a boulder-puzzle sub-agent;
    - a goals scratchpad;
    - periodic context summarisation;
    - a critic ("Guide Gemini").
  - **Gain:** no ablation. The report says Gemini read raw Game Boy pixels poorly, and that with RAM text it played
    "roughly as well as without the vision information". Performance degraded beyond about 100k tokens of context
    (repeating past actions). False beliefs in the goals list caused "context poisoning", and the model "panicked" in
    battles.
  - **Applies:** partly. Screen-only versions of the grid, map, pathfinder and scratchpad transfer. The RAM minimap does
    not.
- **paper:PokeAgent Challenge, 2026** (Karten et al., 2603.15563).
  - **Idea:** the Emerald speedrun baseline harness:
    - frames plus limited state;
    - A* pathfinding;
    - an orchestrator with a route plan;
    - sub-agents for battle, reflection, puzzles and objective verification;
    - context compaction.
  - **Gain:** "raw frontier VLMs without harness ≈0% task completion". There are no harness ablations. The paper says
    long-term memory and detailed, iterated plans matter (qualitative).
  - **Applies:** partly. Same as above, minus the RAM state.
- **paper:PokeLLMon, 2024** (Hu et al., 2402.01118).
  - **Idea:** in-context RL from textual battle feedback, knowledge-augmented generation, and "consistent action
    generation" (self-consistency voting) against panic switching.
  - **Gain (abstract):** a 49% ladder win rate and 56% in invited battles. There is no per-component number in the
    abstract.
  - **Applies:** partly. Voting (F37) transfers to battles.
- **paper:Voyager, 2023** (Wang et al., 2305.16291).
  - **Idea:** an automatic curriculum, a skill library of code, and iterative prompting with env feedback, execution
    errors and **self-verification**.
  - **Gain (verified):** 3.3× unique items. Ablations: removing self-verification costs −73% items; a random curriculum
    costs −93%.
  - **Applies:** partly. Self-verification transfers (F41). The skill library is F53 (protocol).
- **paper:Cradle, 2024** (2403.03186).
  - **Idea:** a general computer-control agent with information gathering, self-reflection, task inference, skill
    curation and memory.
  - **Gain:** not reported per module (the abstract has none).
  - **Applies:** partly. Reflection and memory are covered by other rows.
- **paper:BALROG, 2024** (2411.13543).
  - **Idea:** an LLM/VLM game benchmark.
  - **Gain (abstract):** several models do **worse** with visual observations than with text.
  - **Applies:** yes, as evidence for text-first observations (F04).
- **paper:VideoGameBench, 2025** (2505.18134).
  - **Idea:** VG-Agent, a ReAct agent with a textual scratchpad updated every step; Lite mode pauses the game.
  - **Gain (verified):** 0.48% (full) and 1.6% (Lite) for the best models. Includes Pokémon Crystal (test) and Red
    (dev). There are no memory ablations. Qualitative failures: the model overwrote its scratchpad, and in Pokémon
    Crystal lost track of its objective.
  - **Applies:** partly (F28).
- **paper:lmgame-Bench, 2025** (2505.15146).
  - **Idea:** a "gaming harness" with a perception module (screen → symbolic or text state), a memory module (recent
    states, plus reflection lessons) and reasoning.
  - **Gain (verified):**
    - Harness on/off, for example: Sokoban o4-mini 1.3 → 5.3; Tetris o1 13 → 35; 2048 claude-3.5 57.8 → 108.2; Candy
      Crush claude-3.5 17 → 106.
    - Component ablation: perception helps most on spatial games (Sokoban); memory helps most on 2048.
    - The perception module was a separate model (o3) for some games.
  - **Applies:** partly. A text perception module from our VRAM reader is allowed (F04, F06). Memory transfers (F26,
    F28).

### 1.4 General agent methods

- **paper:ReAct, 2023.**
  - **Idea:** interleave reasoning traces and actions.
  - **Gain (recalled, abstract):** ALFWorld +34 points and WebShop +10 points absolute over imitation and RL baselines
    (PaLM-540B).
  - **Applies:** yes. It is already in every baseline, and AndroidLab shows the gain depends on the modality.
- **paper:Reflexion, 2023.**
  - **Idea:** verbal self-reflection stored across **trials**.
  - **Gain (recalled):** ALFWorld +22 points over 12 trials; HumanEval 91% pass@1.
  - **Applies:** partly. We have one attempt, so only the within-episode reflection variant applies (F34). Cross-trial
    reflection is the `revision` supervisor.
- **paper:Self-Refine, 2023.**
  - **Idea:** iterate generate → self-feedback → refine.
  - **Gain (recalled):** about +20 points absolute on average over 7 non-agent tasks.
  - **Applies:** partly (F38).
- **paper:Self-consistency, 2023.**
  - **Idea:** sample several reasoning paths and take a majority vote.
  - **Gain (recalled):** GSM8K +17.9 points.
  - **Applies:** yes (F37).
- **paper:LLMs cannot self-correct reasoning yet, 2024** (Huang et al.).
  - **Idea:** without external feedback, self-correction often turns right answers into wrong ones.
  - **Gain:** negative results on reasoning benchmarks.
  - **Applies:** yes, as a caveat. Self-checks must be grounded in a new observation (the after-screen) to help (F38,
    F41).
- **paper:Plan-and-Solve, 2023 / plan-and-execute.**
  - **Idea:** plan first, then execute the steps.
  - **Gain:** not reported for GUI agents.
  - **Applies:** partly (F21, F47).
- **paper:LATS, 2024.**
  - **Idea:** MCTS with LM value and reflection.
  - **Gain (recalled):** WebShop and HumanEval gains. It needs state reverts.
  - **Applies:** no (F43).
- **paper:MemGPT, 2023.**
  - **Idea:** tiered memory with self-edited notes.
  - **Gain:** not reported for GUI agents.
  - **Applies:** partly (F27, F28).
- **paper:ExpeL, 2024.**
  - **Idea:** cross-task insights extracted from experience.
  - **Gain:** not verified.
  - **Applies:** partly; this is F53, a protocol issue.
- **harness:Gemma 4** (Google Gemma docs; vLLM recipe).
  - **Idea:** gemma-4-31b-it has a thinking mode, turned on with `enable_thinking=True` in the chat template (via
    `chat_template_kwargs` in vLLM).
  - **Gain:** not reported for agents.
  - **Applies:** yes (F24). Watch for known template and parser issues that leak thoughts into the answer.

### 1.5 Our own

- **ours** (`storage/eval/ablation_tables.md`): the supervisor ablation numbers in the header. Every-leg plan
  supervision hurts Web and Android. On GameBoy it is about neutral.
- **ours** (`cusi/state`, `cusi/envs/text_boxes`, `cusi/envs/gameboy_text`):
  - a screen text reader for all envs (GameBoy from VRAM tile maps);
  - a frame/region and trigram novelty archive that could serve as a loop or no-op signal within an episode
    (design plan §7).

---

## 2. Feature catalogue

Envs: W = Web (WebVoyager), A = Android (M3A), G = GameBoy (`single_visual`). Costs are relative to that env's baseline.
For reference, the baselines make 1 call per step on Web, 2 per step on Android (action and summary), and 3 per step on
GameBoy (action, frame-diff and done-check).

| ID | Feature | Layer | Envs | Sources | Failure buckets targeted | Expected cost | Status |
|---|---|---|---|---|---|---|---|
| F01 | Set-of-Mark variant: Web adds an unmarked raw copy or single-colour boxes; Android sends SoM only (drops the raw image) | obs | W A | user; paper:Set-of-Mark 2023; paper:WebVoyager 2024; paper:SeeAct 2024; paper:VisualWebArena 2024; paper:AndroidWorld 2024; paper:AndroidLab 2024; browse:WebVoyager; browse:android_world | GRD, READ | W +1 image/step (~+30% tokens); A −1 image (~−30% image tokens) | candidate |
| F02 | Text-only observation: element list or a11y tree, no screenshot (T3A / `--text_only` style) | obs | W A | user; paper:AndroidWorld 2024; paper:WebVoyager 2024; paper:AndroidLab 2024; browse:android_world (T3A); browse:WebVoyager; browse:WebEvolver | READ, GRD | −all image tokens; +0 calls | candidate (A); W low priority (−19 pt in paper) |
| F03 | Richer or pruned structured text: pruned a11y tree or DOM with hierarchy and states (W); element list plus bbox, class and current app (A) | obs | W A | user; paper:AgentOccam 2024; paper:Agent-E 2024; paper:AutoDroid 2023; browse:SynthAgent; browse:MobileRL; browse:android_world (T3A) | GRD, READ, PLAN | +20–60% prompt tokens; +0 calls | candidate |
| F04 | Screen text block (`text_boxes`) next to the image every step | obs | W A G | user; ours; paper:lmgame-Bench 2025; paper:Gemini 2.5 report 2025; paper:BALROG 2024 | READ, PLAN, EARLY | +5–20% tokens; +0 calls | candidate |
| F05 | GameBoy grid overlay: 16 px cell lines with row/column labels and the player's cell marked, plus a legend in the prompt | obs | G | user; harness:GameBoyRL (perception/grid); harness:ClaudePlaysPokemon; harness:GeminiPlaysPokemon | GRD, LOOP | +0 calls, +~0 tokens (image edit) | candidate |
| F06 | GameBoy screen tile map: VLM-classified cell tags (walkable / blocked / door / NPC / sign / item…) cached by tile hash, rendered as a text grid | obs | G | harness:GameBoyRL (perception/tiles, strategist); harness:ClaudePlaysPokemon; harness:GeminiPlaysPokemon; paper:lmgame-Bench 2025 | GRD, LOOP, STEPS | first sight of a tile: +1 batched call per unknown tile; after caching about +0; +10–20% tokens | candidate (expensive to build) |
| F07 | Resolution and zoom: GameBoy frame upscaled ×3–4 nearest-neighbour; W/A zoom-crop re-look at a region on request | obs | W A G | user; paper:RegionFocus 2025 | READ, GRD | G: +image tokens only; W/A: +1 call when triggered | candidate |
| F08 | Past screenshots in context (k images; W baseline k=3, A baseline 0 in the action call, G baseline 0) | obs | W A G | user; paper:WebVoyager 2024; browse:GUIOwl v3.5; browse:MobileRL; browse:GUI-Owl-1.5 web; paper:VideoGameBench 2025 | LOOP, READ | +1 image per extra k (~+25–35% tokens each) | candidate |
| F09 | Programmatic change report: diff of element lists (new elements marked `*`, URL/title change) on W/A; frame_changed plus a `text_boxes` diff on G | obs | W A G | user; paper:Agent-E 2024; harness:browser-use; browse:WebCoach; harness:GameBoyRL (`[no change]` tags) | LOOP, GRD, EARLY | +0 calls; +~5% tokens | candidate |
| F10 | Task and step budget restated each step ("Task: … Step k of N, N−k left") | obs | W A G | user; harness:browser-use; browse:WebCoach | PLAN, STEPS | +0 calls; +<2% tokens | candidate |
| F11 | Forced or nudged finish near the budget end: on the last 1–2 steps only ANSWER (W) or answer/status (A) is allowed, with a "give your best answer now" prompt | action | W A | harness:browser-use; browse:WebCoach; browse:GUI-Owl-1.5 web; ours (159/429 W episodes truncated) | STEPS | +0 calls | candidate (G n/a: no answer action) |
| F12 | Composite or extra primitive actions: W `GoTo [url]`, `Select [n]; option`, `PressKey`; A `clear_text`, `type_and_enter` | action | W A | user; harness:browser-use; browse:SynthAgent; browse:mobile-use; paper:Agent-E 2024; paper:AgentOccam 2024 | GRD, STEPS, LOOP | +0 calls; +~3% tokens | candidate |
| F13 | Page-text extraction action: `Extract [query]` sends the page's full visible text (innerText, truncated) to the same model and returns the excerpt as the observation | action | W | harness:browser-use; browse:WebCoach; paper:Agent-E 2024 (text_only DOM view) | READ, STEPS | +1 call when used (~+40% tokens on that step) | candidate |
| F14 | Action-space pruning: drop actions with no value on this benchmark (W `Wait`; A `double_tap`, `long_press` unless needed), or make scrolling page-level only | action | W A | paper:AgentOccam 2024; harness:browser-use | LOOP, FMT, STEPS | +0; slightly fewer tokens | candidate |
| F15 | Coordinate fallback: allow `Click (x,y)` / `click{x,y}` when the target has no mark | action | W A | user; paper:AppAgent 2023 (against); browse:GUIOwl; browse:OS-Genesis | GRD | +0 calls | candidate (low prior: the model is not coordinate-trained) |
| F16 | Multi-action decisions: 1–k actions per call, aborted on an invalid action or no change | action | W A G | user; harness:GameBoyRL (sequence); harness:browser-use | STEPS (calls), LOOP | −30–60% calls per step | candidate |
| F17 | GameBoy macros built from screen state: `walk <dir> <n>` (stop early on frame unchanged), `advance_dialogue` (press A until `text_boxes` shows no dialogue box, max m), `press <btn> <n>` | action | G | user; harness:GameBoyRL (PassDialogue/MoveSteps high-level actions); harness:ClaudePlaysPokemon; harness:GeminiPlaysPokemon | STEPS, LOOP, FMT | −50–80% calls; env steps unchanged (each press still counts) | candidate |
| F18 | GameBoy navigate-to-cell tool: `goto (r,c)` runs a BFS over the F06 map and issues the presses | action | G | harness:GameBoyRL (GridMoveExecutor, navigation.py); harness:ClaudePlaysPokemon (navigator); harness:GeminiPlaysPokemon (pathfinder); paper:PokeAgent 2026 (A*) | GRD, LOOP, STEPS | needs F06; −calls during walking | candidate (needs F06, F05) |
| F19 | Structured per-step reasoning: *screen description → did the last action work / what changed → progress vs. task → next sub-goal → action* | reasoning | W A G | user; harness:browser-use (evaluation_previous_goal/next_goal); paper:Aguvis 2024; paper:UI-TARS 2025; paper:SeeAct 2024; paper:ReAct 2023; paper:AndroidLab 2024 | PLAN, LOOP, READ, EARLY | +0 calls; +30–80% output tokens | candidate |
| F20 | Output format: strict JSON with named fields vs. free text (W text, A `Reason/Action{json}`, G `Reasoning/Action`) | reasoning | W A G | user; harness:browser-use; paper:AndroidWorld 2024 | FMT | +0 calls | candidate |
| F21 | Persistent plan/todo field inside the executor response: written at step 1, carried and updated each step, shown in the prompt | reasoning | W A G | user; harness:browser-use (todo); paper:AgentOccam 2024 (plan); paper:Mobile-Agent-v2 2024 (planning agent); paper:Agent S2 2025 (PHP); paper:VideoGameBench 2025; harness:GeminiPlaysPokemon (goals); paper:Plan-and-Solve 2023 | PLAN, STEPS, EARLY | +0 calls; +~10% tokens | candidate |
| F22 | Two-stage generate-then-ground: call 1 states the intended target in words; call 2 picks the index from the list | reasoning | W A | paper:SeeAct 2024; paper:AndroidLab 2024; browse:android_world (SeeAct) | GRD | +1 call/step | candidate (low prior) |
| F23 | Score-all-actions policy: rate every action 1–5 with a one-line reason and take the argmax (G; on A only the top-N candidate elements) | reasoning | G (A) | harness:GameBoyRL (scored); paper:V-Droid 2025 (trained verifier) | GRD, LOOP | +~2× output tokens; +0 calls | candidate |
| F24 | Thinking mode (Gemma 4 `enable_thinking`), or a larger per-step reasoning budget | compute | W A G | harness:Gemma 4; paper:TTI 2025 (budget forcing plateaus) | PLAN, READ, GRD | +2–5× output tokens; latency | candidate |
| F25 | History form: the model's own one-line conclusion in the action response, replacing the extra summary/frame-diff call (A: M3A summary; G: `visual`), or vs. raw actions | memory | A G (W) | user; paper:AndroidWorld 2024; browse:GUIOwl v3 (`<conclusion>`); harness:browser-use (memory field); harness:GameBoyRL (`actions` vs `visual`) | LOOP (any) | A −1 call/step; G −1 call/step | candidate |
| F26 | Text history depth k (G `history_k` 5 → 10/20; W: summaries of turns older than 3 instead of "(Omitted)"; A: last-k summaries vs all) | memory | W A G | user; browse:UI-Voyager (30 steps); browse:MA-v3 (last 5); paper:AgentOccam 2024 (selective replay); paper:Gemini 2.5 report 2025 (long context hurts) | LOOP, PLAN | ±10–30% tokens | candidate |
| F27 | Notes buffer for facts found during the task (`Note: …` field or `take_note` action), persisted and shown every step and at ANSWER time | memory | W A G | user; browse:mobile-use (take_note); browse:MA-v3 (Notetaker); paper:Mobile-Agent-v2 2024 (memory unit); paper:Mobile-Agent-E 2025; paper:AgentOccam 2024; paper:MemGPT 2023 | READ, EARLY, PLAN | +0 calls; +~5% tokens | candidate |
| F28 | GameBoy objectives-and-places log written by the model: current objective, places seen, what NPCs and signs said (from `text_boxes`), updated each step or every m steps | memory | G | user; harness:GeminiPlaysPokemon; harness:ClaudePlaysPokemon; paper:VideoGameBench 2025; paper:PokeAgent 2026; harness:GameBoyRL (strategist locations/notepad) | PLAN, LOOP | +0 (field) or +1 call/m steps | candidate |
| F29 | Prior-knowledge doc: a per-site, app or game tips doc the same model writes from its priors (parametric), or one from our exploration runs | memory | W A G | paper:AppAgent 2023; paper:AutoDroid 2023; paper:Mobile-Agent-E 2025; harness:GameBoyRL (info_subgoal parametric/retrieval); paper:MobileUse 2025 (exploration); ours (§7) | PLAN, GRD | +1 call per site/app/game (cached); +~10% tokens | candidate (parametric); exploration doc needs a fairness check (§7) |
| F30 | Generic env guidance block: hand-written, task-agnostic tips (W: M3A-GUIDANCE style for web — use search boxes, check filters, verify dates; A: extra tips; G: controls, facing, menus) | reasoning | W A G | paper:AndroidWorld 2024 (M3A guidelines); browse:SynthAgent (15 rules); harness:GameBoyRL (STUCK_HINT) | PLAN, GRD, LOOP | +0 calls; +~5% tokens | candidate (generic only; the dev-overfit risk is noted) |
| F31 | No-op detection: if the screen and element list are unchanged after an action, inject "your last action had no effect; do not repeat it" and ban an identical repeat | recovery | W A G | user; harness:GameBoyRL (STUCK_HINT); paper:MobileUse 2025; paper:Agent-E 2024 | LOOP | +0 calls | candidate |
| F32 | Loop detection over recent (state hash, action) pairs: a revisited state with the same action ≥2× triggers a reflection prompt (or call) asking for a different approach | recovery | W A G | user; paper:MobileUse 2025 (trajectory reflector); browse:mobile-use; paper:WebVoyager 2024 (navigation stuck 44%); paper:VisualWebArena 2024; ours (novelty archive, §7) | LOOP, STEPS | +0 (prompt) or +1 call when fired | candidate |
| F33 | Invalid-output repair: re-ask immediately with the parse error, at most r times, before giving up | recovery | W A G | user; harness:GameBoyRL (error block); browse:WebVoyager; paper:AndroidWorld 2024 | FMT | +1 call per invalid | candidate (keep harness step accounting: a re-ask still uses its step) |
| F34 | Action reflector call: before/after screens → outcome A (as intended) / B (wrong screen) / C (no effect) plus a fix hint into the next prompt, optionally only when triggered (F31/F32 or low confidence) | recovery | W A G | paper:Mobile-Agent-v2 2024; paper:Mobile-Agent-E 2025; browse:MA-v3 (ActionReflector); paper:MobileUse 2025; paper:Reflexion 2023; harness:GameBoyRL (`visual`) | LOOP, GRD, EARLY | +1 call/step (always) or +0.1–0.3 (on demand) | candidate (A, G already partly have it) |
| F35 | Backtrack on a detected wrong turn: W `GoBack` or restore the last good URL; A `back`, or `home` + `open_app`; G `B` until the overworld | recovery | W A G | user; paper:WILBUR 2024; paper:Agent-E 2024; browse:MA-v3 (reflector B/C) | LOOP, PLAN | +0–1 calls (trigger from F34) | candidate (depends on F34) |
| F36 | Specific execution-error messages: replace WebVoyager's generic "cannot be executed" with the cause (index out of range / element not interactable / page unchanged) | recovery | W (A) | browse:WebVoyager; browse:android_world (M3A drops execution errors) | FMT, GRD | +0 | candidate |
| F37 | Self-consistency action vote: sample k actions (temperature >0) and execute the majority parsed action | compute | W A G | user; paper:Self-consistency 2023; paper:Tree search, Koh 2024 (count aggregation); paper:PokeLLMon 2024 | GRD, FMT, LOOP | k× action calls (k=3–5) | candidate |
| F38 | Pre-execution critic: the same model sees the proposal plus the screen and accepts or rejects it with a reason; on reject, re-propose once with the critique | compute | W A G | user; paper:Self-Refine 2023; paper:V-Droid 2025; paper:Tree search, Koh 2024 (value fn); paper:Huang self-correct 2024 (caveat) | GRD, EARLY, PLAN | +1 call/step (+1 on reject) | candidate |
| F39 | Outcome-simulation lookahead: for the top-k candidates, the model predicts the next screen in text and scores progress; the best is executed | compute | W A G | paper:WebDreamer 2024; browse:WebEvolver (world-model search) | PLAN, GRD, LOOP | +2k calls/step (k=3: +6) | candidate (expensive) |
| F40 | Best-of-k with a comparative same-model judge: k proposals are shown as multiple choice and the judge picks one | compute | W A G | user; paper:GTA1 2025; paper:Agent S3/BJudge 2025 | GRD, PLAN | k+1 calls/step | candidate |
| F41 | Completion self-check before finishing: W before ANSWER, A before `status: complete`, G change the done-check (strictness, cite `text_boxes` evidence, 2-of-3 vote); "check again" re-prompt with the current screen | compute | W A G | user; paper:TTI 2025; paper:MobileUse 2025 (global reflector); browse:mobile-use (GlobalReflector); harness:GameBoyRL (done_check); paper:Voyager 2023 (self-verification); paper:Huang self-correct 2024 | EARLY | +1–2 calls per finish attempt (W/A); G ±1 call/step | candidate |
| F42 | Trajectory-level best-of-N: N full rollouts, then a comparative judge over behavior narratives picks the answer | compute | W | paper:Agent S3/BJudge 2025 | all (variance) | N× episodes +1 judge call | candidate (W only; A/G n/a: success is graded on the final env state, which can't be picked) |
| F43 | Tree search with state reset (best-first/MCTS) | compute | — | paper:Tree search, Koh 2024; paper:ExACT 2024; paper:LATS 2024; browse:WebEvolver | — | ≫10× calls | n/a: W live sites and A emulator can't be reset per step cheaply; G save-state lookahead is privileged emulator access (ask user if wanted) |
| F44 | Merged done-check + action call: the action call also outputs `Complete: yes/no`, replacing the separate done-check | compute | G | harness:GameBoyRL (FreeExecutor) | EARLY (cost) | −1 call/step | candidate |
| F45 | Event-triggered supervisor: it intervenes only on stall, loop or no-progress signals (F31/F32, k steps without new screens), not on every leg | supervisor | W A G | user; paper:MobileUse 2025; browse:mobile-use (TrajectoryReflector); browse:WebCoach (intervene flag); harness:GeminiPlaysPokemon (critic); ours (every-leg supervision hurts W/A) | LOOP, PLAN | +1–2 calls per event | candidate |
| F46 | Supervisor input as a behavior narrative (per-step fact lines from before/after) vs. full history or raw screenshots | supervisor | W A G | user; paper:Agent S3/BJudge 2025; browse:WebCoach (condenser) | PLAN, EARLY | ±tokens; narrative +0 if reused from F34/M3A summaries | candidate |
| F47 | Soft plan: the supervisor writes the plan once (and revises it after each completed sub-goal), and the executor sees it as a checklist with no leg boundaries or per-leg judging | supervisor | W A G | user; paper:Agent S2 2025 (PHP); paper:Mobile-Agent-v2 2024; browse:MA-v3 (Manager); paper:Agent-E 2024; harness:GameBoyRL (subgoal); ours (hard legs hurt W/A) | PLAN, STEPS | +1 call + 1 per revision | candidate |
| F48 | Termination veto: the supervisor reviews the first finish (narrative + final screen) and may send the executor back with a reason | supervisor | W A G | browse:mobile-use (GlobalReflector veto); paper:MobileUse 2025; paper:PokeAgent 2026 (objective verification) | EARLY | +1 call per finish | candidate (overlaps F41; keep the one that wins) |
| F49 | Answer rewrite: a final call rewrites the ANSWER to answer exactly what was asked, using the notes and the last screenshots | compute | W (A answer tasks) | browse:WebEvolver (final answer rewrite) | EARLY, READ | +1 call/task | candidate |
| F50 | Within-episode novelty archive as the loop/no-op signal for F31/F32/F45 | recovery | W A G | ours (`cusi/state`, design plan §7) | LOOP | +0 calls; CPU only | candidate (finalisation stage only, §7) |
| F51 | Task- or app-specific prompt hacks and auto-dismissal of popups or consent dialogs | — | — | browse:MA-v3 (goal-string hacks, Chrome dialog auto-dismiss); browse:mobile-use (knowledge file) | ENV | — | n/a: task-specific, or changes the fixed harness |
| F52 | Separate grounding/OCR/icon models or coordinate-native fine-tuned policies | — | — | paper:OS-Atlas 2024; paper:CogAgent 2023; paper:UI-TARS 2025; paper:Aguvis 2024; paper:Agent S2 2025 (MoG); paper:Mobile-Agent v1/v2/E; browse:GUIOwl, OpenMobile, UI-Voyager, OS-Genesis | — | — | n/a: one fixed model, no fine-tuning |
| F53 | Cross-task experience memory (workflows, tips, skills, reflections reused across test tasks) | memory | W A G | paper:Agent Workflow Memory 2024; paper:ExpeL 2024; paper:Reflexion 2023; paper:Mobile-Agent-E 2025; paper:Agent S 2024; paper:Voyager 2023; browse:WebCoach; browse:EvoSkill-GUI | PLAN | +retrieval tokens | n/a for screening: it breaks per-task independence; possible only with dev/exploration-sourced memory and a held-out report (raise with user) |
| F54 | GameBoy mode detection from the screen (dialogue / menu / battle / overworld, from `text_boxes` and frame cues), which switches mode-specific instructions and highlights the relevant buttons | obs | G | harness:GameBoyRL (strategist Dialogue/Menu/Combat executors, which use RAM `agent_state` — we use the screen instead); paper:lmgame-Bench 2025; harness:GeminiPlaysPokemon | PLAN, LOOP, GRD | +0 calls (rule-based on `text_boxes`) | candidate |

---

## 3. Per-feature notes: evidence and implementation

Hook points referred to below:

| Hook | Where | Role |
|---|---|---|
| `decide()` | `cusi/agents/executors/{webvoyager,m3a,gameboy}.py` | prompt build and parse |
| `after_step()` / `after_invalid()` | same files | memory updates |
| `done_check()` | same files | completion check (GameBoy) |
| `Executor.run()` | `cusi/agents/executors/base.py` | the loop |
| `build_step_prompt` | `cusi/agents/executors/gameboy.py` | the GameBoy prompt |
| `ACTION_POLICIES` / `HISTORY_POLICIES` | `cusi/agents/executors/gameboy.py` (from GameBoyRL) | the GameBoy action and history arms |

Prototype wrappers subclass these executors and read flags from `--extra` (design plan §4).

**F01 Set-of-Mark variant.**
- **Evidence:** weak to medium, and mixed by model. VisualWebArena: SoM +1.3 points. AndroidWorld: SoM −5.2 points
  (GPT-4T) and +3.4 (Gemini). AndroidLab: SoM +5.8 (GPT-4o) and −2.2 (Gemini). SeeAct: SoM "not effective" on web.
- **W:** in `_observation_message`, append the unmarked frame (`info["raw_frame"]` if the env keeps it; otherwise add a
  boxes-off screenshot in `cusi/envs/webvoyager`) as a second image before the marked one. Alternatively, force
  `fix_box_color` (black boxes).
- **A:** in `M3AExecutor.decide`, pass `images=[obs["frame"]]` only (drop `info["raw_frame"]`).
- **G:** n/a.

**F02 Text-only observation.**
- **Evidence:** medium. WebVoyager: text-only is −19 points (GPT-4V). AndroidWorld: text-only is +5.2 (GPT-4T) and −3.4
  (Gemini). This is model-dependent, so it must be measured for gemma.
- **A:** call `self.call(..., images=[])`; M3A's prompt already has the element list. Drop "screenshot" wording from
  GUIDANCE (as T3A should have). The summary call stays multimodal or also goes text-only (a sub-switch).
- **W:** use the env's `--text_only` a11y-tree path (`utils_webarena.py`) as the observation text. The screenshot is
  still saved for the judge.
- **Priority:** low on W, given −19 points.

**F03 Richer or pruned structured text.**
- **Evidence:** medium. AgentOccam's observation simplification gave +4.7 points (30.0 → 34.7, GPT-4-T, WebArena dev).
  Agent-E and AutoDroid have no isolated ablation.
- **W:** build a pruned a11y tree via CDP (the code exists in WebVoyager's `utils_webarena.py`). Keep nodes ≥60% in the
  viewport with role, name, value and state (checked, expanded, selected), and map node ids to the SoM indices. Append
  it under the element list, capped at about 3k tokens.
- **A:** in `ui_elements` rendering, add `bbox_pixels`, `class_name`, `resource_id` and `is_scrollable`, and prepend
  `Current app: <package>` (all available from `State.ui_elements` and the forest).
- **G:** n/a (F04 and F06 instead).

**F04 `text_boxes` text block.**
- **Evidence:**
  - lmgame-Bench: the perception module gives large gains (for example Sokoban 1.3 → 5.3), though the module was a
    separate model for some games.
  - Gemini Pokémon: vision added little once RAM text was present (anecdotal).
  - BALROG: vision often hurts.
  - Overall medium for G; weak for W/A, which already get element text.
- **G:** in `build_step_prompt`, add a "Text on screen:" section rendering `obs["text_boxes"]` (region kind plus
  string, reading order). This is already available from `cusi/envs/gameboy_text`. Also pass it to the done-check
  prompt.
- **W/A:** render `obs["text_boxes"]` as a compact "Visible text" section after the element list. It is mainly useful
  for non-interactive text (prices, dates, results) that the element list omits.

**F05 GameBoy grid overlay.**
- **Evidence:** weak (shipped in the Claude and Gemini harnesses, no ablation).
- **G:** draw a 10×9 grid of 16 px cells on an upscaled copy of `obs["frame"]` (reuse
  `GameBoyRL/execution/perception/pokemon/grid.py:split_grid` and `draw_box`). Label rows and columns, and mark the
  player cell (fixed at screen centre in Pokémon overworld) in the legend.
- **Prompt change:** "the player stands in cell (4,4); moving one cell = one direction press".
- Apply it only when F54 says overworld.

**F06 GameBoy screen tile map.**
- **Evidence:** weak (GameBoyRL strategist, Claude/Gemini harnesses; no ablation, and the Claude/Gemini maps came from
  RAM).
- **G:** port `perception/pokemon/tiles.py`:
  - classify each unknown cell with `CLASSIFY_PROMPT` (batched in one `vlm.infer` call per step);
  - cache by tile-image hash across the episode (and across tasks of the same game: a tile cache is not task
    knowledge);
  - render a 9-row text grid (`.` walkable, `#` blocked, `D` door, `N` NPC, `S` sign, `P` player).
- Uses the screen only; no RAM.

**F07 Resolution and zoom.**
- **Evidence:** weak for us. RegionFocus reports +24% on WebVoyager, but on coordinate grounders.
- **G:** upscale 160×144 to 640×576 nearest-neighbour before sending (in `decide`), so pixel-font text is legible.
  Check the vLLM image-token budget.
- **W/A:** add a `Zoom [n]` action, or an automatic re-look. Crop the bbox of element n (or a quadrant) ×2 and return
  it as the next observation image without an env step. The crop costs a call, not an env step; record it as a
  scripted step.

**F08 Past screenshots.**
- **Evidence:** weak. There is no agent ablation of k. GUI-Owl-1.5 (5 images), MobileRL (2) and WebVoyager (3) are
  design choices only.
- **W:** change `MAX_ATTACHED_IMGS` (1/3/5).
- **A:** add the previous step's SoM frame as a first image labelled "previous" in `decide` (keep `self._before`).
- **G:** add the last 1–3 frames (from `report.env_steps[-k:]`) with labels in `decide`.

**F09 Programmatic change report.**
- **Evidence:** weak (Agent-E is qualitative; browser-use has no ablation).
- **W:** in `after_step`, diff `obs_before["texts"]["web_elements"]` against `obs_after` by (tag, text). Prefix new
  entries with `*` in the next element list, and add a line "URL changed: a → b" or "page unchanged".
- **A:** the same over `ui_elements` (text and content_desc), injected into the next M3A history line or the prompt.
- **G:** reuse `info["frame_changed"]`, and add a `text_boxes` diff ("new text: …").

**F10 Task and step budget each step.**
- **Evidence:** weak.
- **W:** append `Task reminder: <task>. Step k of 15.` to `OBS_PROMPT` in each observation message.
- **A:** add `Step k of N` above the history in `action_prompt` (N = `max_steps` from `run()`).
- **G:** `build_step_prompt` already has the task. Add `Step k of 175, n left` (the hint block's STEP_INFO only shows
  with a hint).

**F11 Forced or nudged finish near the budget end.**
- **Evidence:** weak; shipped in browser-use and GUI-Owl-web with no ablation. But mechanically W scores a missing
  ANSWER as 0, and 159/429 W baseline episodes end truncated.
- **W:** when `self._it >= 14`, add "This is your last step: you must reply with ANSWER; <best answer>". If the reply
  isn't ANSWER, parse a fallback answer from the Thought.
- **A:** at `N−1`, tell it to use `answer` (if the task asks a question) and then `status`.
- Pair with F27 so the answer has material.

**F12 Composite or extra actions.**
- **Evidence:** medium for alignment in general (AgentOccam's action-space changes gave +11.6 points). There is no
  isolated evidence for GoTo/Select on WebVoyager.
- **W:** extend `extract_information` and the env's executor with:
  - `GoTo [url]` (`driver.get`);
  - `Select [n]; option` (Selenium `Select`);
  - `PressKey [key]`.
  Add one line each to SYSTEM_PROMPT.
- **A:** add `clear_text` (select-all + delete via adb) as a JSONAction extension, in a wrapper that converts it to a
  long_press + delete sequence or `input_text` with a clear flag.

**F13 Page-text extraction (W).**
- **Evidence:** weak (browser-use, no ablation).
- **W:**
  1. `Extract [question]` makes the env return `document.body.innerText`, truncated to about 8k tokens.
  2. The executor makes one call: "From this page text, answer: <question>; quote the lines".
  3. The result becomes the next observation's text, like the existing PDF path.
- It counts as one env step.

**F14 Action-space pruning.**
- **Evidence:** medium. AgentOccam: action reduction 14.2 → 25.8, then no scrolling → 30.0 (GPT-4-T, WebArena dev).
- **W:** remove `Wait` from SYSTEM_PROMPT. Optionally replace `Scroll [n]` with page-level `Scroll WINDOW` only.
- **A:** remove `double_tap` and `long_press` from the M3A action list (M3A doesn't list double_tap; check the usage
  rates in the baseline first). Measure the frequency of each action in baseline episodes before choosing what to drop.

**F15 Coordinate fallback.**
- **Evidence:** weak and against us. AppAgent: raw coordinates 2.2% vs labels 48.9% (GPT-4V).
- **W:** `Click (x,y)` in 1024×768 screenshot pixels.
- **A:** allow `{"action_type":"click","x":..,"y":..}` (JSONAction already supports it).
- Prompt rule: use it only for unmarked targets.

**F16 Multi-action decisions.**
- **Evidence:** weak. GameBoyRL `sequence` has no recorded ablation here; browser-use allows up to 10 actions per step,
  with no ablation.
- **G:** use the existing `sequence` action policy (arm `sequence_visual`). The parity test already exists.
- **W:** allow `Action: Type [3]; text ⏎ Click [7]`, with up to 3 per reply. The executor queues them and stops when
  the element list changes unexpectedly.
- **A:** M3A JSON list of up to 3 actions.

**F17 GameBoy macros.**
- **Evidence:** weak. The harnesses rely on such tools, and the GameBoyRL high-level actions exist but are not in the
  low_level controller.
- **G:** in a wrapper executor, parse `walk up 4`, `press a 3` and `advance_dialogue` into repeated `env.step("up")`
  calls inside one decision (`_run_decision` already loops a decision's actions).
  - `walk` stops when `frame_changed` is false.
  - `advance_dialogue` presses `a` while `text_boxes` contains a dialogue-region box, up to 10.
- Each press counts toward the 175 steps.

**F18 GameBoy navigate-to-cell.**
- **Evidence:** weak (the Claude navigator and Gemini pathfinder are anecdotal; GameBoyRL GridMove has no ablation).
- **G:** with F05 and F06, add `goto (r,c)`. Run BFS on the F06 grid (`navigation.py:shortest_path`), then issue the
  direction presses, re-observing after each press and stopping on no-change or on a dialogue appearing.

**F19 Structured reasoning template.**
- **Evidence:** medium. AndroidLab: ReAct +8 and +12 points in XML mode, but about ±0.7 in SoM mode. UI-TARS and Aguvis
  thought patterns are trained.
- **W:** replace "Thought:" guidance in SYSTEM_PROMPT with
  `Thought: Observation: … Last action result: … Progress: … Next: …`. Keep the `Action:` line unchanged (the parser
  splits on it).
- **A:** extend M3A's `Reason:` instructions in the same way.
- **G:** change `SingleActionPolicy.response_format` to
  `Screen: … / Last action effect: … / Progress: … / Reasoning: … / Action:`.

**F20 Output format.**
- **Evidence:** weak; no head-to-head on GUI agents found.
- **W:** demand `{"thought":…, "action":…}` and convert it back to WebVoyager's action string before
  `extract_information`.
- **G:** a JSON policy variant.
- **A:** already JSON for the action. Wrap the whole reply in JSON.
- Watch FMT rates.

**F21 Persistent plan/todo field.**
- **Evidence:** medium.
  - Mobile-Agent-v2: removing the planning agent hurts most (88.6 → about 59 basic).
  - Agent S2: proactive replanning +4.6/+6.2.
  - AgentOccam: planning +7.4.
  - All are separate calls or multi-agent setups. The in-response variant is unablated.
- **All envs:** require a `Plan:` block (≤5 checklist lines, with `[x]` marks) in each reply. The executor stores the
  latest plan and re-inserts it into the next prompt.
  - **W:** in the observation message, because older assistant turns are kept anyway; insert it only in the latest
    turn to save tokens.
  - **A:** above the history in `action_prompt`.
  - **G:** a `[CONTEXT_SECTION]` addition.

**F22 Two-stage generate-then-ground.**
- **Evidence:** medium and negative to mixed. AndroidLab SeeAct-mode: −0.7 to +4.3 points. SeeAct says grounding is the
  bottleneck.
- **W/A:**
  1. Call 1 (current prompt) answers with an intended target description, with no index.
  2. Call 2 shows the element list (plus SoM) and asks "which index matches '<description>'; reply with the index or
     NONE".

**F23 Score-all-actions.**
- **Evidence:** weak (GameBoyRL `scored` arm, not ablated here; V-Droid's verifier is trained).
- **G:** use the existing `scored` policy (arm `scored_visual`).
- **A:** M3A first proposes the top 5 candidate (action, index) pairs; one call scores them 1–5. It is merged with F40
  if both are tried.

**F24 Thinking mode.**
- **Evidence:** weak. TTI: per-step budget forcing plateaued (Gemma 3 12B), and there is no agent ablation for Gemma 4.
- **All envs:** pass `chat_template_kwargs={"enable_thinking": True}` through `AgentVLM` for the action call only. Strip
  the thought channel before parsing and before storing history (the guidance warns against feeding thoughts back).
- Check the vLLM `--reasoning-parser gemma4` behaviour first.

**F25 History form.**
- **Evidence:** medium for the reflective summary as a whole. M3A-Simple (no guidelines, no reflection) scores 19.8
  against M3A's 30.6 with GPT-4T. Self-written conclusions have no ablation.
- **A:** drop the summary call and require M3A's reply to end with `Summary: <what this action should achieve and what
  to check next>`. In `after_step`, append `Action selected: … Outcome: <screen changed / unchanged> <Summary>`.
- **G:** arm `single_actions` (the existing `actions` history), or `single_visual` with the frame-diff replaced by the
  model's own `Expected effect:` field compared against `frame_changed`.
- **W:** n/a (raw chat is kept).

**F26 History depth.**
- **Evidence:** weak. The Gemini report says very long context degrades; AgentOccam's selective replay gives +2.1.
- **G:** `history_k` of 5, 10 or 20 (an existing constructor argument).
- **A:** render only the last k M3A summaries, plus "Steps 1..j omitted".
- **W:** in `clip_message_and_obs`, replace omitted observations with a one-line summary of that turn's Thought instead
  of "(Omitted in context.)". The thought text is already present, so this is a formatting choice.

**F27 Notes buffer.**
- **Evidence:** weak to medium. Mobile-Agent-v2's memory unit is within its ablation, but the row mapping is unclear;
  MA-E has no reflector or notetaker ablation.
- **W/A:** add an optional `Note: <fact>` line to the reply format, parsed in `decide`. Keep a deduplicated list, render
  it as "Facts found so far:" each step, and in the forced-answer prompt (F11).
- **G:** `Note:` for dialogue facts (where the shop is, what the NPC asked).

**F28 GameBoy objectives-and-places log.**
- **Evidence:** weak; anecdotal in all game harnesses. VideoGameBench reports scratchpad failures (overwrites).
- **G:** a `Log:` block in the reply. Keep 3 fields (`Objective`, `Places seen`, `Learned`) and show them each step.
  Updates are allowed only as append or edit of a field, so the model can't wipe them. Optionally refresh every m=10
  steps with a separate call from the last frames and `text_boxes` history.

**F29 Prior-knowledge doc.**
- **Evidence:** medium for exploration docs (AppAgent: 48.9 → 73.3 with auto-exploration docs, GPT-4V; MobileUse
  exploration +1.3). Parametric docs: GameBoyRL's `info_subgoal_parametric` is the control arm and showed no gain in
  its README.
- **W:** once per site, call "Write 10 tips for completing tasks on <site> (search, filters, navigation)", cache it,
  and prepend it to the first message.
- **A:** the same per app, chosen by `open_app` target or current package.
- **G:** reuse `info_subgoal_parametric` doc generation, fed to the executor instead of the supervisor.
- The exploration-sourced variant waits for §7.

**F30 Generic env guidance.**
- **Evidence:** medium for M3A guidelines plus reflection together (−10.8 points without both, GPT-4T). The parts are
  not separated.
- **W:** add a GUIDANCE block to SYSTEM_PROMPT, for example:
  - use site search before browsing;
  - apply all constraints (date, price, rating) via filters;
  - verify the final page shows the answer;
  - don't answer from memory.
- **G:** add facing/interact tips and menu-navigation tips to `STEP_PROMPT`.
- Write the guidance before looking at dev failures in detail, to limit overfitting.

**F31 No-op detection.**
- **Evidence:** weak to medium. MobileUse's trajectory reflector (repeat-triggered) gave +0.9. GameBoyRL already has a
  STUCK_HINT; W and A have nothing.
- **W:** in `after_step`, if the SoM screenshot hash and element list are equal before and after, set
  `self._warn_obs = "Your last action had no visible effect. Do not repeat it; try a different element or action."`
  (the warning slot already exists).
- **A:** append the same to the M3A summary line.
- **G:** reject an identical repeat after 2 no-change presses (convert it to invalid with that message via
  `env_invalid_to_decision`-like logic).

**F32 Loop detection.**
- **Evidence:** weak to medium. MobileUse gave +0.9 (combined with repeat triggers). WebVoyager says navigation stuck
  is 44% of failures.
- **All envs:** keep a deque of `(state_key, action_text)`.
  - state_key: W = URL + element-list hash; A = activity + element hash; G = downsampled frame hash, or later the F50
    novelty archive.
- When a pair repeats ≥2 times within 6 steps, either:
  - (a) inject a reflection instruction into the next prompt (+0 calls); or
  - (b) make one reflection call over the last 6 steps that returns "what is going wrong, what to try instead", and
    inject its answer.

**F33 Invalid-output repair.**
- **Evidence:** weak (standard practice). The baselines already return the error on the next turn.
- **All envs:** on a parse failure, re-ask in the same `decide()` with the error appended (at most 2 times) before
  returning `Decision(invalid=…)`.
- Note that this changes step accounting. The base loop counts each invalid as a step, and harness parity (§0) says
  keep it. So count re-asks as steps too, or record this as a deliberate deviation in `plans/decisions.md`.

**F34 Action reflector.**
- **Evidence:** medium to strong across papers. MobileUse action reflector +5.7 and on-demand gating +3.0 (Qwen-72B,
  AndroidWorld). The Mobile-Agent-v2 ablation includes a reflection agent. M3A already has summary-based reflection, so
  the marginal gain on A may be small.
- **W:** after each step, one call with the before/after SoM screenshots, the intended action and the Thought. It
  returns `Outcome: A|B|C; Fix: …`, which is inserted into the next observation as `Reflection:`.
- **A:** change M3A's summary prompt to output an explicit A/B/C outcome and fix (a prompt edit, +0 calls).
- **G:** the `visual` history already describes the change. Add a verdict ("as intended?") to `DIFF_PROMPT`.
- **On-demand variant:** fire only when F31/F32 trigger or the model's `Confidence:` field is low. vLLM can return
  logprobs, so the MobileUse gating can be replicated.

**F35 Backtrack.**
- **Evidence:** weak (WILBUR and Agent-E have no isolated numbers).
- **W:** keep a stack of URLs where the reflector said A. On outcome B, or a loop, the executor issues `GoBack`, or
  navigates to the last good URL itself (a scripted step) and tells the model.
- **A:** on B, issue `navigate_back`. If the package differs from the task app, issue `open_app <app>`.
- **G:** on B in a menu, press `b` up to 3 times until `text_boxes` shows no menu.

**F36 Specific execution errors.**
- **Evidence:** weak.
- **W:** in `cusi/envs/webvoyager` step error handling, map exceptions to messages:
  - `IndexError` → "Element [n] does not exist; valid labels are 0..K";
  - `ElementNotInteractableException` → "Element [n] can't be clicked; try a nearby label or scroll";
  - otherwise the original message.
- **A:** write execution failures into the M3A history (the browse notes M3A drops them, `MSG_EXEC_FAILED`).

**F37 Self-consistency vote.**
- **Evidence:** medium. Koh et al. aggregate 20 samples by count; PokeLLMon uses consistent action generation; the
  self-consistency paper. There is no isolated GUI number. TTI says best-of-n gains are small per compute.
- **All envs:** in `decide`, sample k=3–5 with temperature 1.0 (allowed, since the feature is sampling). Parse each
  sample and take the modal action: W key + index, A JSON with the same type and index, G the button. Ties go to the
  first. Keep the Thought of a sample that voted for the winner.

**F38 Pre-execution critic.**
- **Evidence:** weak to medium. WebDreamer and Koh value functions help, but self-correction without new evidence can
  hurt (Huang 2024). Trained verifiers (V-Droid) don't count.
- **W/A:** after `decide`, one call with the screenshot, task, plan, proposed action and the target element's text:
  "Will this action make progress? yes/no + reason". On no, re-run `decide` once with "A reviewer objected: …".
- **G:** the same with the button and frame. Cheap prompt (≤100 output tokens).

**F39 Outcome-simulation lookahead.**
- **Evidence:** medium. WebDreamer: +6 points on VisualWebArena, +11 on Online-Mind2Web, +4.8 on Mind2Web-Live
  (GPT-4o), horizon 1.
- **W/A:**
  1. Sample k=3 distinct actions (F37 sampling).
  2. For each, one call: "Describe the page after <action>".
  3. One call scores all k descriptions for progress 0–1.
  4. Execute the argmax.
- **G:** the same with button presses. A predictable screen makes it cheap, but evidence on games is absent.

**F40 Best-of-k with a comparative judge.**
- **Evidence:** medium. GTA1 +2.1 / +1.8 points (OSWorld, o3 as planner and judge). The BJudge MCQ comparison works
  better than iterative pairwise comparison (trajectory level).
- **All envs:**
  1. Sample k=4 replies.
  2. Deduplicate the parsed actions.
  3. If more than one remains, one judge call shows the screenshot plus the candidates as A/B/C/D with their Thoughts:
     "Which best advances <task>? Answer with a letter".

**F41 Completion self-check.**
- **Evidence:** medium to strong in direction.
  - TTI: check-again +4–5 points (Gemma 3 12B, WebArena).
  - MobileUse global reflector: +2.5 (AndroidWorld).
  - Voyager: no self-verification costs −73% items.
  - Caveat: re-checks can revise correct answers (TTI Map domain).
- **W:** when `decide` returns ANSWER, don't finish yet. Send "You are about to answer '<ans>'. Check the current
  screenshot: does it show this, and does it satisfy every constraint in the task? Reply KEEP or REVISE + next action."
  At most 1–2 checks per task.
- **A:** the same on `status: complete`. Also ask whether `answer` was given if the goal asks a question.
- **G:** done-check variants:
  - require a quoted `text_boxes` line as evidence;
  - 2-of-3 vote on "Complete";
  - run it every k=3 steps (an existing constant) to save calls.

**F42 Trajectory-level best-of-N (W).**
- **Evidence:** medium for BJudge (AndroidWorld +3.5 points, OSWorld +7.3; GPT-5).
- **W:**
  1. Run N=3 episodes per task (fresh browser each).
  2. Build narratives from the per-step reflector or M3A-style summaries.
  3. One comparative judge call picks the episode.
  4. That episode's ANSWER and screenshots go to `auto_eval`.
- Cost ×N. The judge is the same model, so check for judge-gaming by hand.

**F43 Tree search.**
- n/a; see the table.

**F44 Merged done-check + action call (G).**
- **Evidence:** weak (GameBoyRL `FreeExecutor`, no ablation).
- **G:** add `Complete: yes/no` before `Action:` in the action response. If yes, end with `agent_done` before executing,
  and skip the separate `done_check` call. This saves about one third of calls; check for an EARLY regression.

**F45 Event-triggered supervisor.**
- **Evidence:** medium. MobileUse's trajectory reflector is repeat-triggered; ours shows every-leg supervision hurts W
  (−3 to −12 points) and A (−8 to −25).
- **All envs:** the supervisor stays off until F31/F32 fire, or there are m steps with no new state (F50). It then
  receives the narrative (F46) and returns one hint, injected via the existing hint block, without ending the leg.
- Implement it as a supervisor subclass in `cusi/agents/supervisors/` that runs the executor step-wise (or with a
  callback in `run()`).

**F46 Supervisor input as a narrative.**
- **Evidence:** medium (BJudge narratives 60.2 vs screenshots 56.0 for selection).
- **All envs:** in `cusi/agents/supervisors/_format.py`, render the attempt as per-step lines `action → observed
  change` (from M3A summaries, GameBoy `visual` descriptions or F34 reflections) plus the first and last frames,
  instead of all frames.

**F47 Soft plan.**
- **Evidence:** medium for proactive planning (Agent S2 +4.6/+6.2; Mobile-Agent-v2). Ours shows hard plan legs hurt
  on W and A, so the soft form is the variant to test.
- **All envs:**
  1. The supervisor makes one planning call at the start (the existing `PLAN_PROMPT`).
  2. The plan is shown inside the executor prompt as a checklist (as F21).
  3. The executor marks items done in its reply.
  4. The supervisor revises the remainder only when an item is marked done or F45 fires.
  5. There are no per-leg judges or retries.

**F48 Termination veto.**
- **Evidence:** weak to medium (MobileUse's global reflector, as F41).
- **All envs:** on the first `finish`, the supervisor gets the narrative plus the final screen and decides accept or
  continue with a reason. If it continues, the executor resumes with the remaining steps and the reason as a hint. At
  most one veto.

**F49 Answer rewrite (W).**
- **Evidence:** none found (WebEvolver ships it, unablated).
- **W:** after ANSWER, one call with the task, the notes (F27), the last screenshot and the raw answer: "Rewrite the
  answer to directly and completely answer the task; keep only facts visible or noted". The rewritten text is sent to
  the judge.

**F50 Novelty archive signal.**
- **Evidence:** none external (`ours`).
- **All envs:** instantiate the `cusi/state` archive per episode; a near-zero novelty step counts as a no-op and a
  revisit counts as a loop. This feeds F31, F32 and F45. Finalisation stage only (§7).

**F51–F53:** n/a for screening; see the table.

**F54 GameBoy mode detection.**
- **Evidence:** weak (the GameBoyRL strategist splits executors by mode, using RAM).
- **G:** a rule-based classifier over `text_boxes` region kinds (dialogue box, menu, battle HP/FIGHT text), else
  overworld. Add a mode line plus 2–3 mode-specific tips to `build_step_prompt`, for example:
  - dialogue: "press a to advance";
  - battle: "FIGHT is top-left; use arrows then a";
  - overworld: "face the target before pressing a".
- No RAM reads.

---

## 4. Shortlist: the most promising 10 per env (fixed model, by evidence ÷ cost)

The order is a starting point for screening. The §2c failure masses will re-rank it.

**Web**
1. F41 answer self-check: TTI, Gemma family.
2. F11 forced answer at the budget end, plus F10 budget awareness: 0 calls. About 37% of baseline episodes end
   truncated.
3. F27 notes buffer.
4. F14 action-space pruning: AgentOccam +11.6.
5. F03 pruned a11y text: AgentOccam +4.7.
6. F31/F32 no-op and loop detection: navigation stuck is 44% of WebVoyager failures.
7. F09 change report.
8. F21 plan/todo field.
9. F13 extract action.
10. F39 WebDreamer lookahead (if the budget allows), or F40 best-of-k.

**Android**
1. F41 completion check: MobileUse +2.5, TTI.
2. F34 outcome-classifying reflector as an M3A summary-prompt edit, plus on-demand gating: MobileUse +5.7 / +3.0.
3. F32 loop detection.
4. F21 plan field (or F47 soft plan): MA-v2, Agent S2.
5. F27 notes, for information-retrieval tasks.
6. F02 text-only: the AndroidWorld result is model-dependent and the variant is cheaper.
7. F25 self-summary replacing the summary call: −1 call per step if neutral.
8. F29 parametric app tips: the exploration variant is deferred to §7.
9. F08 previous screenshot.
10. F40 best-of-k or F37 vote.

**GameBoy**
1. F04 `text_boxes` in the prompt: allowed reader; lmgame, Gemini, BALROG.
2. F54 mode detection.
3. F17 macros (walk, advance dialogue): 30 of 49 baseline episodes truncated.
4. F05 grid overlay, plus F07 upscale.
5. F44 merged done-check: −1/3 of calls.
6. F41 done-check with evidence or vote.
7. F28 objectives log.
8. F31/F32 hard no-op ban and loop detection.
9. F37 self-consistency vote.
10. F06 + F18 tile map and navigate tool: highest build cost.

---

## 5. Notes and surprises

- **Best like-for-like evidence.** The best evidence for our benchmarks comes from two papers:
  - paper:MobileUse 2025: a single open model on AndroidWorld with a cumulative reflection ablation.
  - paper:TTI 2025: Gemma 3 12B, prompting only, check-again +5 points.
  Most famous Web and Android numbers (Agent-E 73%, browser-use 89%, Agent S2/S3, GUI-Owl) have no ablation for the
  features we would copy, or depend on separate models.
- **Our evidence disagrees with the hierarchy papers.** Ours shows every-leg plan supervision hurts Web and Android
  badly. The literature's planning gains come from **proactive, soft** planning that the executor tracks (Agent S2 PHP,
  MA-v2 planning agent). That is the opposite of our hard leg boundaries with per-leg judging. Hence F47 and F45 are
  framed as soft or event-triggered.
- **Vision versus text is model-dependent and unsettled.**
  - WebVoyager: text-only −19 points (GPT-4V).
  - AndroidWorld: text-only better for GPT-4T, worse for Gemini.
  - BALROG and the Gemini Pokémon appendix: vision often adds little in games.
  F02 (Android) and F04 (GameBoy) are therefore worth early, cheap tests.
- **Self-correction can hurt.** It needs new evidence (the after-screen, `text_boxes`) to help (Huang 2024; TTI's Map
  domain). Every check feature (F38, F41, F48) is phrased around the current observation, not the model's own
  reasoning.
- **Truncation dominates episode endings.** Web 159/429 and GameBoy 30/49 baseline episodes end truncated (to confirm
  this is the step budget). Cheap budget-awareness and finishing features (F10, F11, F16, F17) may matter more than any
  paper suggests, because papers mostly use larger budgets.
- **Claude and Gemini Pokémon relied on RAM.** Their grid, minimap and navigator features relied on RAM-derived map
  data. Our versions (F05, F06, F18) must build walkability from the screen, as GameBoyRL's strategist tile classifier
  does. That classifier exists and uses the same VLM, so it is a short port.
- **Some features would change the harness.** F33 (free re-asks), F43 (save-state search) and F53 (cross-task memory)
  would change step accounting, use privileged emulator state, or break task independence. They are flagged for a user
  ruling rather than screened.
