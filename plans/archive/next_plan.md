# Next plan

**§0 (unified text representation) comes first.** §A–D (VLM features, decision scores, sliced
descriptions, viewer) and §E (curiosity policy) follow it. §E depends on §0's element tokens.

## How this plan is executed (decided 2026-10-11)

- **No commits and no pushes** during this work, and no check-ins: implement straight through,
  in the order below, testing as I go (Slurm jobs included).
- **Order:**
  1. §0 text boxes: Android and Web, then the GameBoy tile reader for all 22 variants with ROMs.
  2. §0.1–§0.2 novelty: trigram box channel, masking, unseen-text scorer, archive format 2.
     **Then delete the old, incompatible data:**
     - everything under `storage/explore/<env>/`: the `dev_*` runs, `debug`, `decoder`,
       `embedder` and the `elements_*.json` files, all derived from old replays;
     - the `storage/search/*/search_test_*` runs.

     Eval runs are untouched.
  3. §C sliced describe with `discovered`.
  4. §A node features.
  5. §B decision scores.
  6. Real-VLM GameBoy search run, with calibration of `node_threshold` and `tau`.
  7. §E curiosity: GameBoy first, then Web and Android at their lower budgets.
- **VLM for every real-model run:** `google/gemma-4-31b-it`, served by vLLM on 2× RTX A6000, as
  in the ablation evals. It is passed explicitly as `--model`; it is never a default in code.
- **The 4 game variants without ROMs** (Ham Hams Unite, Starbeasts Comet, Runes of Virtue 1–2) are
  not pursued.
- **GameBoy glyph tables:** text moments come from GameBoyWorlds' saved states plus random play,
  and I label the glyph sheets by viewing them.
- **Accepted defaults:**
  - §B: discovered set per scene; re-score the top 8 candidates each decision; score = fraction
    of undiscovered features.
  - §E: MobileNetV3-Small (vision) and MiniLM-L6 (text).
- **Pretrained weights** (MobileNetV3-Small via torchvision, MiniLM-L6 via HF) are downloaded under
  `storage_dir` / `HF_HOME`, never into `~`.

## Implementation status (2026-10-11)

Done, uncommitted. Tests: `tests/cusi_state_unit_test.py`, `search_core_test.py`, `search_store_test.py`,
`search_vlm_test.py`, `curiosity_expander_test.py` all pass; live checks `tests/text_boxes_live.py`,
`tests/curiosity_candidates_live.py` (Slurm: `slurm/env_run.sh`).

**§0 text boxes**
- `cusi/envs/text_boxes.py` (`TextBox`, reading order, rendering, overlays), `android_boxes.py`, `web_boxes.py`,
  `gameboy_text/` (reader, harvest tool, one table per variant).
- **Deviation:** `obs["texts"]` is unchanged (the labelled lists the executors use, so eval prompts stay identical);
  the new text is `obs["text_boxes"]`, which the search, novelty and VLM prompts read. `TextBox` also carries `ref`
  (the env's element index for actions) and `scrollable`, both needed by §E.
- Android: clock and status bar gone (live check: 0 clock-like boxes). Web: page text present (arXiv: 111 boxes, 96
  interactive), boxes overlay correctly.
- **GameBoy:** the glyph key is the tile's **ink mask** (pixels differing from the tile's majority colour index),
  not its raw bytes, so it is independent of palette and colour index. Every matched cell is checked against the
  screen pixels (no text from covered layers or sprites). Roles come from GameBoyWorlds' OCR kinds that are active at
  that step (`can_read_kind`), within the kind's region.
  - Pokémon (6 variants): seeded from the gen1 / gen2 charmaps (tile ids on screen at text moments, plus the font in
    VRAM when the screen-seeded glyphs are where the charmap says), plus a few hand labels.
  - The rest were labelled by eye from glyph sheets (with label proposals from the other tables).
  - Fallbacks implemented: **pixel segmentation** for variable-width fonts (`segments.py`; Harry Potter CoS) and
    **8×16 pairs** (Oracle of Seasons).
  - Results (check runs on saved states plus random play):

    | Variant | Glyphs | Notes |
    |---|---|---|
    | pokemon red / brown / crystal / fools_gold / prism / starbeasts | 97 / 150 / 92 / 91 / 90 / 96 | dialogue and menus read |
    | deja_vu 1 / 2 | 39 / 39 | dialogue read in full |
    | legend_of_zelda links_awakening | 73 | dialogue read |
    | legend_of_zelda oracle_of_seasons | 74 (incl. 8×16 pairs) | dialogue read |
    | survival_kids_1 | 82 | dialogue and menus |
    | sword_of_hope 1 / 2 | 43 / 45 | HUD, menus, messages |
    | harvest_moon 1 / 2 / 3 | 74 / 76 / 47 | text read; roles mostly `text` |
    | bomberman max / quest | 66 / 41 | 8×16-free dialogue font; max's menus use a variable-width font (not read) |
    | harry_potter chamber_of_secrets | 51 segments | partial: some glyphs (e.g. `t`) still unlabelled |
    | bomberman_pocket | 0 | **unreadable**: only text is 16-px multi-colour outlined title cards |

  - **Not portable here, besides the 4 known missing ROMs:** `survival_kids_2` (ROM file missing) and
    `harry_potter_philosophers_stone` (ROM fails PyBoy's header checksum). `pokemon_brown`, `pokemon_starbeasts` and
    `harry_potter_chamber_of_secrets` only load in GameBoyWorlds' `debug_mode` (missing region captures).
- Reading cost: ~3 ms per GameBoy step.

**§0.1/§0.2 novelty**
- As planned: trigram keys (D = 2^20, crc32), chamfer with capped area weights, masking on by default, archive
  `format: 2` with interned keys and a padded gather (checked against brute force), `UnseenTextBuffer` for every env.
  `overlap` / `tfidf` / `dense` removed.
- **Additions:** a screen keeps at most its 64 largest boxes (bounds the gather on dense web pages).
- **Streaming rule, as implemented:** a box seen by the threshold or as a prefix of a seen key scores 0; a box that
  extends the previous screen's box (still typing) is credited only for the share of its characters that are new. A
  line typed over several frames therefore scores a little more than once (1 + Σ new/len, ≈1.4 for two partials).
- Old data deleted (`storage/explore/*/*`, `storage/search/*/search_test_*`).

**§C, §A, §B** as planned (`describe.py` sliced, `features.py`, `scoring.py`), with `--features`, `--vlm_score`,
`--score_top_m`, `--default_score`, `--max_features`. `prior.py` removed.

**Step 6, real-VLM GameBoy run and calibration** (`gemma_vlm_v1`, Gemma 4 31B, 40 expansions):
- With raw P, every decision went to a freshly created node (the last step of each expansion is a node, and an
  unexpanded node gets the full `c·P·√N_total` bonus): ~1.3 effective candidates, yields ~0.
- `cusi/search/calibrate.py` recomputes every saved decision under other settings. **New defaults:** P normalised over
  the candidates (a PUCT prior, `--normalise_p`), `c = 2`, `tau = 0.05` (≈7.5 effective candidates, ~77% of the mass
  on unexpanded nodes). `node_threshold` stays 0.1 (≈ the 95th percentile of step novelty in random runs).
- Separately, Gemma explores GameBoy poorly (it walks into walls; the explorer's yield is ~0 after a few
  expansions). That is the model, not the selection. A rerun with the new defaults is `gemma_vlm_v2`.

**§E curiosity** (`cusi/search/curiosity/`): candidates, the fused policy (909k trained parameters), PPO session,
overlay archive (B0 ∪ episode), expander, saved policies and returns, resume cleanup.
- **Deviation:** the best episode is materialised by **replay on every env** (GameBoy exact, else checked by
  embedding similarity and cut at the first diverging step with `materialise_diverged`); Web does not save a state per
  step.
- The encoder models are required flags (`--policy_vision_encoder mobilenet_v3_small`, `--policy_text_encoder
  <MiniLM id>`), per the no-default-model rule; MobileNet weights live under `$HF_HOME/torch_hub`.
- Speed: ~40 steps/s on CPU with 4 GameBoy copies (100k steps ≈ 40 min).
- **Runs:**
  - GameBoy parity with GameBoyRL's `ppo_curiosity` (mean return 1.35→3.31 against 1.55→3.08).
  - Go-Explore runs on GameBoy, Web and Android: one-hour sessions, no crashes after the Android fixes. The a11y
    tree is refreshed after snapshot loads, and a session whose node cannot be restored stops early or is skipped.
- **Coverage (`--rl_commit_episodes N`):** each session commits its top N distinct episodes, each as its own
  expansion of the node.
  - Shorter sessions with 10 episodes each (`gameboy/curiosity_topk_v1`) reached 3,498 states and 427 texts in 2 h.
    The best-episode-only setup reached 290 and 238 in 7.2 h.
  - Random (5,715 and 947 in 1.2 h) is still ahead per hour.

**Step 6 on Web and Android** (`gemma_vlm_v1`, Gemma 4 31B, 40 expansions, calibrated selection):
- **Web:** 150 nodes, 144 archive states, 1,055 texts, 188 discovered features, mean yield 0.130 (GameBoy 0.020).
- **Android:** 105 nodes, 204 archive states, 286 texts, 178 discovered features, mean yield 0.083; 7 nodes flagged as replay-diverged.
- **Fixes the runs needed:**
  - The decision scorer no longer reads `env_description` through `SceneEnvs`, which rebuilt the other Android scene
    on every decision.
  - Android scene builds are retried, restarting the emulator on the last try.
  - WebDriver calls have a 120 s client timeout; a frozen page restarts the browser instead of hanging the search.

**§D viewer** (`cusi/search/viewer.py` + `viewer.html`, done in this pass): `<run dir>/viewer/index.html`, rebuilt
at the end of every run, or by `python -m cusi.search.viewer --env <env> --run <run>`.
- The tree with thumbnails.
- Node details: features with verdicts, decision score, Q / N / P, flags.
- An expansion player: steps, actions, values, texts, description, windows.
- Timeline charts and an expansions table.
- Static, with no network dependencies; light and dark themes.

---

## 0. Unified text representation: frame + text boxes, for all three envs

### Why

Each env feeds text in its own way today, and each has problems:

| Env | `obs["texts"]` now | Problems |
|---|---|---|
| GameBoy | empty. GameBoyWorlds' "OCR" only captures crops of named regions (dialogue, menus) as images, in `info["text_regions"]`. | No text at all. Region novelty is pixel strips of the crops (`OCRRegionBuffer`). |
| Android | M3A's dump of the accessibility tree: one line per element, text fields mixed with ~10 boolean flags. | Status-bar clock and battery are included: the clock changes every minute (seen: 15:34 and 15:35 in one test run), which gives fake text novelty. The flags dominate `overlap` and `tfidf`. Some elements have flags only. |
| Web | WebVoyager's set-of-mark list: interactive elements in the viewport only (`[i]: <tag> "text"`). | No page content (headings, paragraphs), so two pages of one site look almost the same in text. |

### The representation

```python
@dataclass
class TextBox:
    text: str                      # cleaned, whitespace-normalised
    box: tuple                     # (x0, y0, x1, y1) in pixels of the unlabelled frame (info["raw_frame"])
    role: str                      # heading | body | label | button | link | input | menu | dialogue | text
    interactive: bool
    source: str                    # a11y | dom | tiles
```

- **Observation:** `obs["frame"]` (the labelled frame, unchanged) plus `obs["text_boxes"]`, a list of
  `TextBox` in reading order (top to bottom, then left to right).
- **`raw_frame`** stays the unlabelled frame the boxes refer to.
- **Labels stay separate:** set-of-mark labels and element indices remain in the action layer
  (`obs["actions"]`, `texts` for the action prompt). The observation's text is labels-free.
- **Compatibility while consumers move over:** `obs["texts"]` is rendered from the boxes as
  `{"text_boxes": "<role>: <text>" lines}`. The labelled lists that Web's and Android's action
  prompts need are kept where the executors use them.
- **Helpers:** `TextBox` and its helpers (reading order, merging, rendering) live in
  `cusi/envs/text_boxes.py`.

### Android: filter the accessibility tree (cheap)

Every node has `bbox_pixels`, `text`, `content_description`, `hint_text`, `tooltip`,
`class_name`, `resource_name`, `package_name` and the flags (`android_world/env/representation_utils.py`).
- **Text:** the first non-empty of `text`, then `content_description`, then `hint_text`
  (`tooltip` is appended if it differs). Nodes with no text are dropped.
- **Drop:**
  - nodes that are invisible, zero-area or off-screen;
  - `package_name == "com.android.systemui"` (status bar, navigation bar, notification shade),
    which fixes the clock bug;
  - the IME keyboard (its own package).
- **Merge:** a container whose text equals its only text-bearing descendant's keeps one box: the
  descendant's box, and the container's `interactive`.
- **Role:** from `class_name` and the flags.
  - `EditText` (or `is_editable`) → input.
  - `Button` / `ImageButton` (or clickable with short text) → button.
  - `TextView` → label or body by length.
- **Gap:** text drawn inside images, canvases or WebViews without an accessibility bridge isn't
  in the tree (see the open items).

### Web: text runs from the laid-out DOM (cheap to medium)

- **One `execute_script`** per observation (the bot-check detector's DOM pass can merge into it):
  walk the visible text nodes and get their boxes with `Range.getClientRects()`. Also take
  `input` / `textarea` values and placeholders, `img[alt]`, and `aria-label` of icon-only buttons
  and links.
- **Visibility:** inside the viewport, not `display:none`, `visibility:hidden` or `opacity:0`,
  and not covered by another element (`elementFromPoint` at the box centre, as WebVoyager's
  script already does).
- **Scale:** CSS pixels × `devicePixelRatio` = screenshot pixels.
- **Group** runs into one box per line within each block-level ancestor, and cap the count per
  page (e.g. 300 boxes, in reading order).
- **Role** from the tag and ARIA attributes: `h1`–`h6` / `role=heading` → heading;
  `a` → link; `button` / `role=button` → button; form fields → input; `li` inside `nav` → menu;
  otherwise body. `interactive` uses WebVoyager's own clickable rule (tag, `onclick`, pointer
  cursor).
- **Fallback:** if Chrome's `DOMSnapshot.captureSnapshot` (laid-out text boxes in one call)
  proves faster or more robust on heavy pages, use it instead. Measure both on the task sites.

### GameBoy: read the text from the tiles (all GameBoyWorlds games)

No OCR model and no VLM on crops. Game Boy text is drawn as 8×8 tiles on the background and
window layers (PyBoy: `tilemap_background`, `tilemap_window`, tile data per cell; scroll
`SCX`/`SCY`, window position `WX`/`WY`). Two ways to turn tiles into characters, decided per game:

- **(a) Tile id → character table.** For games whose font sits at fixed tile ids (Pokémon Red/Blue
  and its hacks), one table per game maps tile ids to characters. Exact, and close to free.
- **(c) Glyph table (the general case).** Key the table by the tile's pixel content: a hash of
  its 16 bytes of 2-bit-per-pixel tile data, independent of palette. This still works when a game
  streams font glyphs into whatever VRAM slots are free (common for dialogue engines, e.g.
  Link's Awakening), and on GBC VRAM bank 1.
  - **Building a table:** harvest every distinct tile seen in text moments, which are frames where
    GameBoyWorlds' `can_read_kind` reports dialogue or menu text, plus random-play frames. Keep
    the tiles that look like glyphs (ink inside the cell, not blank or solid). Render them as a
    labelled glyph sheet and label each glyph once. I can do the labelling by viewing the sheets;
    each game's font is usually under 150 glyphs.
  - **Bootstrapping:** where a game's font is stored uncompressed in the ROM in character order
    (e.g. Pokémon), it can seed the table automatically.

**Boxes:** visible tile cells → screen pixels (cell × 8 − scroll, or + window position).
Adjacent text tiles in a row form a run; runs on neighbouring rows inside one GameBoyWorlds
named region (`dialogue_box_full`, menu boxes) form a block. **Role:** `dialogue` or `menu` from
the region the box falls in (its `kind`), else `text` (HUD, signs, titles).

**Per-game risks, to be checked game by game:**
- **Variable-width fonts:** glyphs are drawn as pixels across tile boundaries, so one tile is not
  one character. The fallback for such a game is glyph segmentation at pixel level inside the text
  region (split columns on blank vertical runs, then match glyph bitmaps against the labelled set).
  This is still template matching, not OCR.
- **Text drawn with sprites** (OAM) rather than background tiles: sprite tiles are matched against
  the same glyph table, at sprite positions.
- **Japanese or graphical fonts and stylised title screens:** recorded as unreadable, not guessed.

**Coverage:** 26 game variants in GameBoyWorlds (`storage/rom_data/`):

| Game | Variants | ROM present |
|---|---|---|
| Pokémon | red, brown, crystal, fools_gold, prism, starbeasts, starbeasts_comet | all but starbeasts_comet |
| Bomberman | max, pocket, quest | all |
| Deja Vu | 1, 2 | all |
| Harry Potter | chamber_of_secrets, philosophers_stone | all |
| Harvest Moon | 1, 2, 3 | all |
| Legend of Zelda | links_awakening, oracle_of_seasons | all |
| Survival Kids | 1, 2 | all |
| Sword of Hope | 1, 2 | all |
| Ham Hams Unite | 1 | **missing** |
| Runes of Virtue | 1, 2 | **missing** |

That is 22 portable. The 4 without ROMs can't be ported or checked here; add them if their ROMs
appear. A game is only given up if neither (a) nor (c) nor pixel-level segmentation reads its text,
and the reason is recorded.

**Per-game output:** `cusi/envs/gameboy_text/<variant>.json`, holding the method (`ids` or
`glyphs`), the table, and notes. Code in the project root, no ROM data in it.

**Per-game check:**
1. On saved text moments (GameBoyWorlds' region reports), the decoded dialogue and menu boxes are
   non-empty.
2. A rendered overlay of boxes and strings on sample frames, which I inspect, shows correct text in
   the right places.
3. Unknown glyphs are reported, not dropped silently.

### Downstream changes (with §0.1 below)

- Prompts: the explorer, the describer and the feature and score calls get `text_boxes`.
- Practice and eval executors keep their current labelled prompts unless switched explicitly.

### Order of work (§0)

1. `TextBox` plus helpers; Android filtering; Web DOM extraction. Live checks on the env test pages:
   the clock is gone, Web page content is present, and boxes overlay correctly (rendered images
   inspected).
2. GameBoy tile reader with method (a) on Pokémon Red. Then (c) glyph tables for the remaining
   variants, one game family at a time, with the per-game check. Variable-width or sprite-text
   fallbacks only where a game needs them.
3. The state-embedding and novelty changes (§0.1).
4. Rerun the search env tests and `curiosity` smoke tests.

### Required by §E (curiosity policy)

The policy points at elements, so the representation must also carry **interactive elements
without visible text**: icon-only buttons, image links, unlabelled inputs. They become boxes with
role `button`, `link` or `input`, `interactive=True`, and the best available text: Android
`content_description`, Web `aria-label` / `title` / `alt`, otherwise empty. Novelty (§0.1/§0.2)
ignores boxes with empty text; the policy uses all of them.

### Open items

- Android text that is drawn rather than in the tree: accept the gap for now.
- Whether Web needs both the DOM text and the set-of-mark list in prompts, or the DOM text alone.

## 0.1 State embedding and novelty on frame + text boxes

### Now (`cusi/state`)

- **`StateEncoder`:** an image embedder (`random_patch` / `cnn` / `siglip`) on the unlabelled frame,
  plus a text embedder (`none` / `overlap` / `tfidf` / `dense`) on the screen's text lines joined
  as one bag. It gives `StateEmbedding(image, text)`.
- **State similarity** = `w_image · image_sim + (1 − w_image) · text_sim` (`w_image` 0.5; 1.0 when
  text is `none`, which is always the case on GameBoy).
- **`NoveltyArchive`:** every state seen (deduplicated, compacted by KMeans above 10k).
  `embedding` novelty = 1 − max similarity to the archive.
- **Region novelty** is a different mechanism per env:
  - GameBoy: `OCRRegionBuffer`, pixel strips of the dialogue and menu crops, scored as the share
    of strips never seen.
  - Android and Web: `TextLineBuffer`, the share of the screen's raw lines never seen.
- **`combination`** = `α · region + (1 − α) · frame`.
- **Problems:**
  - Text is position-free and polluted (clock, flags) on Android, thin on Web, and absent on
    GameBoy.
  - "Text novelty" means three different things across the three envs.
  - Any text change is counted twice, once by the image (its pixels changed) and once by the text
    channel.

### New

1. **One text channel for all envs, built from `text_boxes`.** A screen's text is a set of
   `(role, text)` items instead of one bag.
   - **Text similarity between two screens:** symmetric mean-max (chamfer) over their boxes.
     Each box is matched to its most similar box on the other screen, with only same-role
     matches counting. The two directions are averaged, and boxes are weighted by box area, so a
     heading outweighs a footnote.
   - **Box-to-box similarity:** character-trigram cosine (decided; mechanics in §0.2).
   - **Archive scaling:** a set-vs-set comparison per stored state is heavier than one vector
     dot; §0.2 does it as one sparse product plus a vectorised gather.
2. **Image embedding on the frame with text masked out** (`--mask_text`). The text boxes are
   filled with each box's background colour before embedding. The image channel then measures
   layout and visuals (screens, sprites, maps), the text channel measures content, and a text
   change no longer moves both. A side effect: changing digits (clocks, counters) stop moving
   the image channel too. `--no_mask_text` keeps today's behaviour, for comparison.
3. **One region scorer for all envs: unseen text.** The share of the screen's text boxes (area
   weighted) whose `(role, normalised text)` was never seen in the archive: no archived key with
   the same role has trigram cosine ≥ 0.9 (§0.2). This replaces both `OCRRegionBuffer` and
   `TextLineBuffer`. On GameBoy it now rewards new dialogue or menu *content*, not new pixel
   strips.
   - **Streaming dialogue** (text typed one character at a time): a box that is a prefix of a seen
     string, or of the same box on the next frame, doesn't count as new. In practice a dialogue
     box scores when its text stops changing or when the line completes. Without this, every
     typed character would look like a new string.
4. **`combination`** keeps its form: `α · text novelty + (1 − α) · image novelty`, now with
   both channels computed the same way on every env.
5. **`embedding` novelty** (one similarity over both channels) uses the new text similarity in
   place of the bag. `w_image` keeps its meaning, but no longer has to fall back to 1.0 on
   GameBoy.
6. **Unchanged:** the archive's deduplication, compaction and save format (plus each state's
   box set); the `world_model` scorer (image only; it can be trained on masked frames for
   consistency); the scorer interface used by both the tree search and curiosity PPO.

### Contrast

| | Now | New |
|---|---|---|
| Text input | raw lines: tree dump / set-of-mark list / nothing on GameBoy | `text_boxes` (role, text, box) on all three |
| Text similarity | one bag vector per screen | set matching over boxes (same role, area weighted) |
| Image input | whole unlabelled frame | frame with text boxes masked (optional) |
| Region novelty | GameBoy pixel strips of crops; Android/Web unseen raw lines | unseen `(role, text)` boxes on all envs, streaming-aware |
| Double counting | text changes move both channels | separated |
| Known spurious novelty | Android clock; black frames (fixed) | clock removed at the source |

**Decided (2026-10-10):**
- Text masking in the image channel, on by default.
- Character trigrams for text (§0.2), with:
  - digit masking in the key;
  - a "seen" threshold of 0.9;
  - area weights capped at 25% per box;
  - `overlap`, `tfidf` and `dense` removed once nothing uses them.
- Hashed trigram vectors at D = 2^20 with a stable hash (no vocabulary).

### Compatibility

- Archives and replays scored the old way aren't comparable with the new ones. The archive gets
  a format version, and old runs (the `dev_*` explore runs, the `search_test_*` runs) are
  rescored or regenerated.
- `trajectories.rescore` (curiosity `tasks`) uses the new scorer.

### Tests

- Unit:
  - chamfer similarity on hand-built box sets (identical = 1, disjoint = 0, role mismatch
    ignored);
  - masking leaves pixels outside the boxes unchanged;
  - a streaming dialogue sequence scores once.
- Live (per env, with `debug`-style random runs): reward traces compared with the current scorer.
  - Android: no novelty from the clock.
  - Web: page changes register through text.
  - GameBoy: new dialogue scores and repeats score 0.

## 0.2 Text boxes with character trigrams: the mechanics

### One box → one vector

1. **Key** = `(role, normalised text)`. Normalising means lowercase (with `casefold`), collapse
   whitespace, strip, cap at 500 characters, and replace every digit with `#` (the novelty key
   only; the box keeps its real text for prompts). Without digit masking, counters look half new:
   `HP 23/45` vs `HP 22/45` is only 0.67. With it they're identical, and `Level 5` vs
   `Level 50` (`level #` vs `level ##`) is still close.
2. **Trigrams** of the key text padded as `"  " + text + " "`, so short strings like `OK` still
   have three trigrams.
3. **Vector:** trigram counts hashed into D = 2^20 dimensions with a stable hash (blake2b or
   crc32, never Python's process-salted `hash()`), L2-normalised.
   - **Sparse:** about one non-zero per character, so D costs nothing in memory or compute.
   - **No vocabulary and no IDF:** nothing is fitted or saved, the same string always gives the
     same vector, and the representation doesn't depend on what else has been seen.
   - **Collisions:** with ~30k distinct trigrams in use, about 3% of trigrams share a bucket at
     2^20 (about 11% at 2^18). A collision only adds one spurious shared entry when the two
     colliding trigrams fall in the two strings being compared.
   - **Rejected:** an exact vocabulary interned with the archive. No collisions, but more
     bookkeeping and saved state.
4. **Box similarity** = cosine, and 0 if the roles differ.

Measured on example strings (case already folded):

| Pair | Cosine |
|---|---|
| `Wi-Fi` / `WI-FI` | 1.00 |
| `Hello! Welcome to the world of POKéMON!` / its half-typed prefix | 0.91 |
| `Settings` / `Search settings` | 0.79 |
| `Network & internet` / `Internet` | 0.65 |
| `Sign in` / `Sign up` | 0.62 |
| `PIKACHU used THUNDERSHOCK!` / `PIKACHU used TACKLE!` | 0.59 |
| `Wi-Fi` / `Wi-Fi preferences` | 0.58 |
| `OK` / `On` | 0.33 |
| two different arXiv paper titles | 0.31 |
| `Sign in` / `Log in` | 0.27 |

Same strings match, different labels stay apart even when short, and paraphrases are *not*
matched (`Sign in` / `Log in` is 0.27). That's the known cost of trigrams against a sentence model.

### Two screens → text similarity (chamfer)

For screens A and B with boxes `a_i` (area weights `w_i`, summing to 1 per screen):

```
sim(A→B) = Σ_i w_i · max_j cos(a_i, b_j)        (same role only)
text_sim(A, B) = ½ (sim(A→B) + sim(B→A))
```

- Every box looks for its best partner on the other screen. Big boxes count more. Extra boxes
  on either side lower the score, through the other direction.
- **Both screens without text:** the text channel is undefined, so state similarity is the image
  similarity alone (its weight renormalised). This is common on GameBoy overworld frames.
- **One screen with text and one without:** text similarity 0.
- **Area weights are capped** (no box above 25%), so one full-screen paragraph can't decide the
  whole screen.

### Against the archive (up to 10k states, ~30 boxes each)

- **Interning:** each archived state stores the ids of its distinct box keys. Distinct keys are
  stored once, so nav links repeated across pages cost nothing extra. Keys are expected to number
  in the tens of thousands.
- **Key matrix:** a sparse `K × 2^20` matrix of key vectors, grown as new keys arrive.
- **One query screen with `q` boxes:**
  1. One sparse product `Q · Kᵀ` gives every query box's cosine to every key (`q × K`), with roles
     masked.
  2. `sim(query→state)`: for every state, max over its keys, then the weighted mean. Done
     vectorised with a padded index array of the archive's key ids per state
     (`N × L`, L = most boxes on one screen), so it's one gather plus max: `q × N × L`, about
     9M floats at 30 × 10k × 30.
  3. `sim(state→query)`: the same gather with the other axis maxed.
  4. `text_sim` for all states at once; state similarity = `w_image · image_sim +
     (1 − w_image) · text_sim`, and novelty = 1 − max.
- **Cost:** on the order of milliseconds to tens of milliseconds per query on CPU at full archive
  size. No GPU, and cheap next to an env step.

### Text novelty (the region scorer)

- **Unseen-text score:** for each box, `seen = max cos to any archived key with the same role ≥
  0.9`. Box-level max over keys, not states, so it comes straight out of the `Q · Kᵀ` product.
  The 0.9 threshold matches case and spacing variants (`Wi-Fi` / `WI-FI`) and keeps distinct
  labels apart (`Sign in` / `Sign up`, 0.62).
- **Score** = area-weighted share of unseen boxes. A soft version, `Σ w_i (1 − max cos)`, is the
  fallback if the hard threshold proves too jumpy.
- **Streaming text:** a box whose key is a prefix of an archived key with the same role, or of the
  same box on the next frame, counts as seen. The half-typed Pokémon line above (0.91) is caught by
  the threshold anyway; the prefix rule makes it independent of the threshold.

### Archive bookkeeping

- **Deduplication:** a new state is a duplicate only if its image is near-identical *and* its set
  of box keys is the same. Today's text check is "the text embedder says identical".
- **Compaction:** KMeans on images as now; each kept state keeps its key ids. Keys no longer used
  by any state are dropped.
- **Saving:** `archive.pt` gains the key strings with their roles, each state's key ids, and a
  `format: 2` field. The key vectors are rebuilt from the strings on load.

### Image masking (decided: on)

- Each text box's pixels in the unlabelled frame are filled with the median colour of the 1-pixel
  ring around the box (its background) before the image embedder sees the frame. The canvas is
  resized afterwards, as now.
- This happens in `StateEncoder.encode`, so the tree search, curiosity PPO, the world model and
  the archive all see the same masked frames.
- The frames shown to VLMs and stored in replays stay unmasked.

### Flags and code

- **Flags:**
  - `--text_embedder trigram` (new): the box-set channel above. `none` stays; `overlap`, `tfidf`
    and `dense` are removed once their callers move over, since a set channel replaces a bag
    channel.
  - `--mask_text/--no_mask_text`, default on.
  - `--text_seen_threshold`, default 0.9.
- **Code:**
  - `cusi/state/encoders/text.py`: the trigram key vectoriser.
  - `cusi/state/archive.py`: key interning and the chamfer gather.
  - `cusi/state/regions.py`: one `UnseenTextBuffer` replacing `TextLineBuffer` and
    `OCRRegionBuffer`.
  - `cusi/state/state.py`: masking.

### Tests

- The table above as a unit test.
- Chamfer properties: identical screens = 1, disjoint = 0, role mismatch ignored, empty-vs-empty
  falls back to image.
- The gather against a brute-force loop on a random archive.
- Deduplication and compaction keep the key ids consistent.
- Masking changes only box pixels.
- Streaming dialogue scores once.

---

# VLM features and decision scores, sliced descriptions

Follows `plans/archive/tree_search_plan.md` (tree search built; VLM explorer with tree context and
`--describe` done). Three pieces:

- **A. Node features:** each node gets a list of features it could surface, instead of a prior score.
- **B. Decision scores:** at selection time, the VLM scores nodes against what has already been
  discovered.
- **C. Sliced, composed descriptions:** so `--describe` never exceeds the server's limits.

The viewer is the pass after this one (§D).

---

## A. Node features (replaces the 0–10 prior)

**Now:** `VLMPrior` (`cusi/search/prior.py`) gives each new node a score from 0 to 10 (`Score:`,
`Reason:`), stored as `prior`, `prior_reason` and `prior_raw`, and used as `P(n)` in the energy.

**New:** `VLMFeatures` (replacing `VLMPrior`). Once, when a node is created, the VLM gets the node's
(unlabelled) frame, its texts and the env description. It is asked for the interesting features or
functionalities that exploring from this screen could surface: screens, menus, settings, content,
interactions or game areas reachable from here that someone exploring the environment would want
to find.
- **Reply format:** `Features:` followed by one `- <feature>` per line, each a short noun phrase
  ("Wi-Fi network list", "advanced search filters", "the Pokémon Center interior"). At most
  `--max_features` (default 8); duplicates and empty lines are dropped.
- **Stored on the node:** `features` (the list) and `features_raw` (the reply), in `nodes.jsonl`.
  These replace `prior`, `prior_reason` and `prior_raw`, which are removed. If parsing fails, the
  node gets `features = []` and the flag `features_parse_failed`.
- **Flag:** `--features/--no_features` replaces `--prior/--no_prior`.
- The TODO in `prior.py` is resolved by B, so it goes, along with the file (`features.py`
  replaces it).
- The fake-VLM tests change with it.

## B. VLM decision score (per decision, against what has been discovered)

**Idea:** a node is worth expanding if its features haven't been discovered yet. "Discovered"
comes from the transition descriptions (§C).

### B.1 What has been discovered

The describe call also returns, besides the description, the features the sub-trajectory actually
surfaced: `Discovered:` followed by one `- <feature>` per line, in the same style as A. They are
stored on the expansion as `discovered` (a list), in `expansions.jsonl`. No extra call is needed.

The discovered set for a decision is the union of `discovered` over the scene's expansions, with
exact duplicates (after normalising case and whitespace) removed. It is a list of short phrases,
so it stays small where the full descriptions would not: 1000 expansions × 3 sentences ≈ 60k
tokens.

The full descriptions stay what the explorer's prompt uses ("earlier explorations from this
node"); the scorer sees the phrase lists.

### B.2 The score

`VLMScorer.score(nodes, discovered)`: one text-only prompt per node, all batched in one `infer`
call (vLLM handles them in parallel).
- **Prompt:** the env description, the node's `features`, and the discovered list.
- **Asked:** for each of the node's features, whether exploration has already surfaced it
  (`yes`, `partly` or `no`, plus a short reason).
- **Reply format:** one line per feature, `<k>. <yes|partly|no> — <reason>`.
- **Score:** the fraction of features still undiscovered, with `no` = 1 and `partly` = 0.5,
  divided by the number of features. A node with no features gets `--default_score` (0.5).

This is more grounded than a free 0–10 rating: it is a count over the node's own predictions,
and the text says which features remain.

The score is `P(n)` in the existing energy, `E = Q + c·P·√N_total/(1+N)`. A node whose features
are all discovered loses its exploration bonus, though it can still be chosen through Q.

### B.3 Which nodes get scored each decision

Scoring every candidate at every decision costs one call per candidate per iteration, and the
candidate count grows with the tree. Proposal:
- **Cache:** each node keeps its latest score. Never-scored nodes count as 1.0 (optimistic), so
  they get scored soon.
- **Each decision:** compute energies with the cached scores, take the `--score_top_m`
  candidates with the highest energy (default 8), re-score them in one batched call against the
  current discovered set, recompute their energies, then sample.

Scores are therefore per decision for every plausible pick, and stale scores never keep a node
artificially high (it is re-scored the moment it rises to the top). `--score_top_m 0` scores
every candidate every decision, which is exact but costs grow with the tree.

### B.4 What is saved

Each expansion row's selection record gets a `scores` list, one entry per node scored at that
decision: `{"node_id", "score", "verdicts": [{"feature", "verdict", "reason"}], "raw"}`. It also
gets `n_discovered`, the size of the discovered set the scores were computed against. The
cached score per node is rebuilt on load by replaying the records in order, like `Q` and `N`, so
nothing else is persisted.

On resume the next decision re-scores from scratch, which is equivalent, and the selector
reseeding keeps sampling reproducible given the scores (with temperature 0).

Iterations whose restore fails write no record today. Their scoring calls are lost, and that is
accepted.

### B.5 Flags and costs

- **Flags:** `--features` (A), `--describe` (needed for `discovered`) and `--vlm_score`. Each
  needs `--model_name`. `--vlm_score` requires `--features` and `--describe`.
- **Calls per iteration:** one describe call (sliced, §C), one features call per new node (one
  image), and up to `score_top_m` text-only scoring calls, batched.

## C. Sliced, composed descriptions (GameBoyRL's strategy)

**Why:** one describe call sends `k_steps + 1` frames. That breaks on two server limits from
`configs/project_vars.yaml`:
- `vllm_max_images` (16): any `--k_steps` of 16 or more.
- `vllm_max_model_len` (32,768 tokens): Android frames (1080×2400) are about 3,300 tokens each
  for Qwen-VL (one token per 28×28-pixel patch), so 11 frames ≈ 36k. At the default 10 steps,
  every Android call breaks.

**GameBoyRL** (`execution/supervisors/checker.py`, `window_trajectory`): cut the trajectory into
fixed windows of 10 frames, describe every window in one batched image call (each window's prompt
carries its frame range and actions), then a text-only call merges the window descriptions into
one that cites the frame ranges.

**Ours:**
- **Frames:** the start frame, then each step's unlabelled frame (`k_steps + 1`). Frame `i`
  follows action `i`.
- **Window size from the server limits, not a fixed 10.** `frames_per_window` =
  `min(vllm_max_images, floor(image_budget / tokens_per_frame))`, where:
  - `image_budget` = `vllm_max_model_len − prompt_reserve (2,000) − max_new_tokens`;
  - `tokens_per_frame` = `ceil(w/28) · ceil(h/28)` for the largest frame, which is the Qwen-VL
    patch rule (an estimate for other model families).

  Android comes out at 8 frames, Web at 16, GameBoy at 16. If even one frame is over the budget,
  frames are downscaled to fit; this is the only resizing.
- **One window** (the common GameBoy/Web case at 10 steps): one call, as now.
- **Several windows:**
  - one batched `infer` over the windows, each with the frames and actions of its range
    ("frames 1–8 of 11, actions 1–7");
  - then one text-only compose call over the labelled window descriptions. It returns the final
    `Description:` (citing frame ranges) and the `Discovered:` list (B.1).
  - Windows overlap by one frame, so the frame an action leads from is always in the same window
    as the frame it leads to.
- **Stored:** `Expansion.description` and `discovered` as before, plus `window_descriptions` (the
  per-window texts) for the viewer.
- Limits are read from `parameters` (the same config the server is started with), so they stay in
  sync.
- **Failures:** still crash the run, which resumes from its last commit.
- **Tests:** a 25-step fake expansion on a 1080×2400 canvas gives several windows, each within
  the image and token limits. The compose call gets every window's text. A 10-step GameBoy-sized
  expansion stays one call.

## D. Viewer (done 2026-10-11; see Implementation status)

A static HTML page per run:
- the tree with node thumbnails;
- per node: features, latest score with verdicts, stats;
- click a node or expansion to step through its frames and actions, with its description and
  window descriptions;
- selection probabilities and scores over time.

## Order of work

1. C (sliced describe, with `discovered`), so the `discovered` source exists. Tests with a fake
   VLM.
2. A (features replace the prior). Store and tests.
3. B (decision scores, top-M cache, saved records, rebuilt on load). Tests: scores and verdicts
   persisted, cache rebuilt on resume, and a node whose features are all discovered scores 0.
4. A real-VLM GameBoy run (explorer + describe + features + scores), together with calibrating
   `node_threshold` and `tau` (`plans/archive/tree_search_plan.md`). Then Web and Android.

## Choices (defaults accepted 2026-10-11)

- **Scope of the discovered set:** per scene (proposed), or global across scenes.
- **Top-M re-scoring** (proposed) versus scoring every candidate each decision.
- **Score form:** fraction of undiscovered features (proposed), or the VLM rating the node as a
  whole given features and discovered list.
- **`partly` = 0.5,** and `--default_score` 0.5 for nodes without features.

---

# E. Curiosity policy: a small fused policy with a pointer head, run inside the tree search

## E.0 Where curiosity stands (2026-10-10)

- `run_explore.py ppo`:
  - per-token PPO on a generative VLM policy (Qwen3.5-0.8B + LoRA), one scene, episodes from
    `reset()`;
  - curiosity reward from a `cusi.state` scorer, with the archive reset every episode;
  - writes `replay/`.
- Downstream of `replay/`: `world_model`, `decoder`, `train_embedder`, `elements`, and `tasks`.
  `tasks` takes outlier-novelty stretches, then the VLM infers a task per group, and the result
  is written as an attempts dir for `run_practice.py --source curiosity`.
- Only smoke-tested (2 iterations per env). Known problems:
  - the first update moves the policy too far (KL 0.3–1.2), so every iteration early-stops;
  - one run collapsed to invalid replies;
  - the first frames after a reset score 0.
- Speed: about 1 step/s on GameBoy, 0.15 on Web, 0.06 on Android.
- No code connects it to `cusi.search`; they share only `cusi.state`.

## E.1 Decision: replace the generative policy with a small fused policy and discrete actions

**Decided (2026-10-10):** no text generation. Every step enumerates the legal actions, and a small
policy picks one. It isn't from scratch: it stitches two frozen pretrained encoders (vision,
text) into a small trained fusion network with a pointer head over the candidates.

**Gains:** no invalid actions (they're never candidates), no generation, a cheap and stable PPO
step. The generative policy's problems (KL jumps, collapse into invalid replies, prompt-format
fights) go with it.

**Limit:** sample budget, not the model. GameBoyRL trains for **100,000 steps** per run from one
start state (30-step episodes, about 3,300 episodes), and E.5 runs one such session per expansion.
Measured env speed with no model in the loop (search test runs):
- GameBoy: ~65 steps/s with state saves, faster without.
- Web: ~2.5 s/step.
- Android: ~10 s/step.

At GameBoyRL's budget that is minutes per session on GameBoy, days on Web, and weeks on Android
(E.5.5). The frozen encoders help sample efficiency but don't close a gap of that size.

## E.2 Legal actions (candidates) per env

| Env | Candidates |
|---|---|
| GameBoy | the 7 buttons (UP, DOWN, LEFT, RIGHT, A, B, START), always legal |
| Android | `click` / `long_press` on each interactive element; `scroll` 4 directions on each scrollable element and on the screen; `navigate_back`; `navigate_home`; `open_app` over the fixed installed-app list; `wait`; `type(field, s)` per editable field |
| Web | `click` on each clickable element; `scroll` up/down on the window and on each scrollable element; `go back`; `wait`; `type(field, s)` per input field. Not `google` (leaves the scene) or `answer` (tasks only). |

- Elements come from §0's text boxes plus interactive elements without text (§0, "Required by §E").
- The candidate count is capped (256 elements in reading order).
- Each candidate maps to the env's existing text action, so the env interfaces don't change
  (`Click [i]` on Web, the JSON action on Android, the button name on GameBoy).

**Typing with fixed string sets (decided):** each input field gets up to `--max_type_strings`
(default 8) `type(field, s)` candidates. Sources:
- a short per-env, per-scene list: generic words, plus terms for the scene's site or app (e.g. a
  few research topics on arXiv, repository names on GitHub);
- words from the field's own placeholder, label or `aria-label` ("Search arXiv" gives "search").

The lists live in a small config file per env. The policy learns which string fits which field
through the string's text embedding (E.3).

## E.3 Policy network

```
frame (unlabelled) ──► frozen MobileNetV3-Small trunk ──► feature map F
                                         ├─ 4×4 grid pooling ──► 16 visual tokens
                                         └─ RoI-pool at each element box ─────────────┐
elements ──► frozen MiniLM-L6 text embedding (cached per string) ─────────────────────┤
             role embedding · box geometry (Fourier features + MLP) ──────────────────┴─► element tokens (d=128)
[STATE] + [PREV_ACTION] + 16 visual tokens + ≤256 element tokens ──► 2-layer transformer (d=128, 4 heads)
     z = STATE output (state latent)      h_e = contextualised element tokens
candidate c ──► e_c = MLP(type emb ⊕ target h_e (or learned "no target") ⊕ argument emb)
logit_c = MLP(z ⊕ e_c), softmax over the legal candidates      value = MLP(z)
```

**Vision:** MobileNetV3-Small (ImageNet), frozen, about 2.5M parameters. It is used for its
spatial feature map: grid tokens cover icons, images, sprites and layout, and RoI pooling gives
each element its look.
- **Input:** the **unmasked** unlabelled frame, resized with its aspect kept: tall Android
  screens become a 14×7 map, Web and GameBoy roughly square. Masking stays a novelty-only
  choice (§0.1).

**Text:** all-MiniLM-L6-v2, frozen, 384-dimensional. Each distinct string is embedded once and
cached (strings repeat heavily across screens). The same embeddings serve the arguments: typed
strings and app names. Novelty keeps its own trigram channel (§0.2).

**Element token** = projected text embedding + role embedding + box geometry (normalised
x0, y0, x1, y1, w, h) + projected RoI visual feature. Empty-text elements get a learned "no text"
vector.

**Memory:** the `[PREV_ACTION]` token is the embedding of the last chosen candidate. Optionally
add the previous step's `z`, as GameBoyRL stacks 2 frames.

**GameBoy:** the same network. The candidates are the 7 buttons with no target, so the pointer
head reduces to a fixed 7-way head; the element tokens come from the tile text reader (§0).

**Sizes:** about 25M frozen, under 1M trained. Per step: one MobileNet pass (milliseconds), cached
text, and a transformer over at most ~270 tokens. CPU is possible; GPU if available.

**Later options:** unfreeze the vision trunk's last block; warm-start the head with behaviour
cloning on search or explore trajectories.


## E.4 Training (inside one RL session, E.5)

- **PPO with GameBoyRL's hyperparameters** (`cleanrl/ppo_curiosity.py` and
  `scripts/core/utils.sh`) as the starting point:

  | Setting | Value |
  |---|---|
  | Total steps per session | 100,000 |
  | Episode length | 30 |
  | Rollout | 128 steps |
  | Minibatches × epochs | 4 × 4 |
  | Learning rate | 2.5e-4, annealed |
  | Clip | 0.1 (value loss clipped too) |
  | Entropy / value coefficients | 0.01 / 0.5 |
  | γ / GAE λ | 0.99 / 0.95 |
  | Grad-norm clip | 0.5 |
  | Advantage normalisation | on |

  All are flags (`--rl_total_steps`, `--k_steps`, …). Only the fusion network and heads are
  trained (Adam). Logits outside the candidate set don't exist, so no masking tricks are needed.
- **Reward** per step: the session reward of E.5.1 (item 4), minus `invalid_action_penalty` when
  the env still rejects an action (e.g. a stale element). Return normalisation is optional.
- **Rollout buffer:** each step stores the candidate list, chosen index, log-probability, value
  and entropy, so updates recompute logits on the same candidate sets.
- **Parallel env copies** (`--rl_num_envs`, GameBoy first): several emulator instances in one
  session, each restoring node `n`. Rollouts are collected in parallel and the batch is
  `num_envs × 128`. GameBoyRL used 1; GameBoy emulators are cheap, so this is the main speed-up.

## E.5 Curiosity as an expander: one RL session per expansion

**Decided (2026-10-11).** When the search expands a node with `--expander curiosity`, it trains a
**fresh** policy from that node in a full curiosity RL session. The expansion put into the tree is
**the best episode of the session**, not a rollout of the final policy.

### E.5.1 The session, step by step

For the selected node `n`:

1. **Fresh policy:** a newly initialised fusion network and heads (the frozen encoders are shared
   and pretrained). Nothing carries over between sessions: a policy is specific to its start
   state.
2. **Reward buffer base `B₀`,** fixed for the whole session (`--reward_buffer`):
   - `global` (default): a read-only snapshot of the search's global archive at session start.
     It holds everything the run has committed so far.
   - `local`: empty, except node `n`'s own state, so the start screen isn't rewarded.
3. **Episodes** until `--rl_total_steps` (default 100,000) env steps or `--rl_max_minutes` (default
   60), whichever comes first. Each episode:
   - restores node `n` (state load; restore skipping only for the first);
   - runs up to `k_steps` (default 30) policy actions;
   - stops early if the episode terminates.
4. **Rewards within an episode:** each step is scored against `B₀ ∪ (this episode's earlier
   states)`, then added to the episode's own part. The episode's part is dropped when the episode
   ends.
   - Episodes never pay for, or are penalised by, each other's discoveries, so the reward is
     **stationary** across the session. This is a proper RL objective: find what is new relative
     to `B₀` from `n` in `k_steps`.
   - `local` gives GameBoyRL's within-episode novelty (its `reset_curiosity_module=True`);
     `global` gives novelty relative to the whole run.
5. **PPO updates** per E.4. Truncation at `k_steps` bootstraps from the value; termination
   doesn't.
6. **Best episode:** the episode with the highest **own return** (session reward, in both modes;
   ties go to the earlier one). Its actions, frames, texts, per-step session rewards and episode
   index are kept, along with its saved states (E.5.2). Every other episode is reduced to its
   return.

### E.5.2 Committing the best episode

The env is no longer in the best episode's states, so they must be available at commit:
- **Web: save a state at every step of every episode** (cheap dicts, small next to a 2.5 s step).
  When an episode becomes the new best, its states replace the previous best's; otherwise they're
  deleted when the episode ends.
- **GameBoy: replay the best episode at session end.** Restore `n` and redo its actions. This is
  exact, because GameBoy is deterministic and the replay is checked step by step against the
  recorded frames. Replay is chosen over saving every step because saving costs 0.07 s per step:
  at 100,000 steps per session that adds ~2 hours, while one replay costs under a second. If a
  replay ever doesn't match exactly, the run fails loudly; a determinism bug shouldn't be hidden.
- **Android: replay** (decided earlier: root snapshots only, no per-node snapshots). A snapshot
  per step is impossible (~3.6 GB each). Divergence is checked by embedding similarity
  (`replay_similarity`); the expansion is committed up to the last matching step and flagged
  `materialise_diverged`.

Then the expansion is built like any other expander's:
- **Step values** = the search's scorer against the global archive (`add=True`). They decide which
  steps become nodes (`node_threshold`, last step always a node) and the expansion's yield.
- **In `global` mode** these equal the session rewards, because `B₀` is the archive and nothing was
  added during the session. **In `local` mode** they differ, and the session reward is kept in
  `Step.extra["session_reward"]`.
- **Then as usual:** describe (`--describe`), node features (§A), commit.

The global archive changes **only** at this commit: rejected episodes never enter it.

### E.5.3 The session's final policy

- Each session ends with one final policy, trained from scratch from node `n`.
- **It doesn't start later sessions** (decided: always a fresh policy).
- **It is saved** with the expansion as `policies/<exp_id>.pt` (trained weights only, about 4 MB,
  no optimiser state), as a record of "the explorer trained from node `n`" for analysis or later
  use.
- `--keep_policies {all,none}` (default `all`) controls this; 1,000 expansions ≈ 4 GB.

### E.5.4 Resume and determinism

- A session is atomic: the expansion and its policy file are committed together. A crash
  mid-session loses that session; the resumed run starts it again with a fresh policy, the same
  seeds and the same `B₀`.
- Seeds for initialisation and sampling are fixed by `(seed, iteration)`. On GameBoy a repeated
  session is reproducible up to GPU nondeterminism in the updates; on Web and Android only
  approximately.
- **Session progress** (returns per episode, updates) is logged as the session runs, so a
  multi-hour session is observable.

### E.5.5 Budget at GameBoyRL scale

One session is `rl_total_steps` env steps plus about one restore per 30 steps:

| Env | 100,000-step session | Notes |
|---|---|---|
| GameBoy | ~25 min at 65 steps/s; minutes with `--rl_num_envs 8` and no per-step saves | Feasible. A 200-expansion search ≈ a day on one process before parallel envs. |
| Web | ~70 h per instance (2.5 s/step, plus ~10 s restores every episode) | Not feasible per node at this budget, even with 8 browsers (~9 h per node). |
| Android | ~11.5 days per instance (10 s/step, plus ~45 s replay restore per episode) | Not feasible per node. |

**Decided (2026-10-11): start with at most an hour per session, on every env, and tune from
there.**
- `--rl_max_minutes` (default 60) stops a session at whichever comes first: that wall-clock limit
  or `--rl_total_steps`.
- The per-env starting values come from the measured step speed:
  - GameBoy can approach 100,000 steps within the hour with parallel emulator copies;
  - Web gets roughly 1,000–1,500 steps (about 100–150 episodes of 10 steps);
  - Android roughly 300 (about 30 episodes of 10 steps).
- These are first settings to experiment with, not final values. The session log records steps,
  episodes and time, so they can be re-tuned from real runs.

### E.5.6 Configurations of existing methods

| Method | Configuration |
|---|---|
| GameBoyRL curiosity | `--expander curiosity --roots_only --reward_buffer local --k_steps 30 --max_expansions <scenes>` (one session per root; its best episode is kept, its policy saved) |
| Go-Explore with a learned explorer | `--expander curiosity --reward_buffer global` (energy selection over all nodes) |
| AutoPlay / OS-Genesis style | `--expander vlm` or `random`, with `--roots_only` |

## E.6 Fields that differ between expanders

Every expansion has the same core, whoever produced it:
- `node_id`, `iteration`, `children`, `yield`, `stats` and the selection `record`;
- `steps`, each with `action`, `frame`, `texts` / text boxes, `info`, `value` (global novelty),
  `components` and `node_id`.

So the tree, the stats, selection, restores and storage don't care which expander produced an
expansion.

What differs is kept in two places:
1. **`Expansion.expander`** (`random` | `vlm` | `curiosity`) plus **`Expansion.expander_info`**, a
   dict whose keys depend on the expander:
   - `random`: `{}`.
   - `vlm`: `{"model_name", "context": {"path_actions", "tried_ids"}}` — what the prompt was given.
   - `curiosity`: `{"reward_buffer", "total_steps", "episodes", "returns_summary" (per-100-episode
     mean / max, plus the full list in `curiosity/<exp_id>_returns.npy`), "best_episode",
     "best_return", "updates", "policy": "policies/<exp_id>.pt", "materialise": "replay" |
     "states", "seconds"}`.
2. **`Step.extra`**, namespaced by the same expander:
   - `vlm`: `{"prompt_chars", "reply"}`;
   - `curiosity`: `{"session_reward", "logp", "value", "entropy", "n_candidates", "candidate"}`.
     These come from the best episode as it was played during the session.

**Add-on fields are expander-agnostic.** They depend on the flags, not on who explored:
- `Expansion.description` / `discovered` / `window_descriptions` (`--describe`, §C): the describer
  only needs frames and actions, so curiosity expansions are described exactly like VLM or random
  ones.
- `Node.features` (`--features`, §A) and the decision scores (§B).

They are `null` when the flag is off, and consumers must accept that.

**Consumers:**
- **Generic** (tree, viewer, task synthesis, practice start states) read only the core and the
  add-ons.
- **Expander-specific** (e.g. plotting a session's return curve) branch on `expander`.
- The VLM explorer's "earlier explorations" context uses actions plus `description` of any earlier
  expansion from the node, whoever produced it.

A run has one `--expander`, recorded in `config.json`. The schema still allows mixed runs later.

## E.7 Order of work (after §0)

1. Candidate enumeration per env (E.2) and the fixed string lists. Test that every candidate maps
   to an action the env accepts, on live pages and screens.
2. The policy network (E.3), with frozen encoders and the text cache. Unit tests: candidate sets
   of any size, GameBoy reduces to 7 logits.
3. The core plus `expander` / `expander_info` / namespaced `Step.extra` schema (E.6) in the tree
   and store. Existing random and VLM expanders are migrated to it.
4. `--expander curiosity` on GameBoy:
   - the session (E.5.1), with the stationary reward over `B₀`, `global` and `local`, and parallel
     env copies;
   - replay materialisation (E.5.2);
   - saved policies (E.5.3);
   - atomic commit and resume (E.5.4).

   Fake-env tests:
   - the best episode is the one committed;
   - `global` session rewards equal the committed step values;
   - rejected episodes never touch the global archive;
   - an interrupted session restarts identically.
5. A GameBoy parity check: one root, `local`, 100,000 steps, compared with GameBoyRL's
   `ppo_curiosity` on the same `init_state`. Then the Go-Explore configuration against the random
   and VLM expanders.
6. Web (state saving per step) and Android (replay), at the hour-per-session budgets of E.5.5.
7. Move curiosity `tasks` and the world-model / embedder training to read search runs.

## E.8 Decisions

- Vision trunk: MobileNetV3-Small. Text encoder: MiniLM-L6.
- Android and Web use curiosity at lowered budgets (an hour per session to start, E.5.5).
- Always a fresh policy per session.
- Best episode by its own return in both modes.
- `run_explore.py ppo` stays for now and is **marked for deletion** once step 7 is done. It is the
  generative-policy baseline until then.
