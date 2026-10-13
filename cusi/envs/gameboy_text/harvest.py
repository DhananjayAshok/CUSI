"""
Build a game's glyph table (cusi/envs/gameboy_text/<variant>.json) from the tiles it actually shows.

    python -m cusi.envs.gameboy_text.harvest collect --variant pokemon_red     # play every saved state, keep tiles
    python -m cusi.envs.gameboy_text.harvest sheet --variant pokemon_red       # glyph sheets to label by eye
    python -m cusi.envs.gameboy_text.harvest seed --variant pokemon_red --charmap gen1   # Pokémon: label from ids
    python -m cusi.envs.gameboy_text.harvest build --variant pokemon_red       # labels -> the table
    python -m cusi.envs.gameboy_text.harvest check --variant pokemon_red       # overlays + unknown glyph report

collect plays each of GameBoyWorlds' saved states (plus random buttons, A-heavy to advance text) and records every
inked tile with how often it appeared, how often at a text moment (GameBoyWorlds' OCR metric reported a text region)
and a screen strip around one occurrence. Labels are written by hand into labels.json next to the sheets:
{"<sheet index>": "<char>"} ("" = not text: added to the table's ignore list).
"""
import json
import os
import pickle
from collections import Counter
import click
import numpy as np
from PIL import Image, ImageDraw
from cusi.utils.parameter_handling import load_parameters
from cusi.envs.gameboy_text.reader import BLANK_KEY, TileTextReader, emulator_kinds, load_table, table_path

#: Variants whose GameBoyWorlds data lacks some region captures, so they only load in its debug_mode.
NEEDS_DEBUG_MODE = ("pokemon_brown", "pokemon_starbeasts", "harry_potter_chamber_of_secrets")
BUTTON_P = {"A": 0.35, "B": 0.08, "UP": 0.13, "DOWN": 0.13, "LEFT": 0.13, "RIGHT": 0.13, "START": 0.05}

# Pokémon character maps (tile id -> character), from the pokered / pokecrystal charmaps.
_GEN_COMMON = {0x7F: " ", **{0x80 + i: chr(ord("A") + i) for i in range(26)}, 0x9A: "(", 0x9B: ")", 0x9C: ":",
               0x9D: ";", 0x9E: "[", 0x9F: "]", **{0xA0 + i: chr(ord("a") + i) for i in range(26)}, 0xBA: "é",
               0xE0: "'", 0xE3: "-", 0xE6: "?", 0xE7: "!", 0xE8: ".", 0xED: "▶", 0xEE: "▼", 0xEF: "♂",
               0xF0: "¥", 0xF1: "×", 0xF3: "/", 0xF4: ",", 0xF5: "♀", **{0xF6 + i: str(i) for i in range(10)}}
CHARMAPS = {
    "gen1": {**_GEN_COMMON, 0xBB: "'d", 0xBC: "'l", 0xBD: "'s", 0xBE: "'t", 0xBF: "'v", 0xE1: "PK", 0xE2: "MN",
             0xE4: "'r", 0xE5: "'m", 0xF2: "."},
    # Gen 2's contraction tiles are left out: hacks (Fool's Gold) reuse those ids for box borders.
    "gen2": {**_GEN_COMMON, 0xE1: "PK", 0xE2: "MN", 0xF2: ".", 0xEA: "é"},
}


def work_dir(*, parameters: dict, variant: str) -> str:
    d = os.path.join(parameters["storage_dir"], "gameboy_text", variant)
    os.makedirs(d, exist_ok=True)
    return d


def make_env(*, variant: str, parameters: dict):
    from cusi.envs.gameboy import GameBoyPlayEnv
    return GameBoyPlayEnv(game=variant, mode="free_play", session_name=f"harvest_{variant}",
                          gameboy_worlds_debug=variant in NEEDS_DEBUG_MODE, parameters=parameters)


def saved_states(*, env) -> list:
    states_dir = os.path.join(env._env._emulator.state_parser.rom_data_path, "states")
    return sorted(f[:-len(".state")] for f in os.listdir(states_dir)
                  if f.endswith(".state") and not f.startswith("custom_"))


def load_named_state(*, env, name: str):
    emulator = env._env._emulator
    emulator.set_init_state(name)
    env._fresh = False
    return env.reset()


def active_kinds(*, info: dict) -> list:
    """The OCR kinds GameBoyWorlds reported at this step (or in its transition)."""
    regions = info.get("text_regions") or {}
    kinds = set(regions.get("ocr_regions") or {})
    for transition in regions.get("transition_ocr_regions") or []:
        kinds |= set(transition)
    return sorted(kinds)


def is_text_moment(*, info: dict) -> bool:
    """GameBoyWorlds' OCR metric reported a text region at this step or during its transition."""
    regions = info.get("text_regions") or {}
    return bool(regions.get("ocr_regions")) or bool(regions.get("transition_ocr_regions"))


def reader_for(*, env, table) -> TileTextReader:
    emulator = env._env._emulator
    return TileTextReader(pyboy=emulator._pyboy, table=table, named_regions=emulator.state_parser.named_screen_regions,
                          kinds=emulator_kinds(emulator=emulator))


@click.group()
def main():
    pass


@main.command()
@click.option("--variant", required=True)
@click.option("--steps", default=150, show_default=True, help="Random steps after loading each saved state.")
@click.option("--max_states", default=60, show_default=True)
@click.option("--seed", default=0, show_default=True)
def collect(variant, steps, max_states, seed):
    parameters = load_parameters()
    out = work_dir(parameters=parameters, variant=variant)
    env = make_env(variant=variant, parameters=parameters)
    reader = reader_for(env=env, table=None)
    rng = np.random.default_rng(seed)
    stats: dict = {}
    names, p = list(BUTTON_P), np.array(list(BUTTON_P.values()))
    states = saved_states(env=env)
    if len(states) > max_states:
        states = [states[i] for i in sorted(rng.choice(len(states), max_states, replace=False))]
    frames_dir = os.path.join(out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    n_frames = n_text = 0
    for name in states:
        try:
            obs, info = load_named_state(env=env, name=name)
        except Exception as e:
            print(f"skip state {name}: {e}")
            continue
        for t in range(steps + 1):
            if t > 0:
                obs, _r, _te, _tr, info = env.step(names[int(rng.choice(len(names), p=p))])
            text_moment = is_text_moment(info=info)
            screen = np.asarray(env._env._emulator._pyboy.screen.ndarray)[:, :, :3]
            cells = reader.visible_cells()
            n_frames += 1
            n_text += text_moment
            if text_moment and n_text % 5 == 1:
                Image.fromarray(screen).save(os.path.join(frames_dir, f"{name}_{t}.png"))
            for c in cells:
                if c.key == BLANK_KEY:
                    continue
                s = stats.get(c.key)
                if s is None:
                    s = stats[c.key] = {"count": 0, "text_count": 0, "tiles": Counter(), "strip": None,
                                        "text_tiles": Counter()}
                s["count"] += 1
                s["tiles"][(c.bank, c.tile)] += 1
                if text_moment:
                    s["text_count"] += 1
                    s["text_tiles"][(c.bank, c.tile)] += 1
                if s.get("strip_rank", -1) < 3 and 0 <= c.y <= 136 and 0 <= c.x <= 152:
                    # Prefer an occurrence the screen really shows (not covered), then one at a text moment.
                    shown = reader.shows(cell=c, screen=screen)
                    rank = 2 * shown + text_moment
                    if rank > s.get("strip_rank", -1):
                        x0, x1 = max(0, c.x - 24), min(160, c.x + 32)
                        s["strip"] = (screen[c.y:c.y + 8, x0:x1].copy(), c.x - x0)
                        s["strip_rank"] = rank
    env.close()
    with open(os.path.join(out, "harvest.pkl"), "wb") as f:
        pickle.dump({"variant": variant, "stats": stats, "n_frames": n_frames, "n_text": n_text,
                     "states": states}, f)
    print(f"{variant}: {len(states)} states, {n_frames} frames ({n_text} text moments), {len(stats)} inked tiles")


def _load(variant: str, parameters: dict) -> tuple:
    out = work_dir(parameters=parameters, variant=variant)
    with open(os.path.join(out, "harvest.pkl"), "rb") as f:
        return out, pickle.load(f)


def _mask(key: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(key), dtype=np.uint8)).reshape(8, 8).astype(bool)


def reference_glyphs(*, exclude: str) -> tuple:
    """(masks (N, 8, 8) bool, chars) of every other table's labelled glyphs, for label proposals."""
    from cusi.envs.gameboy_text.reader import TABLE_DIR
    masks, chars = [], []
    for name in sorted(os.listdir(TABLE_DIR)):
        if not name.endswith(".json") or name == f"{exclude}.json":
            continue
        with open(os.path.join(TABLE_DIR, name)) as f:
            for key, char in json.load(f)["glyphs"].items():
                masks.append(_mask(key))
                chars.append(char)
    return (np.stack(masks) if masks else np.zeros((0, 8, 8), dtype=bool)), chars


def propose(*, key: str, refs: tuple) -> tuple:
    """(char, pixel distance) of the nearest reference glyph, allowing one-pixel shifts."""
    masks, chars = refs
    if not chars:
        return None, 64
    m = _mask(key)
    best = np.full(len(chars), 64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
            best = np.minimum(best, (masks != shifted[None]).sum(axis=(1, 2)))
    i = int(best.argmin())
    return chars[i], int(best[i])


def glyph_like(*, key: str) -> bool:
    """Ink covers a plausible share of the cell and leaves the border mostly clear."""
    m = _mask(key)
    frac = m.mean()
    return 0.04 <= frac <= 0.6


@main.command()
@click.option("--variant", required=True)
@click.option("--min_text", default=1, show_default=True, help="Keep tiles seen at a text moment at least this often.")
@click.option("--max_glyphs", default=320, show_default=True)
@click.option("--all_tiles", is_flag=True, help="Also tiles never seen at a text moment (games without OCR kinds).")
@click.option("--unknown", "from_check", is_flag=True, help="Only the unknown glyphs next to text from the last check.")
def sheet(variant, min_text, max_glyphs, all_tiles, from_check):
    parameters = load_parameters()
    out, h = _load(variant, parameters)
    if os.path.exists(os.path.join(out, "labels.json")):
        raise click.UsageError(f"{out}/labels.json is not applied yet: run build first")
    table = load_table(variant=variant) or {}
    known = set(table.get("glyphs", {})) | set(table.get("ignore", []))
    stats = h["stats"]
    keys = [k for k, s in stats.items() if glyph_like(key=k) and k not in known
            and (s["text_count"] >= min_text or all_tiles)]
    keys.sort(key=lambda k: (-stats[k]["text_count"], -stats[k]["count"]))
    if from_check:
        with open(os.path.join(out, "check.json")) as f:
            keys = [k for k, _n in json.load(f)["unknown"] if k not in known and k in stats]
    keys = keys[:max_glyphs]
    per_sheet, cols = 64, 4
    index = {}
    refs = reference_glyphs(exclude=variant)
    proposals = {}
    for s0 in range(0, len(keys), per_sheet):
        chunk = keys[s0:s0 + per_sheet]
        cell_w, cell_h = 4 * 56 + 90, 4 * 8 + 14
        rows = (len(chunk) + cols - 1) // cols
        img = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for i, key in enumerate(chunk):
            n = s0 + i
            index[str(n)] = key
            r, c = divmod(i, cols)
            x, y = c * cell_w, r * cell_h
            glyph = Image.fromarray(np.where(_mask(key), 0, 255).astype(np.uint8)).resize((40, 40), Image.NEAREST)
            draw.text((x + 2, y + 14), str(n), fill=(200, 0, 0))
            char, dist = propose(key=key, refs=refs)
            if char is not None and dist <= 10:
                proposals[str(n)] = char
                draw.text((x + 72, y + 30), f"{char}", fill=(0, 150, 0) if dist <= 4 else (160, 160, 0))
            draw.rectangle([x + 27, y + 1, x + 68, y + 42], outline=(0, 120, 255))
            img.paste(glyph.convert("RGB"), (x + 28, y + 2))
            strip = stats[key]["strip"]
            if strip is not None:
                pix, off = strip
                st = Image.fromarray(pix).resize((pix.shape[1] * 4, 32), Image.NEAREST)
                img.paste(st, (x + 80, y + 6))
                colour = (255, 0, 0) if stats[key].get("strip_rank", 0) >= 2 else (255, 170, 0)
                draw.rectangle([x + 80 + off * 4, y + 5, x + 80 + off * 4 + 32, y + 38], outline=colour)
        img.save(os.path.join(out, f"sheet_{s0 // per_sheet}.png"))
    with open(os.path.join(out, "sheet_index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(out, "proposals.json"), "w") as f:
        json.dump(proposals, f, ensure_ascii=False)
    print(f"{variant}: {len(keys)} unlabelled glyph-like tiles on {(len(keys) + per_sheet - 1) // per_sheet} sheets "
          f"in {out}")


@main.command()
@click.option("--variant", required=True)
@click.option("--mode", required=True, type=click.Choice(["segments", "pairs"]))
@click.option("--steps", default=120, show_default=True)
@click.option("--max_states", default=60, show_default=True)
@click.option("--seed", default=0, show_default=True)
def collect_shapes(variant, mode, steps, max_states, seed):
    """For games the 8x8 table cannot read: variable-width glyphs segmented at pixel level inside the OCR kinds'
    regions (segments), or the tile above each labelled glyph of an 8x16 font (pairs). Writes shapes_<mode>.pkl."""
    from cusi.envs.gameboy_text.segments import segments_at
    parameters = load_parameters()
    out = work_dir(parameters=parameters, variant=variant)
    env = make_env(variant=variant, parameters=parameters)
    table = load_table(variant=variant)
    reader = reader_for(env=env, table=table)
    rng = np.random.default_rng(seed)
    names, p = list(BUTTON_P), np.array(list(BUTTON_P.values()))
    states = saved_states(env=env)
    if len(states) > max_states:
        states = [states[i] for i in sorted(rng.choice(len(states), max_states, replace=False))]
    stats: dict = {}
    for name in states:
        try:
            obs, info = load_named_state(env=env, name=name)
        except Exception as e:
            print(f"skip state {name}: {e}")
            continue
        for t in range(steps + 1):
            if t > 0:
                obs, _r, _te, _tr, info = env.step(names[int(rng.choice(len(names), p=p))])
            screen = np.asarray(env._env._emulator._pyboy.screen.ndarray)[:, :, :3]
            found = []
            if mode == "segments":
                for kind in active_kinds(info=info):
                    if kind in reader.kind_regions:
                        found += segments_at(screen=screen, corners=reader.kind_regions[kind][1])
            else:
                cells = reader.visible_cells()
                at = {(c.layer, c.x, c.y): c for c in cells}
                for c in cells:
                    up = at.get((c.layer, c.x, c.y - 8))
                    if c.key in reader.glyphs and up is not None and up.key != BLANK_KEY and up.key not in reader.glyphs \
                            and 8 <= c.y <= 136 and 0 <= c.x <= 152:
                        x0, x1 = max(0, c.x - 24), min(160, c.x + 32)
                        found.append((f"{up.key}|{c.key}", screen[c.y - 8:c.y + 8, x0:x1].copy(), (c.x - x0, 8)))
            for key, strip, pos in found:
                s = stats.setdefault(key, {"count": 0, "strip": (strip, pos)})
                s["count"] += 1
    env.close()
    with open(os.path.join(out, f"shapes_{mode}.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"{variant}: {len(stats)} distinct {mode}")


def _shape_image(key: str) -> np.ndarray:
    """A shape key as a black-on-white bitmap."""
    from cusi.envs.gameboy_text.segments import key_bitmap
    if "|" in key:
        up, low = key.split("|")
        bits = np.concatenate([_mask(up), _mask(low)], axis=0)
    else:
        bits = key_bitmap(key=key)
    return np.where(bits, 0, 255).astype(np.uint8)


@main.command()
@click.option("--variant", required=True)
@click.option("--mode", required=True, type=click.Choice(["segments", "pairs"]))
@click.option("--min_count", default=2, show_default=True)
@click.option("--max_shapes", default=256, show_default=True)
def sheet_shapes(variant, mode, min_count, max_shapes):
    """Sheets of unlabelled shapes from collect-shapes (same labels.json / build workflow as sheet)."""
    parameters = load_parameters()
    out = work_dir(parameters=parameters, variant=variant)
    if os.path.exists(os.path.join(out, "labels.json")):
        raise click.UsageError(f"{out}/labels.json is not applied yet: run build first")
    with open(os.path.join(out, f"shapes_{mode}.pkl"), "rb") as f:
        stats = pickle.load(f)
    table = load_table(variant=variant) or {}
    known = set(table.get("segments") or {}) | set(table.get("pairs") or {}) | set(table.get("ignore", []))
    keys = sorted((k for k, s in stats.items() if s["count"] >= min_count and k not in known),
                  key=lambda k: -stats[k]["count"])[:max_shapes]
    per_sheet, cols = 48, 4
    index = {}
    for s0 in range(0, len(keys), per_sheet):
        chunk = keys[s0:s0 + per_sheet]
        cell_w, cell_h = 330, 72
        rows = (len(chunk) + cols - 1) // cols
        img = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for i, key in enumerate(chunk):
            n = s0 + i
            index[str(n)] = key
            r, c = divmod(i, cols)
            x, y = c * cell_w, r * cell_h
            bm = _shape_image(key)
            scale = max(1, min(4, 64 // max(bm.shape)))
            g = Image.fromarray(bm).resize((bm.shape[1] * scale, bm.shape[0] * scale), Image.NEAREST)
            draw.text((x + 2, y + 2), str(n), fill=(200, 0, 0))
            img.paste(g.convert("RGB"), (x + 26, y + 2))
            draw.rectangle([x + 25, y + 1, x + 26 + g.width, y + 2 + g.height], outline=(0, 120, 255))
            pix, (px, pw) = stats[key]["strip"]
            st = Image.fromarray(pix).resize((pix.shape[1] * 3, pix.shape[0] * 3), Image.NEAREST)
            img.paste(st, (x + 100, y + 4))
            draw.rectangle([x + 100 + px * 3, y + 3, x + 100 + (px + pw) * 3, y + 5 + pix.shape[0] * 3],
                           outline=(255, 0, 0))
        img.save(os.path.join(out, f"sheet_{s0 // per_sheet}.png"))
    with open(os.path.join(out, "sheet_index.json"), "w") as f:
        json.dump(index, f)
    print(f"{variant}: {len(keys)} unlabelled {mode} on {(len(keys) + per_sheet - 1) // per_sheet} sheets in {out}")


@main.command()
@click.option("--variant", required=True)
@click.option("--charmap", required=True, type=click.Choice(sorted(CHARMAPS)))
@click.option("--min_share", default=0.9, show_default=True,
              help="A key is labelled when this share of its text-moment occurrences use one charmap tile id.")
def seed(variant, charmap, min_share):
    """Label keys from Pokémon's fixed font tile ids (bank 0, tiles 0x80-0xFF in unsigned addressing)."""
    parameters = load_parameters()
    out, h = _load(variant, parameters)
    cmap = CHARMAPS[charmap]
    labels, conflicts = {}, []
    for key, s in h["stats"].items():
        if not s["text_count"]:
            continue
        chars = Counter()
        for (bank, tile), n in s["text_tiles"].items():
            if bank == 0 and tile in cmap:
                chars[cmap[tile]] += n
        if not chars:
            continue
        char, n = chars.most_common(1)[0]
        if n / s["text_count"] >= min_share and char.strip():
            labels[key] = char
        elif char.strip():
            conflicts.append((key, dict(chars), s["text_count"]))
    with open(os.path.join(out, "seed_labels.json"), "w") as f:
        json.dump(labels, f, indent=0, ensure_ascii=False)
    print(f"{variant}: seeded {len(labels)} glyphs ({''.join(sorted(set(labels.values())))}); "
          f"{len(conflicts)} ambiguous: {conflicts[:10]}")


@main.command()
@click.option("--variant", required=True)
@click.option("--charmap", required=True, type=click.Choice(sorted(CHARMAPS)))
@click.option("--steps", default=80, show_default=True)
@click.option("--max_states", default=30, show_default=True)
@click.option("--min_share", default=0.8, show_default=True)
@click.option("--min_anchor", default=0.9, show_default=True,
              help="Use a frame's VRAM only if this share of the screen-seeded glyphs sit at their charmap ids.")
def seed_vram(variant, charmap, steps, max_states, min_share, min_anchor):
    """Pokémon: at text moments the font sits at its fixed tile ids, so read the key of every charmap id from VRAM
    (bank 0, unsigned) and label it, also for characters the play never showed. Sprites can occupy those ids, so a
    frame counts only when the glyphs already seeded from the screen (seed) are where the charmap says. Adds to
    seed_labels.json."""
    from cusi.envs.gameboy_text.reader import decode_tiles, ink_masks, mask_keys
    parameters = load_parameters()
    out = work_dir(parameters=parameters, variant=variant)
    cmap = CHARMAPS[charmap]
    env = make_env(variant=variant, parameters=parameters)
    reader = reader_for(env=env, table=None)
    rng = np.random.default_rng(0)
    names, p = list(BUTTON_P), np.array(list(BUTTON_P.values()))
    states = saved_states(env=env)
    if len(states) > max_states:
        states = [states[i] for i in sorted(rng.choice(len(states), max_states, replace=False))]
    with open(os.path.join(out, "seed_labels.json")) as f:
        anchors = json.load(f)
    char_tiles: dict = {}
    for tile, char in cmap.items():
        char_tiles.setdefault(char, []).append(tile)
    votes: dict = {}
    n_used = n_seen = 0
    for name in states:
        obs, info = load_named_state(env=env, name=name)
        for t in range(steps):
            obs, _r, _te, _tr, info = env.step(names[int(rng.choice(len(names), p=p))])
            if not is_text_moment(info=info):
                continue
            keys = mask_keys(masks=ink_masks(tiles=decode_tiles(data=reader._vram(0))))
            n_seen += 1
            placed = [any(keys[t] == key for t in char_tiles.get(char, [])) for key, char in anchors.items()]
            if not placed or sum(placed) / len(placed) < min_anchor:
                continue
            n_used += 1
            for tile, char in cmap.items():
                if char.strip() and keys[tile] != BLANK_KEY:
                    votes.setdefault(str(keys[tile]), Counter())[char] += 1
    env.close()
    path = os.path.join(out, "seed_labels.json")
    labels = {}
    if os.path.exists(path):
        with open(path) as f:
            labels = json.load(f)
    added = 0
    for key, counter in votes.items():
        char, n = counter.most_common(1)[0]
        if n / sum(counter.values()) >= min_share and key not in labels:
            labels[key] = char
            added += 1
    with open(path, "w") as f:
        json.dump(labels, f, indent=0, ensure_ascii=False)
    print(f"{variant}: {n_used}/{n_seen} text-moment frames had the font in place; +{added} glyphs from VRAM; now "
          f"{len(labels)}: {''.join(sorted(set(labels.values())))}")


@main.command()
@click.option("--variant", required=True)
@click.option("--kinds", "regions", default=None,
              help='JSON {ocr_kind: {"role": "dialogue" | "menu", "region": region name}}; default: keep.')
@click.option("--notes", default=None)
@click.option("--min_chars", default=None, type=int)
def build(variant, regions, notes, min_chars):
    """Merge seed_labels.json and labels.json (sheet index -> char) into the table, keeping earlier labels."""
    parameters = load_parameters()
    out, _h = _load(variant, parameters)
    table = load_table(variant=variant) or {"variant": variant, "method": "glyphs", "glyphs": {}, "ignore": [],
                                            "kinds": None, "min_chars": 2, "notes": ""}
    glyphs, ignore = dict(table["glyphs"]), set(table["ignore"])
    seed_path = os.path.join(out, "seed_labels.json")
    if os.path.exists(seed_path):
        with open(seed_path) as f:
            glyphs.update(json.load(f))
    labels_path = os.path.join(out, "labels.json")
    applied = 0
    if os.path.exists(labels_path):
        with open(os.path.join(out, "sheet_index.json")) as f:
            index = json.load(f)
        with open(labels_path) as f:
            labels = json.load(f)
        segments, pairs = dict(table.get("segments") or {}), dict(table.get("pairs") or {})
        for n, char in labels.items():
            key = index[str(n)]
            # segment keys ("HxW:bits") and 8x16 pair keys ("upper|lower") go to their own tables
            target = segments if ":" in key else pairs if "|" in key else glyphs
            if char == "":
                ignore.add(key)
                target.pop(key, None)
            else:
                target[key] = char
                ignore.discard(key)
        if segments:
            table["segments"] = dict(sorted(segments.items()))
        if pairs:
            table["pairs"] = dict(sorted(pairs.items()))
        # Labels refer to the current sheet_index.json, which the next sheet overwrites: archive them.
        while os.path.exists(os.path.join(out, f"labels_applied_{applied}.json")):
            applied += 1
        os.rename(labels_path, os.path.join(out, f"labels_applied_{applied}.json"))
    table["glyphs"], table["ignore"] = dict(sorted(glyphs.items())), sorted(ignore)
    if regions is not None:
        table["kinds"] = json.loads(regions)
    if notes is not None:
        table["notes"] = notes
    if min_chars is not None:
        table["min_chars"] = min_chars
    with open(table_path(variant=variant), "w") as f:
        json.dump(table, f, indent=1, ensure_ascii=False)
    print(f"{variant}: {len(glyphs)} glyphs ({''.join(sorted(set(glyphs.values())))}), {len(ignore)} ignored")


@main.command()
@click.option("--variant", required=True)
@click.option("--steps", default=60, show_default=True)
@click.option("--max_states", default=12, show_default=True)
@click.option("--seed", "seed_", default=1, show_default=True)
def check(variant, steps, max_states, seed_):
    """Read text at text moments of saved states: overlays to inspect, decoded strings, unknown glyph counts."""
    from cusi.envs.text_boxes import overlay
    parameters = load_parameters()
    out = work_dir(parameters=parameters, variant=variant)
    table = load_table(variant=variant)
    if table is None:
        raise click.UsageError(f"No table for {variant}; build it first")
    env = make_env(variant=variant, parameters=parameters)
    reader = reader_for(env=env, table=table)
    rng = np.random.default_rng(seed_)
    names, p = list(BUTTON_P), np.array(list(BUTTON_P.values()))
    states = saved_states(env=env)
    if len(states) > max_states:
        states = [states[i] for i in sorted(rng.choice(len(states), max_states, replace=False))]
    check_dir = os.path.join(out, "check")
    os.makedirs(check_dir, exist_ok=True)
    n_text = n_text_read = n_frames = n_with_boxes = 0
    unknown = Counter()
    strings = Counter()
    for name in states:
        obs, info = load_named_state(env=env, name=name)
        for t in range(steps + 1):
            if t > 0:
                obs, _r, _te, _tr, info = env.step(names[int(rng.choice(len(names), p=p))])
            res = reader.read(active_kinds=active_kinds(info=info))
            text_moment = is_text_moment(info=info)
            n_frames += 1
            n_with_boxes += bool(res.boxes)
            n_text += text_moment
            n_text_read += text_moment and any(b.role in ("dialogue", "menu") for b in res.boxes)
            unknown.update(res.unknown)
            for b in res.boxes:
                strings[f"{b.role}: {b.text}"] += 1
            if (text_moment or res.boxes) and rng.random() < 0.15:
                frame = np.asarray(env._env._emulator._pyboy.screen.ndarray)[:, :, :3]
                overlay(frame=frame, boxes=res.boxes, scale=3).save(os.path.join(check_dir, f"{name}_{t}.png"))
    env.close()
    report = {"variant": variant, "frames": n_frames, "frames_with_boxes": n_with_boxes, "text_moments": n_text,
              "text_moments_with_dialogue_or_menu_box": n_text_read, "unknown_glyph_cells": sum(unknown.values()),
              "distinct_unknown": len(unknown), "unknown": unknown.most_common(), "top_strings": strings.most_common(60)}
    with open(os.path.join(out, "check.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("top_strings", "unknown")}))
    for s, n in strings.most_common(60):
        print(f"  {n:4d}  {s[:120]}")


if __name__ == "__main__":
    main()
