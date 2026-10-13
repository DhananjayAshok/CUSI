"""
GameBoy text from the tile layers, without OCR: every visible background / window cell's tile is reduced to its ink
mask (pixels that differ from the tile's majority colour index, so the key does not depend on the palette or on which
colour index the font uses), and a per-game glyph table maps masks to characters. Each matched cell is checked
against the screen (the cell's pixels must be a recolouring of the tile), so sprites, raster effects and covered
layers do not produce text. Neighbouring characters on a row form runs; runs inside a dialogue region form one box.

Tables live in cusi/envs/gameboy_text/<variant>.json:
    {"variant", "method": "glyphs", "glyphs": {mask_hex: char}, "ignore": [mask_hex, ...],
     "kinds": {ocr_kind: {"role": "dialogue" | "menu", "region": region name or "box": [x0, y0, x1, y1]}},
     "min_chars": 2, "notes": "...",
     optional "pairs": {"<upper key>|<lower key>": char} for 8x16 fonts (a cell is read with the cell above it),
     optional "segments": {segment key: char} for variable-width fonts, read by pixel segmentation inside the kinds'
     regions (cusi/envs/gameboy_text/segments.py)}
A box gets a kind's role when it lies in the kind's GameBoyWorlds region while GameBoyWorlds' OCR metric reports that
kind (can_read_kind); otherwise its role is "text".
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from cusi.envs.text_boxes import TextBox, reading_order

TABLE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREEN_W, SCREEN_H = 160, 144
LCDC, SCY, SCX, WY, WX = 0xFF40, 0xFF42, 0xFF43, 0xFF4A, 0xFF4B
#: A run ends at this many blank cells in a row (one blank cell is a space).
MAX_GAP = 1


def table_path(*, variant: str) -> str:
    return os.path.join(TABLE_DIR, f"{variant}.json")


def load_table(*, variant: str) -> Optional[dict]:
    path = table_path(variant=variant)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def decode_tiles(*, data: np.ndarray) -> np.ndarray:
    """6144 bytes of tile data (one VRAM bank) -> (384, 8, 8) colour indices 0-3."""
    d = np.asarray(data, dtype=np.uint8)[:0x1800].reshape(384, 8, 2)
    lo = np.unpackbits(d[:, :, 0:1], axis=-1)
    hi = np.unpackbits(d[:, :, 1:2], axis=-1)
    return (lo | (hi << 1)).astype(np.uint8)


def ink_masks(*, tiles: np.ndarray) -> np.ndarray:
    """(N, 8, 8) colour indices -> (N, 8, 8) bool, pixels that differ from the tile's most common index."""
    flat = tiles.reshape(len(tiles), 64)
    counts = np.stack([(flat == v).sum(axis=1) for v in range(4)], axis=1)
    majority = counts.argmax(axis=1)
    return (flat != majority[:, None]).reshape(-1, 8, 8)


def mask_keys(*, masks: np.ndarray) -> np.ndarray:
    """(N, 8, 8) bool -> N hex keys (16 characters: the 64-bit mask, row-major)."""
    packed = np.packbits(masks.reshape(len(masks), 64), axis=1)
    return np.array([bytes(row).hex() for row in packed])


BLANK_KEY = "0" * 16


@dataclass
class Cell:
    x: int                 # screen pixel of the cell's left edge (may be < 0 or off the right side)
    y: int
    layer: str             # "bg" | "window"
    key: str               # ink-mask key
    char: Optional[str]    # the table's character; " " for blank; None for unknown or not text
    tile: int = 0          # tile index (0-383) and VRAM bank
    bank: int = 0


@dataclass
class ReadResult:
    boxes: list
    unknown: list = field(default_factory=list)    # keys of unknown inked cells next to text
    cells: list = field(default_factory=list)      # every visible cell (for harvesting)


def kind_role(*, kind: str) -> Optional[str]:
    """The text role of a GameBoyWorlds OCR kind (None: the kind covers the whole screen and says nothing)."""
    if kind == "full_screen":
        return None
    return "menu" if "menu" in kind or "option" in kind else "dialogue"


def emulator_kinds(*, emulator) -> dict:
    """{kind: {"role", "region"}} from the game's OCR metric (GameBoyWorlds' self.kinds)."""
    metric = getattr(emulator, "state_tracker", None) and emulator.state_tracker.metrics.get("ocr")
    kinds = getattr(metric, "kinds", None) or {}
    return {k: {"role": kind_role(kind=k), "region": region} for k, region in kinds.items() if kind_role(kind=k)}


def _runs_in_cgb_mode(pyboy) -> bool:
    """Whether the emulator runs the cartridge as a Game Boy Color (VRAM bank 1 and attribute maps exist)."""
    try:
        pyboy.memory[1, 0x8000]
        return True
    except (AssertionError, IndexError, ValueError):
        return False


class TileTextReader:
    def __init__(self, *, pyboy, table: Optional[dict], named_regions: Optional[dict] = None,
                 kinds: Optional[dict] = None) -> None:
        """table: a loaded glyph table (None: harvesting only, every inked cell is unknown). named_regions: the state
        parser's {name: NamedScreenRegion}, which the kinds' regions refer to. kinds: {kind: {"role", "region"}}
        (default: the table's "kinds"; see emulator_kinds)."""
        self.pyboy = pyboy
        self.table = table or {"glyphs": {}, "ignore": [], "regions": {}}
        self.glyphs = dict(self.table.get("glyphs", {}))
        self.pairs = dict(self.table.get("pairs") or {})
        self.segments = dict(self.table.get("segments") or {})
        self.ignore = set(self.table.get("ignore", []))
        self.min_chars = int(self.table.get("min_chars", 2))
        self.kind_regions = {}        # kind -> (role, (x0, y0, x1, y1))
        for kind, spec in (self.table.get("kinds") or kinds or {}).items():
            if "box" in spec:        # an explicit text area, for games whose named region is only a detector patch
                self.kind_regions[kind] = (spec["role"], tuple(spec["box"]))
                continue
            name = spec["region"]
            if named_regions is None or name not in named_regions:
                raise ValueError(f"glyph table region {name!r} is not a named region of this game")
            self.kind_regions[kind] = (spec["role"], tuple(named_regions[name].get_corners()))
        self.cgb = _runs_in_cgb_mode(pyboy)
        self._active = []

    # ------------------------------------------------------------ VRAM

    def _vram(self, bank: int) -> np.ndarray:
        if self.cgb:
            # PyBoy rejects a banked slice ending at 0xA000, so the last byte is read on its own.
            data = list(self.pyboy.memory[bank, 0x8000:0x9FFF]) + [self.pyboy.memory[bank, 0x9FFF]]
            return np.asarray(data, dtype=np.uint8)
        return np.asarray(self.pyboy.memory[0x8000:0xA000], dtype=np.uint8)

    def _layer_cells(self, *, layer: str, lcdc: int, banks: list, attrs: Optional[np.ndarray], keys: list) -> list:
        """The cells of one layer that are on screen, with their keys."""
        if layer == "bg":
            map_base = 0x1C00 if lcdc & 0x08 else 0x1800
            sx, sy = self.pyboy.memory[SCX], self.pyboy.memory[SCY]
            origins = [((c * 8 - sx % 8), (r * 8 - sy % 8), (sx // 8 + c) % 32, (sy // 8 + r) % 32)
                       for r in range(19) for c in range(21)]
        else:
            map_base = 0x1C00 if lcdc & 0x40 else 0x1800
            wx, wy = self.pyboy.memory[WX] - 7, self.pyboy.memory[WY]
            if wx >= SCREEN_W or wy >= SCREEN_H:
                return []
            origins = [(wx + c * 8, wy + r * 8, c, r) for r in range(32) for c in range(32)
                       if wx + c * 8 < SCREEN_W and wy + r * 8 < SCREEN_H]
        signed = not (lcdc & 0x10)
        vram0 = banks[0]
        out = []
        for x, y, mc, mr in origins:
            index = int(vram0[map_base + 32 * mr + mc])
            tile = (((index ^ 0x80) - 128) + 256) if signed else index
            bank, flip = 0, 0
            if attrs is not None:
                a = int(attrs[map_base + 32 * mr + mc])
                bank, flip = (a >> 3) & 1, (a >> 5) & 3
            key = keys[bank][tile] if flip == 0 else self._flipped_key(bank=bank, tile=tile, flip=flip)
            out.append(Cell(x=x, y=y, layer=layer, key=key, char=None, tile=tile, bank=bank))
        return out

    def _flipped_key(self, *, bank: int, tile: int, flip: int) -> str:
        t = self._tiles[bank][tile]
        if flip & 1:
            t = t[:, ::-1]
        if flip & 2:
            t = t[::-1, :]
        return str(mask_keys(masks=ink_masks(tiles=t[None]))[0])

    def visible_cells(self) -> list:
        """Every on-screen cell of the background and window layers (window cells cover the background)."""
        lcdc = self.pyboy.memory[LCDC]
        if not lcdc & 0x80:
            return []
        banks = [self._vram(0)] + ([self._vram(1)] if self.cgb else [])
        self._tiles = [decode_tiles(data=b) for b in banks]
        keys = [mask_keys(masks=ink_masks(tiles=t)) for t in self._tiles]
        attrs = banks[1] if self.cgb else None
        bg_on = bool(lcdc & 0x01) or self.cgb
        cells = []
        window = []
        if lcdc & 0x20 and (lcdc & 0x01 or self.cgb):
            window = self._layer_cells(layer="window", lcdc=lcdc, banks=banks, attrs=attrs, keys=keys)
        if bg_on:
            bg = self._layer_cells(layer="bg", lcdc=lcdc, banks=banks, attrs=attrs, keys=keys)
            if window:
                wx, wy = self.pyboy.memory[WX] - 7, self.pyboy.memory[WY]
                bg = [c for c in bg if not (c.x + 8 > wx and c.y + 8 > wy)]
            cells += bg
        return cells + window

    # ------------------------------------------------------------ reading

    def _matches_screen(self, *, cell: Cell, screen: np.ndarray, tiles_by_key: dict) -> bool:
        """The screen block at the cell is a recolouring of the glyph's mask (ink pixels one colour set, the
        rest another, disjoint)."""
        if cell.x < 0 or cell.y < 0 or cell.x + 8 > SCREEN_W or cell.y + 8 > SCREEN_H:
            return False
        block = screen[cell.y:cell.y + 8, cell.x:cell.x + 8]
        mask = tiles_by_key[cell.key]
        ink, paper = block[mask], block[~mask]
        if len(paper) == 0 or len(ink) == 0:
            return False
        paper_colours = {int(v) for v in np.unique(paper)}
        ink_colours = {int(v) for v in np.unique(ink)}
        return len(paper_colours) == 1 and not (paper_colours & ink_colours)

    def shows(self, *, cell: Cell, screen: np.ndarray) -> bool:
        """Whether the screen (RGB) at the cell is a recolouring of the cell's tile (it is not covered)."""
        if cell.key == BLANK_KEY:
            return False
        rgb = np.asarray(screen)[:, :, :3].astype(np.int64)
        packed = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
        mask = np.unpackbits(np.frombuffer(bytes.fromhex(cell.key), dtype=np.uint8)).reshape(8, 8).astype(bool)
        return self._matches_screen(cell=cell, screen=packed, tiles_by_key={cell.key: mask})

    def read(self, *, active_kinds=None, screen: Optional[np.ndarray] = None, keep_cells: bool = False) -> ReadResult:
        """Text boxes on the current screen. active_kinds: the OCR kinds GameBoyWorlds reports now (None: all of the
        table's kinds). screen: the RGB(A) frame (default: the emulator's screen)."""
        kinds = list(self.kind_regions) if active_kinds is None else [k for k in active_kinds if k in self.kind_regions]
        self._active = [self.kind_regions[k] for k in kinds]
        cells = self.visible_cells()
        if screen is None:
            screen = self.pyboy.screen.ndarray
        rgb = np.asarray(screen)[:, :, :3].astype(np.int64)
        packed = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
        masks = {}
        for c in cells:
            if c.key == BLANK_KEY:
                c.char = " "
            elif c.key in self.glyphs:
                if c.key not in masks:
                    masks[c.key] = np.unpackbits(np.frombuffer(bytes.fromhex(c.key), dtype=np.uint8)
                                                 ).reshape(8, 8).astype(bool)
                c.char = self.glyphs[c.key] if self._matches_screen(cell=c, screen=packed, tiles_by_key=masks) \
                    else None
        if self.pairs:
            above = {(c.layer, c.x, c.y): c for c in cells}
            for c in cells:
                up = above.get((c.layer, c.x, c.y - 8))
                if up is not None and c.char not in (None, " ") and f"{up.key}|{c.key}" in self.pairs:
                    c.char = self.pairs[f"{up.key}|{c.key}"]
        boxes, unknown = self._runs(cells=cells)
        if self.segments:
            boxes, unknown = boxes + self._segment_boxes(screen=np.asarray(screen), unknown=unknown), unknown
            boxes = reading_order(boxes=boxes)
        return ReadResult(boxes=boxes, unknown=unknown, cells=cells if keep_cells else [])

    def _segment_boxes(self, *, screen: np.ndarray, unknown: list) -> list:
        """Variable-width text, segmented at pixel level inside the active kinds' regions."""
        from cusi.envs.gameboy_text.segments import read_segments
        out = []
        for k, (role, corners) in enumerate(self._active):
            lines, unk = read_segments(screen=screen, corners=corners, table=self.table)
            unknown += unk
            if role == "dialogue" and lines:
                text = " ".join(t for t, _b in lines)
                box = (min(b[0] for _t, b in lines), min(b[1] for _t, b in lines),
                       max(b[2] for _t, b in lines), max(b[3] for _t, b in lines))
                out.append(TextBox(text=text, box=box, role=role, interactive=False, source="tiles"))
            else:
                out += [TextBox(text=t, box=b, role=role, interactive=False, source="tiles") for t, b in lines]
        return out

    def _role_of(self, box: tuple) -> tuple:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for k, (role, (x0, y0, x1, y1)) in enumerate(self._active):
            if x0 <= cx < x1 and y0 <= cy < y1:
                return role, k
        return "text", None

    def _runs(self, *, cells: list) -> tuple:
        rows: dict = {}
        for c in cells:
            rows.setdefault((c.layer, c.y), []).append(c)
        runs, unknown = [], []
        for (_layer, y), row in rows.items():
            row.sort(key=lambda c: c.x)
            cur, gap = [], 0
            for i, c in enumerate(row):
                if c.char is not None and c.char != " ":
                    cur.append(c)
                    gap = 0
                    continue
                if c.char == " " and cur:
                    gap += 1
                    if gap <= MAX_GAP:
                        cur.append(c)
                        continue
                if c.char is None and c.key not in self.ignore and c.key != BLANK_KEY and cur:
                    unknown.append(c.key)     # an inked cell right after text: likely an unlabelled glyph
                runs.append(cur)
                cur, gap = [], 0
                if c.char is None and c.key not in self.ignore and c.key != BLANK_KEY and i + 1 < len(row) \
                        and row[i + 1].char not in (None, " "):
                    unknown.append(c.key)
            runs.append(cur)
        boxes = []
        for run in runs:
            while run and run[-1].char == " ":
                run = run[:-1]
            text = "".join(c.char for c in run)
            if len(text.replace(" ", "")) < self.min_chars:
                continue
            box = (run[0].x, run[0].y, run[-1].x + 8, run[0].y + 8)
            role, region = self._role_of(box)
            boxes.append((role, region, text, box))
        return self._blocks(runs=boxes), unknown

    @staticmethod
    def _blocks(*, runs: list) -> list:
        """Runs inside one dialogue region become one box (lines joined in reading order); others stay apart."""
        out, dialogue = [], {}
        for role, region, text, box in runs:
            if role == "dialogue":
                dialogue.setdefault(region, []).append((text, box))
                continue
            out.append(TextBox(text=text, box=box, role=role, interactive=False, source="tiles"))
        for region, parts in dialogue.items():
            parts.sort(key=lambda p: (p[1][1], p[1][0]))
            box = (min(p[1][0] for p in parts), min(p[1][1] for p in parts), max(p[1][2] for p in parts),
                   max(p[1][3] for p in parts))
            out.append(TextBox(text=" ".join(p[0] for p in parts), box=box, role="dialogue", interactive=False,
                               source="tiles"))
        return reading_order(boxes=out)
