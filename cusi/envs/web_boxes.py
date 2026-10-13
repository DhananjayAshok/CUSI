"""
Web text boxes from the laid-out DOM, in one execute_script: the visible text runs (Range.getClientRects, checked with
elementFromPoint), grouped per element unit (a set-of-mark element, else the nearest block or role element), plus
every set-of-mark element as an interactive box (its text, else aria-label / title / alt / value / placeholder).
"""
from cusi.envs.text_boxes import TextBox, clean_text, clip_box, reading_order

#: At most this many non-interactive boxes per page (set-of-mark elements are always kept).
MAX_CONTENT_BOXES = 300
MAX_BOX_CHARS = 1000

# arguments[0]: the set-of-mark elements (WebVoyager's web_eles), in label order. Returns
# [devicePixelRatio, [[text, x0, y0, x1, y1, role, interactive, ref, scrollable], ...]] in CSS pixels of the viewport.
TEXT_BOXES_JS = r"""
const marked = arguments[0] || [];
const maxChars = arguments[1];
const vw = window.innerWidth, vh = window.innerHeight;
const markIndex = new Map();
marked.forEach((el, i) => { if (el) markIndex.set(el, i); });
const BLOCK_ROLE_TAGS = new Set(['A', 'BUTTON', 'LABEL', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LI', 'TD', 'TH',
  'P', 'SUMMARY', 'OPTION', 'CAPTION', 'FIGCAPTION', 'DT', 'DD', 'LEGEND']);
const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'IFRAME', 'TEXTAREA', 'SELECT']);
const styleCache = new Map();
function style(el) { let s = styleCache.get(el); if (!s) { s = getComputedStyle(el); styleCache.set(el, s); } return s; }
function shown(el) {
  if (el.checkVisibility) return el.checkVisibility({opacityProperty: true, visibilityProperty: true});
  for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
    const s = style(e);
    if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
  }
  return true;
}
const unitCache = new Map();
function unitOf(el) {
  // The marked element containing el, else the nearest block-level or role element.
  if (unitCache.has(el)) return unitCache.get(el);
  let unit = null, block = null;
  for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
    if (markIndex.has(e)) { unit = e; break; }
    if (!block && (BLOCK_ROLE_TAGS.has(e.tagName) || e.hasAttribute('role') ||
                   !style(e).display.startsWith('inline'))) block = e;
  }
  unit = unit || block || el;
  unitCache.set(el, unit);
  return unit;
}
function roleOf(el) {
  for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
    const t = e.tagName, r = (e.getAttribute('role') || '').toLowerCase();
    if (/^H[1-6]$/.test(t) || r === 'heading') return 'heading';
    if (t === 'INPUT' && /^(submit|button|reset|image|checkbox|radio|file|color|range)$/i.test(e.getAttribute('type') || '')) return 'button';
    if (t === 'SELECT') return 'button';
    if (t === 'INPUT' || t === 'TEXTAREA' || r === 'textbox' || r === 'searchbox' || r === 'combobox') return 'input';
    if (t === 'BUTTON' || r === 'button') return 'button';
    if (t === 'A' || r === 'link') return 'link';
    if (r === 'menuitem' || r === 'tab' || (t === 'LI' && e.closest('nav, [role=navigation], [role=menu], [role=menubar], [role=tablist]'))) return 'menu';
    if (t === 'LABEL' || t === 'FIGCAPTION' || t === 'CAPTION' || t === 'LEGEND') return 'label';
    if (t === 'P' || t === 'ARTICLE' || t === 'MAIN' || t === 'SECTION' || t === 'BODY') return 'body';
  }
  return 'body';
}
function clickable(el) {
  // WebVoyager's rule: form fields, buttons, links, onclick or a pointer cursor.
  for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
    const t = e.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || t === 'BUTTON' || t === 'A' || e.onclick != null) return true;
    if (e === document.body) break;
  }
  return style(el).cursor === 'pointer';
}
function scrollable(el) {
  const s = style(el);
  return (s.overflowY === 'auto' || s.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 4;
}
function visibleAt(el, r) {
  const x = r.left + r.width / 2, y = r.top + r.height / 2;
  if (x < 0 || y < 0 || x >= vw || y >= vh) return false;
  const hit = document.elementFromPoint(x, y);
  return !!hit && (hit === el || el.contains(hit) || hit.contains(el));
}
const groups = new Map();   // unit element -> {parts: [], rect}
function addTo(unit, text, r) {
  let g = groups.get(unit);
  if (!g) { g = {parts: [], x0: 1e9, y0: 1e9, x1: -1e9, y1: -1e9, order: groups.size}; groups.set(unit, g); }
  if (text) g.parts.push(text);
  g.x0 = Math.min(g.x0, Math.max(0, r.left)); g.y0 = Math.min(g.y0, Math.max(0, r.top));
  g.x1 = Math.max(g.x1, Math.min(vw, r.right)); g.y1 = Math.max(g.y1, Math.min(vh, r.bottom));
}
const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT, {
  acceptNode(n) {
    const p = n.parentElement;
    if (!p || SKIP.has(p.tagName) || !n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
    return NodeFilter.FILTER_ACCEPT;
  }});
const range = document.createRange();
const shownCache = new Map();
for (let n = walker.nextNode(); n; n = walker.nextNode()) {
  const p = n.parentElement;
  let ok = shownCache.get(p);
  if (ok === undefined) { ok = shown(p); shownCache.set(p, ok); }
  if (!ok) continue;
  range.selectNodeContents(n);
  const rects = [...range.getClientRects()].filter(r => r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0
                                                       && r.top < vh && r.left < vw && visibleAt(p, r));
  if (!rects.length) continue;
  const unit = unitOf(p);
  const text = n.nodeValue.replace(/\s+/g, ' ').trim();
  rects.forEach((r, k) => addTo(unit, k === 0 ? text : '', r));
}
// Images with alt text that are not inside a marked element.
document.querySelectorAll('img[alt]').forEach(img => {
  const alt = (img.getAttribute('alt') || '').trim();
  if (!alt || !shown(img)) return;
  const r = img.getBoundingClientRect();
  if (r.width <= 0 || r.height <= 0 || !visibleAt(img, r)) return;
  addTo(unitOf(img), alt, r);
});
const out = [];
groups.forEach((g, unit) => {
  if (g.x1 <= g.x0 || g.y1 <= g.y0) return;
  const ref = markIndex.has(unit) ? markIndex.get(unit) : -1;
  out.push([g.parts.join(' ').slice(0, maxChars), g.x0, g.y0, g.x1, g.y1, roleOf(unit), ref >= 0 || clickable(unit), ref,
            ref >= 0 && scrollable(unit)]);
});
// Marked elements without a visible text run: icon buttons, image links, empty inputs, inputs with values.
marked.forEach((el, i) => {
  if (!el) return;
  const t = el.tagName;
  const isField = t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT';
  if (groups.has(el) && !isField) return;
  let text = '';
  if (isField) {
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (type !== 'password' && type !== 'hidden') text = (t === 'SELECT' ? (el.selectedOptions[0] || {}).text : el.value) || '';
    text = text || el.getAttribute('placeholder') || '';
  }
  text = text || el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('alt') || '';
  if (!text) { const img = el.querySelector && el.querySelector('img[alt]'); if (img) text = img.getAttribute('alt') || ''; }
  const r = el.getBoundingClientRect();
  const box = [Math.max(0, r.left), Math.max(0, r.top), Math.min(vw, r.right), Math.min(vh, r.bottom)];
  if (box[2] <= box[0] || box[3] <= box[1]) return;
  if (groups.has(el)) {    // a field with a text run inside (e.g. a SELECT): replace that row
    const k = out.findIndex(row => row[7] === i);
    if (k >= 0) out.splice(k, 1);
  }
  out.push([text.replace(/\s+/g, ' ').trim().slice(0, maxChars), ...box, roleOf(el), true, i, scrollable(el)]);
});
return [window.devicePixelRatio || 1, out];
"""


def web_text_boxes(*, rows: list, dpr: float, width: int, height: int) -> list:
    """The script's rows (CSS pixels) -> TextBox items in screenshot pixels, in reading order, content capped."""
    boxes = []
    for text, x0, y0, x1, y1, role, interactive, ref, scrollable in rows:
        box = clip_box(box=(x0 * dpr, y0 * dpr, x1 * dpr, y1 * dpr), width=width, height=height)
        if box is None:
            continue
        text = clean_text(text=text)
        ref = int(ref) if ref is not None and int(ref) >= 0 else None
        if not text and ref is None:
            continue
        boxes.append(TextBox(text=text, box=tuple(int(round(v)) for v in box), role=str(role),
                             interactive=bool(interactive), source="dom", ref=ref, scrollable=bool(scrollable)))
    ordered = reading_order(boxes=boxes)
    out, n_content = [], 0
    for b in ordered:
        if b.ref is None:
            n_content += 1
            if n_content > MAX_CONTENT_BOXES:
                continue
        out.append(b)
    return out
