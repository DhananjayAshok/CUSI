"""
Android text boxes from the accessibility tree's UI elements (AndroidWorld's UIElement, duck-typed so this module
needs no android_world import). Status bar, navigation bar and keyboard are dropped: they are not the app's content.
"""
from typing import Optional
from cusi.envs.text_boxes import TextBox, clean_text, clip_box, contains, reading_order

SYSTEM_PACKAGES = ("com.android.systemui",)
#: A clickable element with text up to this long is a button; a TextView up to this long is a label.
SHORT_TEXT = 40


def is_ime(*, package: Optional[str]) -> bool:
    p = (package or "").lower()
    return "inputmethod" in p or p.endswith(".ime") or "keyboard" in p or "swiftkey" in p


def element_text(*, ui) -> str:
    """The first non-empty of text, content description and hint, with the tooltip appended if it differs."""
    text = ""
    for value in (ui.text, ui.content_description, ui.hint_text):
        text = clean_text(text=value)
        if text:
            break
    tooltip = clean_text(text=getattr(ui, "tooltip", None))
    if tooltip and tooltip != text:
        text = f"{text} ({tooltip})" if text else tooltip
    return text


def element_role(*, ui, text: str) -> str:
    cls = (ui.class_name or "").split(".")[-1]
    if ui.is_editable or "EditText" in cls:
        return "input"
    if "Tab" in cls or "MenuItem" in cls:
        return "menu"
    if "Button" in cls or ui.is_checkable or cls in ("Switch", "CheckBox", "Chip"):
        return "button"
    if ui.is_clickable and len(text) <= SHORT_TEXT and "TextView" not in cls:
        return "button"
    if "TextView" in cls:
        return "label" if len(text) <= SHORT_TEXT else "body"
    return "button" if ui.is_clickable else "text"


def android_text_boxes(*, ui_elements: list, width: int, height: int) -> list:
    """One TextBox per visible app element with text, plus interactive or scrollable elements without text.
    ref is the element's index in ui_elements (what click/scroll actions take)."""
    boxes = []
    for i, ui in enumerate(ui_elements):
        package = ui.package_name or ""
        if not ui.is_visible or package in SYSTEM_PACKAGES or is_ime(package=package) or ui.bbox_pixels is None:
            continue
        b = ui.bbox_pixels
        box = clip_box(box=(b.x_min, b.y_min, b.x_max, b.y_max), width=width, height=height)
        if box is None:
            continue
        text = element_text(ui=ui)
        interactive = bool(ui.is_clickable or ui.is_long_clickable or ui.is_editable or ui.is_checkable)
        scrollable = bool(ui.is_scrollable)
        if not text and not interactive and not scrollable:
            continue
        boxes.append(TextBox(text=text, box=box, role=element_role(ui=ui, text=text), interactive=interactive,
                             source="a11y", ref=i, scrollable=scrollable))
    return reading_order(boxes=merge_containers(boxes=boxes))


def merge_containers(*, boxes: list) -> list:
    """A box whose text equals that of a box inside it is one element drawn twice (a clickable container and its
    label): keep the inner box, interactive if either is, with the interactive one's ref."""
    drop = set()
    for i, outer in enumerate(boxes):
        if not outer.text:
            continue
        for j, inner in enumerate(boxes):
            if i == j or j in drop or inner.text != outer.text or not contains(outer=outer.box, inner=inner.box):
                continue
            if outer.box == inner.box and j < i:
                continue             # identical boxes: keep the first
            if outer.interactive and not inner.interactive:
                inner.ref, inner.interactive = outer.ref, True
                if inner.role in ("label", "text", "body"):
                    inner.role = outer.role
            inner.scrollable = inner.scrollable or outer.scrollable
            drop.add(i)
            break
    return [b for k, b in enumerate(boxes) if k not in drop]
