#!/usr/bin/env python3
"""Create an HTML visualization of predictions.

Usage:
  python -m pipeline.prediction.visualize_predictions --dataset my_dataset --split val --model mymodel

This script expects prediction JSON files created by `generate_predictions.py`.
It looks for predictions at `results/{dataset}/predictions/{split}/{model}.json` by default.

Images may be local file paths, dataset-relative paths, bytes, or PIL images.
"""
import argparse
import base64
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.datasets import (
    REPO_ROOT,
    column_candidates,
    get_dataset_config,
    load_examples,
    normalize_dataset_name,
    resolve_column,
    resolve_repo_local_path,
)
from common.prediction_io import (
    find_eval_path,
    find_prediction_path,
    load_eval_labels,
)



def load_predictions(pred_path: Path):
    with pred_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def dataset_data_root(dataset_name: str) -> Path:
    cfg = get_dataset_config(dataset_name)
    local_dir = str(cfg.get("local_dir") or dataset_name.split("/")[-1].lower())
    return REPO_ROOT / "data" / local_dir


def _is_pil_image(x: Any) -> bool:
    return hasattr(x, "save") and hasattr(x, "mode") and hasattr(x, "size")


def _save_pil_to_assets(img: Any, assets_dir: Path, stem: str) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    out = assets_dir / f"{stem}.png"
    if not out.exists():
        try:
            img.save(out, format="PNG")
        except Exception:
            img.convert("RGB").save(out, format="PNG")
    return out


def _path_to_url_rel(p: Path, out_parent: Path) -> str:
    rel = os.path.relpath(p, start=out_parent)
    rel = rel.replace(os.sep, "/")
    return quote(rel)


def image_tag_for_example(example: Dict[str, Any], img_col: str, data_root: Path, out_path: Path, idx: int) -> str:
    img = example.get(img_col)
    out_parent = out_path.parent
    assets_dir = out_parent / "assets"

    if _is_pil_image(img):
        saved = _save_pil_to_assets(img, assets_dir, stem=f"img_{idx:06d}")
        rel_url = _path_to_url_rel(saved, out_parent)
        return f'<img src="{rel_url}" class="viz-img" loading="lazy" />'

    if isinstance(img, str):
        if img.startswith("http://") or img.startswith("https://"):
            return f'<img src="{html.escape(img, quote=True)}" class="viz-img" loading="lazy" />'

        normalized = resolve_repo_local_path(img)
        candidates = [
            data_root / "images" / img,
            data_root / img,
            Path(img),
        ]
        if normalized is not None:
            candidates.insert(0, normalized)
        for c in candidates:
            if c.exists():
                rel_url = _path_to_url_rel(c, out_parent)
                return f'<img src="{rel_url}" class="viz-img" loading="lazy" />'
        return f'<div class="missing">missing image: {html.escape(str(img))}</div>'

    if isinstance(img, (bytes, bytearray)):
        b64 = base64.b64encode(img).decode("utf-8")
        return f'<img src="data:image/png;base64,{b64}" class="viz-img" loading="lazy" />'

    if isinstance(img, dict):
        b = img.get("bytes")
        if b:
            b64 = base64.b64encode(b).decode("utf-8")
            return f'<img src="data:image/png;base64,{b64}" class="viz-img" loading="lazy" />'
        pth = img.get("path")
        if pth:
            p = resolve_repo_local_path(pth) or Path(pth)
            if not p.exists():
                p = data_root / pth
            if p.exists():
                rel_url = _path_to_url_rel(p, out_parent)
                return f'<img src="{rel_url}" class="viz-img" loading="lazy" />'

    return '<div class="missing">no image field</div>'


def _variant_order_key(v: str) -> Tuple[int, int, str]:
    vv = (v or "").strip().lower()
    if vv == "original":
        return (0, -1, vv)
    if vv == "reconstruction":
        return (1, -1, vv)
    m = re.match(r"^(?:chart_)?seed_(\d+)$", vv)
    if m:
        return (2, int(m.group(1)), vv)
    return (3, 10**9, vv)


def _variant_family_group(v: str) -> str:
    vv = (v or "").strip().lower()
    if vv == "original":
        return "original"
    if vv == "reconstruction":
        return "reconstruction"
    if vv.startswith("seed_"):
        return "seed"
    return vv or "missing"


def _configured_candidates(cfg: Dict[str, Any], primary_key: str, fallback_key: str, defaults: Sequence[str]) -> List[str]:
    return column_candidates(cfg, primary_key, fallback_key, defaults)


def _example_value(example: Dict[str, Any], candidates: Sequence[str]) -> str:
    for key in candidates:
        value = example.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _dataset_has_any_column(dataset_obj: Any, candidates: Sequence[str]) -> bool:
    columns = getattr(dataset_obj, "column_names", None)
    if columns is not None:
        return any(col in columns for col in candidates)
    sample = dataset_obj[0] if len(dataset_obj) else {}
    return any(col in sample for col in candidates)


def _family_id_candidates(cfg: Dict[str, Any]) -> List[str]:
    return _configured_candidates(
        cfg,
        "family_id_col",
        "family_id_cols",
        ("question_id", "family_id", "Family ID"),
    )


def _variant_family_keys(
    examples: Sequence[dict],
    variant_candidates: Sequence[str],
    family_id_candidates: Sequence[str],
) -> List[str]:
    """Return a stable question-family key for rows with reconstruction variants."""
    original_counts: Dict[str, int] = {}
    for ex in examples:
        variant = _example_value(ex, variant_candidates).strip().lower()
        if variant != "original":
            continue
        h = _example_value(ex, family_id_candidates).strip()
        if h:
            original_counts[h] = original_counts.get(h, 0) + 1

    duplicate_family_ids = {h for h, count in original_counts.items() if count > 1}
    counters: Dict[Tuple[str, str], int] = {}
    keys: List[str] = []
    for i, ex in enumerate(examples):
        h = _example_value(ex, family_id_candidates).strip()
        if not h:
            keys.append(f"__missing_family_{i}")
            continue
        if h not in duplicate_family_ids:
            keys.append(h)
            continue

        group = _variant_family_group(_example_value(ex, variant_candidates))
        counter_key = (h, group)
        slot = counters.get(counter_key, 0) % original_counts[h]
        counters[counter_key] = counters.get(counter_key, 0) + 1
        keys.append(f"{h}::slot::{slot}")
    return keys


def build_row_html(
    items: List[str],
    *,
    group_by_hash: bool = False,
    hashes: Optional[List[str]] = None,
    variants: Optional[List[str]] = None,
) -> str:
    if not group_by_hash:
        return "".join(items)

    if hashes is None or variants is None or len(hashes) != len(items) or len(variants) != len(items):
        return "".join(f'<div class="example-row">{item}</div>' for item in items)

    grouped: Dict[str, List[int]] = {}
    for i in range(len(items)):
        h = (hashes[i] or "").strip()
        if not h:
            h = f"__ungrouped_{i}"
        grouped.setdefault(h, []).append(i)

    row_chunks: List[str] = []
    for idxs in grouped.values():
        ordered = sorted(idxs, key=lambda j: (_variant_order_key(variants[j]), j))
        row_chunks.append(
            '<div class="example-row">' + "".join(items[j] for j in ordered) + "</div>"
        )
    return "".join(row_chunks)


def _modal_css_js_block() -> str:
    """Shared modal CSS/JS for zoom/pan on thumbnail images."""
    return r"""
/* ---------- modal (scroll + pan + zoom) ---------- */
body.modal-open { overflow: hidden; }

.modal {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.85);
  display: flex;
  align-items: stretch;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  position: relative;
  width: 100%;
  height: 100%;
}

#closeBtn {
  position: absolute;
  top: 12px;
  right: 12px;
  z-index: 1002;
  padding: 6px 10px;
}

.controls {
  position: absolute;
  top: 12px;
  left: 12px;
  z-index: 1002;
  display: flex;
  gap: 8px;
}

.controls button {
  padding: 6px 10px;
}

.hint {
  position: absolute;
  bottom: 12px;
  left: 12px;
  z-index: 1002;
  color: #eee;
  font-size: 12px;
  opacity: 0.85;
}

/* This is the important part: make the viewing area scrollable */
.modal-viewport {
  position: absolute;
  inset: 0;
  overflow: auto;
  cursor: grab;
  /* Keep some padding so the image isn't glued to the edge */
  padding: 32px;
  box-sizing: border-box;
}

.modal-viewport.dragging { cursor: grabbing; }

.modal-img {
  display: block;
  transform-origin: top left; /* makes scroll math sane */
  user-select: none;
  -webkit-user-drag: none;
  max-width: none; /* allow natural size; scaling handles sizing */
  max-height: none;
}
"""


def _modal_html_block() -> str:
    return r"""
<div id="imgModal" class="modal" style="display:none">
  <div class="modal-content">
    <button id="closeBtn" type="button">Close</button>
    <div class="controls">
      <button id="zoomOut" type="button">-</button>
      <button id="zoomIn" type="button">+</button>
      <button id="zoomReset" type="button">Reset</button>
    </div>
    <div class="hint">Scroll to move • Drag to pan • Ctrl/⌘ + wheel to zoom</div>

    <div id="modalViewport" class="modal-viewport">
      <img id="modalImg" src="" class="modal-img" />
    </div>
  </div>
</div>
"""


def _modal_js_block() -> str:
    return r"""
  // ---------- modal logic ----------
  const modal = document.getElementById('imgModal');
  const viewport = document.getElementById('modalViewport');
  const modalImg = document.getElementById('modalImg');
  let scale = 1;

  function setScale(newScale, anchorClientX, anchorClientY) {
    newScale = Math.max(0.05, Math.min(20, newScale));
    if (!viewport) return;

    const rect = viewport.getBoundingClientRect();
    const ax = (anchorClientX != null) ? anchorClientX : (rect.left + rect.width / 2);
    const ay = (anchorClientY != null) ? anchorClientY : (rect.top + rect.height / 2);

    const px = ax - rect.left + viewport.scrollLeft;
    const py = ay - rect.top + viewport.scrollTop;

    const prevScale = scale;
    scale = newScale;
    modalImg.style.transform = 'scale(' + scale + ')';

    const ratio = scale / prevScale;
    viewport.scrollLeft = px * ratio - (ax - rect.left);
    viewport.scrollTop  = py * ratio - (ay - rect.top);
  }

  function openModalFromImg(imgEl) {
    modal.style.display = 'flex';
    document.body.classList.add('modal-open');
    modalImg.src = imgEl.src;

    scale = 1;
    modalImg.style.transform = 'scale(1)';
    viewport.scrollTop = 0;
    viewport.scrollLeft = 0;
  }

  function closeModal() {
    modal.style.display = 'none';
    document.body.classList.remove('modal-open');
  }

  function bindThumbClicks() {
    document.querySelectorAll('.viz-img').forEach(function(img) {
      img.style.cursor = 'zoom-in';
      img.addEventListener('click', function() { openModalFromImg(img); });
    });
  }

  // Close handlers
  document.getElementById('closeBtn').addEventListener('click', closeModal);
  modal.addEventListener('click', function(e) { if (e.target === modal) closeModal(); });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && modal.style.display !== 'none') closeModal();
  });

  // Buttons
  document.getElementById('zoomIn').addEventListener('click', function() {
    const r = viewport.getBoundingClientRect();
    setScale(scale * 1.25, r.left + r.width/2, r.top + r.height/2);
  });
  document.getElementById('zoomOut').addEventListener('click', function() {
    const r = viewport.getBoundingClientRect();
    setScale(scale / 1.25, r.left + r.width/2, r.top + r.height/2);
  });
  document.getElementById('zoomReset').addEventListener('click', function() {
    const r = viewport.getBoundingClientRect();
    setScale(1, r.left + r.width/2, r.top + r.height/2);
    viewport.scrollTop = 0;
    viewport.scrollLeft = 0;
  });

  // Wheel: ctrl/meta + wheel => zoom. otherwise normal scroll
  viewport.addEventListener('wheel', function(e) {
    if (!(e.ctrlKey || e.metaKey)) return;
    e.preventDefault();
    const factor = (e.deltaY < 0) ? 1.1 : (1/1.1);
    setScale(scale * factor, e.clientX, e.clientY);
  }, { passive: false });

  // Drag-to-pan (scroll viewport)
  let dragging = false;
  let startX = 0, startY = 0, startScrollLeft = 0, startScrollTop = 0;

  viewport.addEventListener('mousedown', function(e) {
    if (e.button !== 0) return;
    dragging = true;
    viewport.classList.add('dragging');
    startX = e.clientX;
    startY = e.clientY;
    startScrollLeft = viewport.scrollLeft;
    startScrollTop = viewport.scrollTop;
    e.preventDefault();
  });

  window.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    viewport.scrollLeft = startScrollLeft - dx;
    viewport.scrollTop = startScrollTop - dy;
  });

  window.addEventListener('mouseup', function() {
    if (!dragging) return;
    dragging = false;
    viewport.classList.remove('dragging');
  });

  bindThumbClicks();
"""


def render_html_from_dataset(
    predictions,
    eval_labels: Optional[List[int]],
    dataset_name: str,
    split: str,
    model: str,
    out_path: Path,
    data_root: Path,
):
    cfg = get_dataset_config(dataset_name)
    dataset_name = normalize_dataset_name(dataset_name)
    ds = load_examples(dataset_name, split)

    q_col = resolve_column(cfg, ds, "question_col", "question_cols")
    img_col = resolve_column(cfg, ds, "image_col", "image_cols")
    answer_col = resolve_column(cfg, ds, "answer_col", "answer_cols")

    type_label = "Reasoning type"
    type_key_candidates = [
        "reasoning_type",
        "Reasoning Type",
        "reason_type",
        "type",
    ]

    variant_key_candidates = _configured_candidates(cfg, "variant_col", "variant_cols", ("variant", "Variant"))
    family_id_candidates = _family_id_candidates(cfg)
    variant_enabled = _dataset_has_any_column(ds, variant_key_candidates)

    def variant_sort_key(v: str) -> Tuple[int, str]:
        vv = (v or "").strip().lower()
        return (0 if vv == "original" else 1, vv)

    row_items: List[str] = []
    row_family_keys: List[str] = []
    row_variants: List[str] = []
    variant_values = set()

    n = min(len(predictions), len(ds))
    if eval_labels is not None and len(eval_labels) > 0:
        n = min(n, len(eval_labels))

    # initial global accuracy (JS will update based on TYPE ONLY)
    acc_num = 0
    acc_den = 0
    if eval_labels is not None and len(eval_labels) > 0:
        acc_den = n
        acc_num = sum(1 for i in range(n) if int(eval_labels[i]) == 1)

    family_keys = _variant_family_keys([ds[i] for i in range(n)], variant_key_candidates, family_id_candidates) if variant_enabled else []

    for i in range(n):
        ex = ds[i]
        p = predictions[i]

        if isinstance(p, dict) and "prediction" in p:
            pred_text = str(p["prediction"])
        else:
            pred_text = str(p)

        # Visible dataset-order index
        idx_str = f"#{i+1}"

        q_raw = str(ex.get(q_col) or "")
        q = html.escape(q_raw)
        gt = html.escape(str(ex.get(answer_col) or "")) if answer_col else ""

        corr_html = ""
        corr_attr = ""
        if eval_labels is not None and i < len(eval_labels):
            ok = int(eval_labels[i]) == 1
            corr_attr = "1" if ok else "0"
            badge = '<span class="badge ok">✅ Correct</span>' if ok else '<span class="badge bad">❌ Wrong</span>'
            corr_html = f"<p><b>Match:</b> {badge}</p>"

        raw_type = ""
        for k in type_key_candidates:
            if not k:
                continue
            v = ex.get(k)
            if v not in (None, ""):
                raw_type = str(v)
                break
        raw_variant = ""
        if variant_enabled:
            raw_variant = _example_value(ex, variant_key_candidates)
            if raw_variant:
                variant_values.add(raw_variant)
        family_key = ""
        if variant_enabled:
            family_key = family_keys[i] if i < len(family_keys) else _example_value(ex, family_id_candidates)

        type_show = html.escape(raw_type)
        type_extra = f"<p><b>{type_label}:</b> {type_show}</p>" if type_show else ""
        variant_attr = html.escape(raw_variant, quote=True)
        variant_show = html.escape(raw_variant)
        variant_extra = f"<p><b>Variant:</b> {variant_show}</p>" if raw_variant else ""

        img_tag = image_tag_for_example(ex, img_col, data_root, out_path, idx=i)

        # Store original (unescaped) question text for searching in a data-* attribute.
        q_search_attr = html.escape(q_raw, quote=True)

        row_items.append(
            f"""<div class="example"
  data-idx="{i}"
  data-variant="{variant_attr}"
  data-correct="{corr_attr}"
  data-qsearch="{q_search_attr}">
  <div class="left">{img_tag}</div>
  <div class="right">
    <div class="ex-index">Example {idx_str}</div>
    {corr_html}
    <p><b>Question:</b> <span class="qtext">{q}</span></p>
    <p><b>Answer:</b> {gt}</p>
    {type_extra}
    {variant_extra}
    <p><b>Model ({html.escape(model)}):</b> {html.escape(pred_text)}</p>
 </div>
 </div>""".strip()
        )
        row_family_keys.append(family_key)
        row_variants.append(raw_variant)

    css = """
.viz-img { max-width: 320px; max-height: 240px; }
.example { display:flex; gap:12px; padding:12px; border-bottom:1px solid #ddd; margin:0; }
.example-row { display:block; }
.left { flex:0 0 340px; }
.right { flex:1; }
.missing { color: #a00; }
body { font-family: Arial, sans-serif; margin: 0; padding: 12px; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:700; font-size:12px; }
.badge.ok { background:#e6f4ea; color:#137333; border:1px solid #c6e6cc; }
.badge.bad { background:#fce8e6; color:#a50e0e; border:1px solid #f5c2c0; }
#searchInput { padding:6px 10px; min-width:260px; }
#clearSearch { padding:6px 10px; }
mark.qhl { padding:0 2px; border-radius:3px; }
.ex-index { font-weight:700; color:#555; margin-bottom:6px; }
"""

    if variant_enabled:
        css += """
.example-row {
  display: flex;
  flex-wrap: nowrap;
  gap: 12px;
  margin: 0 0 12px 0;
  overflow-x: auto;
  padding-bottom: 4px;
}
.example {
  border: 1px solid #ddd;
  border-radius: 10px;
  margin: 0;
  flex: 0 0 520px;
}
"""

    variant_options_html = "".join(
        [f'<option value="{html.escape(v, quote=True)}">{html.escape(v)}</option>' for v in sorted(variant_values, key=variant_sort_key)]
    )

    rows_html = build_row_html(
        row_items,
        group_by_hash=variant_enabled,
        hashes=row_family_keys,
        variants=row_variants,
    )

    if acc_den > 0:
        acc_pct = 100.0 * acc_num / acc_den
        acc_html = (
            f'<span id="accBadge" class="badge ok">'
            f'Accuracy: <span id="accNum">{acc_num}</span>/<span id="accDen">{acc_den}</span> '
            f'(<span id="accPct">{acc_pct:.2f}</span>%)'
            f"</span>"
        )
    else:
        acc_html = (
            '<span id="accBadge" class="badge bad">'
            'Accuracy: <span id="accNum">0</span>/<span id="accDen">0</span> '
            '(<span id="accPct">0.00</span>%)'
            "</span>"
        )

    modal_css = _modal_css_js_block()

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Predictions: {html.escape(dataset_name)} {html.escape(split)} {html.escape(model)}</title>
<style>{css}
{modal_css}
</style>
</head>
<body>
<h1>Predictions: {html.escape(dataset_name)} / {html.escape(split)} / {html.escape(model)}</h1>

<div style="margin:12px 0; display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
  <div id="exampleCount"><b>Examples:</b> {len(row_items)}</div>
  <div>{acc_html}</div>

  {'<div id="variantBar"><label for="variantFilter"><b>Filter by variant:</b></label><select id="variantFilter"><option value="">All</option>' + variant_options_html + '</select></div>' if variant_enabled else ''}
  <div id="correctBar">
    <label for="correctFilter"><b>Filter by correctness:</b></label>
    <select id="correctFilter">
      <option value="">All</option>
      <option value="1">Correct</option>
      <option value="0">Wrong</option>
    </select>
  </div>

  <div id="searchBar" style="display:flex; align-items:center; gap:8px;">
    <label for="searchInput"><b>Search:</b></label>
    <input id="searchInput" type="text" placeholder="e.g., revenue growth" />
    <button id="clearSearch" type="button">Clear</button>
  </div>
</div>

{rows_html}

{_modal_html_block()}

<script>
document.addEventListener("DOMContentLoaded", function() {{
{_modal_js_block()}

  // ---------------- Filters ----------------
  const variantSel = document.getElementById('variantFilter');
  const corrSel = document.getElementById('correctFilter');
  const searchInput = document.getElementById('searchInput');
  const clearBtn = document.getElementById('clearSearch');

  function updateCountShown() {{
    const totalShown = Array.from(document.querySelectorAll('.example')).filter(e => {{
      if (e.style.display === 'none') return false;
      const row = e.closest('.example-row');
      if (row && row.style.display === 'none') return false;
      return true;
    }}).length;
    const el = document.getElementById('exampleCount');
    if (el) el.innerHTML = '<b>Examples:</b> ' + totalShown;
  }}

  function updateRowVisibility() {{
    document.querySelectorAll('.example-row').forEach(function(row) {{
      const hasVisible = Array.from(row.querySelectorAll('.example')).some(ex => ex.style.display !== 'none');
      row.style.display = hasVisible ? '' : 'none';
    }});
  }}

  // Accuracy badge is computed from the variant filter only.
  function updateAccuracyFromPrimaryFilters() {{
    const variantVal = variantSel ? variantSel.value : '';
    const all = Array.from(document.querySelectorAll('.example'));

    const selected = all.filter(ex => {{
      const v = ex.getAttribute('data-variant') || '';
      return !variantVal || v === variantVal;
    }});

    let den = 0;
    let num = 0;
    selected.forEach(ex => {{
      const c = ex.getAttribute('data-correct');
      if (c === '0' || c === '1') {{
        den += 1;
        if (c === '1') num += 1;
      }}
    }});

    const badge = document.getElementById('accBadge');
    const numEl = document.getElementById('accNum');
    const denEl = document.getElementById('accDen');
    const pctEl = document.getElementById('accPct');
    if (!badge || !numEl || !denEl || !pctEl) return;

    if (den === 0) {{
      badge.classList.remove('ok');
      badge.classList.add('bad');
      numEl.textContent = '0';
      denEl.textContent = '0';
      pctEl.textContent = '0.00';
      return;
    }}

    const pct = 100.0 * num / den;
    badge.classList.remove('bad');
    badge.classList.add('ok');
    numEl.textContent = String(num);
    denEl.textContent = String(den);
    pctEl.textContent = pct.toFixed(2);
  }}

  function parseKeywords(s) {{
    return (s || '')
      .toLowerCase()
      .split(/\\s+/)
      .map(x => x.trim())
      .filter(x => x.length > 0);
  }}

  function escapeHtml(s) {{
    return (s || '')
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }}

  function highlightQuestion(exEl, keywords) {{
    const qSpan = exEl.querySelector('.qtext');
    if (!qSpan) return;

    const original = exEl.getAttribute('data-qsearch') || '';
    let safe = escapeHtml(original);

    if (!keywords || keywords.length === 0) {{
      qSpan.innerHTML = safe;
      return;
    }}

    keywords.forEach(kw => {{
      if (!kw) return;
      try {{
        const re = new RegExp(kw.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&'), 'gi');
        safe = safe.replace(re, (m) => '<mark class="qhl">' + m + '</mark>');
      }} catch (e) {{}}
    }});

    qSpan.innerHTML = safe;
  }}

  function applyFilters() {{
    const variantVal = variantSel ? variantSel.value : '';
    const corrVal = corrSel ? corrSel.value : '';
    const keywords = parseKeywords(searchInput ? searchInput.value : '');

    document.querySelectorAll('.example').forEach(function(ex) {{
      const v = ex.getAttribute('data-variant') || '';
      const c = ex.getAttribute('data-correct') || '';
      const q = (ex.getAttribute('data-qsearch') || '').toLowerCase();

      const okVariant = (!variantVal || v === variantVal);
      const okCorr = (!corrVal || c === corrVal);

      let okSearch = true;
      for (let i = 0; i < keywords.length; i++) {{
        if (q.indexOf(keywords[i]) === -1) {{
          okSearch = false;
          break;
        }}
      }}

      ex.style.display = (okVariant && okCorr && okSearch) ? 'flex' : 'none';

      if (ex.style.display !== 'none') {{
        highlightQuestion(ex, keywords);
      }} else {{
        highlightQuestion(ex, []);
      }}
    }});

    updateRowVisibility();
    updateCountShown();
    updateAccuracyFromPrimaryFilters();
  }}

  if (variantSel) variantSel.addEventListener('change', applyFilters);
  if (corrSel) corrSel.addEventListener('change', applyFilters);

  if (searchInput) {{
    searchInput.addEventListener('input', applyFilters);
    searchInput.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape') {{
        searchInput.value = '';
        applyFilters();
      }}
    }});
  }}

  if (clearBtn) {{
    clearBtn.addEventListener('click', function() {{
      if (searchInput) searchInput.value = '';
      applyFilters();
      if (searchInput) searchInput.focus();
    }});
  }}

  applyFilters();
}});
</script>

</body>
</html>"""

    ensure_dir(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(html_doc)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--model", required=True, help="Prediction model name")
    p.add_argument("--predictions", help="Path to predictions JSON (optional)")
    p.add_argument("--eval", help="Path to eval file (optional). Default: {model_name}_eval.json next to predictions")
    p.add_argument("--limit", type=int, default=None, help="Visualize a limited prediction file from a larger dataset run")
    p.add_argument("--out", help="Output HTML file (optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    split = args.split

    data_root = dataset_data_root(dataset)
    model = args.model
    pred_path = Path(args.predictions) if args.predictions else find_prediction_path(dataset, split, model, limit=args.limit)

    out_file = (
        Path(args.out)
        if args.out
        else pred_path.with_suffix(".html")
    )

    if not pred_path.exists():
        print(f"[ERROR] predictions file not found: {pred_path}")
        raise SystemExit(2)

    preds = load_predictions(pred_path)
    print(f"[OK] loaded predictions: {pred_path} ({len(preds)} entries)")

    eval_path = Path(args.eval) if args.eval else find_eval_path(pred_path, model)
    eval_labels: Optional[List[int]] = None
    if eval_path.exists():
        eval_labels = load_eval_labels(eval_path)
        print(f"[OK] loaded eval labels: {eval_path} ({len(eval_labels)} entries)")
    else:
        print(f"[WARN] eval file not found (skipping): {eval_path}")

    render_html_from_dataset(preds, eval_labels, dataset, split, model, out_file, data_root)
    print(f"Wrote visualization to: {out_file}")
