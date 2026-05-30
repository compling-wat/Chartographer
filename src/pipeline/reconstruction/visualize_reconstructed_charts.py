#!/usr/bin/env python3
"""
Visualize original images against rendered reconstruction variants.

Pass variant names via --charts, e.g.:
  python -m pipeline.reconstruction.visualize_reconstructed_charts --split val \
    --charts reconstruction seed_0

The script resolves them as:
  <results_root>/<dataset>_<split>/images/<variant>

Shows (1 + N) images per example:
  - Original (from --original_dir)
  - One cell per generated variant
"""

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.datasets import get_dataset_config, load_examples, normalize_dataset_name

SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
HF_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"
HF_DATASETS_CACHE_DIR = HF_CACHE_DIR / "datasets"


def _configure_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))
    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _path_to_url_rel(p: Path, out_parent: Path) -> str:
    rel = os.path.relpath(p, start=out_parent)
    rel = rel.replace(os.sep, "/")
    return quote(rel)


def _img_tag(path: Optional[Path], out_parent: Path) -> str:
    if path is None:
        return '<div class="missing">missing</div>'
    if not path.exists():
        return f'<div class="missing">missing: {html.escape(path.name)}</div>'
    rel_url = _path_to_url_rel(path, out_parent)
    return f'<img src="{rel_url}" class="viz-img" loading="lazy" />'


def _resolve_image_by_stem(dir_path: Path, name: str) -> Optional[Path]:
    """
    Resolve image path by exact name first; if missing, fallback by basename stem
    to handle extension mismatches (e.g., generated .png vs original .jpg).
    """
    exact = dir_path / name
    if exact.exists():
        return exact

    stem = Path(name).stem
    if not dir_path.exists():
        return None

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if p.stem == stem:
            return p
    return None


def _modal_css_block() -> str:
    return r"""
body.modal-open { overflow: hidden; }
.modal {
  position: fixed; inset: 0; background: rgba(0,0,0,0.85);
  display: flex; align-items: stretch; justify-content: center; z-index: 1000;
}
.modal-content { position: relative; width: 100%; height: 100%; }
#closeBtn { position: absolute; top: 12px; right: 12px; z-index: 1002; padding: 6px 10px; }
.controls { position: absolute; top: 12px; left: 12px; z-index: 1002; display: flex; gap: 8px; }
.controls button { padding: 6px 10px; }
.hint { position: absolute; bottom: 12px; left: 12px; z-index: 1002; color: #eee; font-size: 12px; opacity: 0.85; }
.modal-viewport { position: absolute; inset: 0; overflow: auto; cursor: grab; padding: 32px; box-sizing: border-box; }
.modal-viewport.dragging { cursor: grabbing; }
.modal-img {
  display: block; transform-origin: top left; user-select: none; -webkit-user-drag: none;
  max-width: none; max-height: none;
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

  document.getElementById('closeBtn').addEventListener('click', closeModal);
  modal.addEventListener('click', function(e) { if (e.target === modal) closeModal(); });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && modal.style.display !== 'none') closeModal();
  });

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

  viewport.addEventListener('wheel', function(e) {
    if (!(e.ctrlKey || e.metaKey)) return;
    e.preventDefault();
    const factor = (e.deltaY < 0) ? 1.1 : (1/1.1);
    setScale(scale * factor, e.clientX, e.clientY);
  }, { passive: false });

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


def list_images(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    names: List[str] = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            names.append(p.name)
    names.sort()
    return names


def resolve_split_dir(results_root: Path, dataset: str, split: str) -> Tuple[Path, str]:
    normalized = normalize_dataset_name(dataset)
    candidates = [results_root / f"{normalized}_{split}"]
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidate.name
    return candidates[0], candidates[0].name


def pick_index_names(original_dir: Path, gen_dirs: List[Path], prefer: str) -> List[str]:
    orig = set(list_images(original_dir))
    gens = set()
    for d in gen_dirs:
        gens |= set(list_images(d))

    if prefer == "original":
        base = sorted(orig)
        return base if base else sorted(orig | gens)

    if prefer == "generated":
        base = sorted(gens)
        return base if base else sorted(orig | gens)

    return sorted(orig | gens)


def _norm_name(x: str) -> str:
    return os.path.basename(x).casefold()


def _norm_stem(x: str) -> str:
    return Path(os.path.basename(x)).stem.casefold()


def _qa_lookup(idx: Dict[str, List[Tuple[str, str]]], name: str) -> List[Tuple[str, str]]:
    key = _norm_name(name)
    if key in idx:
        return idx[key]
    stem = _norm_stem(name)
    for k, v in idx.items():
        if _norm_stem(k) == stem:
            return v
    return []


def _image_name_from_example(ex: Dict[str, object], cfg: Dict[str, object]) -> Optional[str]:
    candidate_cols = [
        cfg.get("figure_path_col"),
        cfg.get("original_figure_path_col"),
        cfg.get("image_col"),
        "figure_path",
        "original_figure_path",
        "image",
        "image_path",
        "path",
        "filename",
        "file",
        "img",
    ]

    for col in candidate_cols:
        if not isinstance(col, str) or not col:
            continue
        img = ex.get(col)
        name: Optional[str] = None
        if isinstance(img, str):
            name = os.path.basename(img)
        elif isinstance(img, dict):
            p = img.get("path") or img.get("filename")
            if isinstance(p, str) and p:
                name = os.path.basename(p)
        else:
            fn = getattr(img, "filename", None)
            if isinstance(fn, str) and fn:
                name = os.path.basename(fn)
        if name:
            return name
    return None


def _qa_pairs_from_example(ex: Dict[str, object], cfg: Dict[str, object]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    
    def _valid_q(q: object) -> bool:
        if not isinstance(q, str):
            return False
        q_text = q.strip()
        if not q_text:
            return False
        # Filter out numeric question ids that sometimes appear in auxiliary question fields.
        if q_text.isdigit():
            return False
        return True

    question_col = cfg.get("question_col")
    answer_col = cfg.get("answer_col")
    if isinstance(question_col, str) and isinstance(answer_col, str):
        q = ex.get(question_col)
        a = ex.get(answer_col)
        if _valid_q(q) and a is not None:
            pairs.append((str(q), str(a)))

    q_cols = cfg.get("descriptive_question_cols")
    a_cols = cfg.get("descriptive_answer_cols")
    if isinstance(q_cols, list) and isinstance(a_cols, list):
        for q_col, a_col in zip(q_cols, a_cols):
            q = ex.get(q_col)
            a = ex.get(a_col)
            if not _valid_q(q) or a is None:
                continue
            q_text = str(q).strip()
            a_text = str(a).strip()
            if q_text and a_text:
                pairs.append((q_text, a_text))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for q, a in pairs:
        item = (q.strip(), a.strip())
        if item in seen or not item[0] or not item[1]:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def build_qa_index(dataset: str, split: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build mapping: normalized_filename -> [(question, answer), ...] from the source dataset.
    """
    _configure_hf_cache()
    cfg = get_dataset_config(dataset)
    ds = load_examples(dataset, split)

    idx: Dict[str, List[Tuple[str, str]]] = {}
    for ex in ds:
        name = _image_name_from_example(ex, cfg)
        if not name:
            continue

        pairs = _qa_pairs_from_example(ex, cfg)
        if not pairs:
            continue
        idx.setdefault(_norm_name(name), []).extend(pairs)

    return idx


def build_qa_index_from_json(path: Path, dataset_cfg: Optional[Dict[str, object]] = None) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build mapping normalized image filename -> (question, answer) from local QA json list.
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected list in QA JSON: {path}")

    idx: Dict[str, List[Tuple[str, str]]] = {}
    for ex in obj:
        if not isinstance(ex, dict):
            continue
        cfg = dataset_cfg if isinstance(dataset_cfg, dict) else {}
        name = _image_name_from_example(ex, cfg)
        if not name:
            continue
        q_col = "question"
        a_col = "answer"
        if isinstance(dataset_cfg, dict):
            qc = dataset_cfg.get("question_col")
            ac = dataset_cfg.get("answer_col")
            if isinstance(qc, str) and qc:
                q_col = qc
            if isinstance(ac, str) and ac:
                a_col = ac

        q_val = ex.get(q_col)
        a_val = ex.get(a_col)

        q = "" if q_val is None else str(q_val)
        a = "" if a_val is None else str(a_val)
        idx.setdefault(_norm_name(name), []).append((q, a))
    return idx


def qa_file_for_chart_label(split_dir: Path, label: str) -> Optional[Path]:
    """
    Infer QA json path from rendered variant label:
      - reconstruction -> qa_reconstruction.json
      - seed_N         -> qa_seed_N.json
      - <label>        -> qa_<label>.json (if exists)
    """
    if label == "reconstruction":
        return split_dir / "qa_reconstruction.json"
    m = re.fullmatch(r"seed_(\d+)", label)
    if m:
        seed = int(m.group(1))
        return split_dir / f"qa_seed_{seed}.json"
    p = split_dir / f"qa_{label}.json"
    return p if p.exists() else None


def _qa_html_block(items: List[Tuple[str, str]], missing_msg: str) -> str:
    if items:
        blocks: List[str] = []
        for q, a in items:
            blocks.append(
                f'<div class="qa-item">'
                f'<div class="qa-line"><span class="qa-k">Q:</span> <span class="qa-v">{html.escape(q)}</span></div>'
                f'<div class="qa-line"><span class="qa-k">A:</span> <span class="qa-v">{html.escape(a)}</span></div>'
                f"</div>"
            )
        return f'<div class="qa">{"".join(blocks)}</div>'
    return (
        f'<div class="qa missing-qa">'
        f'<div class="qa-line"><span class="qa-k">Q/A:</span> '
        f'<span class="qa-v">{html.escape(missing_msg)}</span></div>'
        f"</div>"
    )


def render_html(
    split: str,
    out_path: Path,
    original_dir: Path,
    gen_dirs: List[Path],
    gen_labels: List[str],
    prefer: str,
    limit: Optional[int],
    qa_index: Dict[str, List[Tuple[str, str]]],
    qa_by_label: Dict[str, Dict[str, List[Tuple[str, str]]]],
):
    out_parent = out_path.parent
    ensure_dir(out_path)

    names = pick_index_names(original_dir, gen_dirs, prefer=prefer)
    if limit is not None and limit > 0:
        names = names[:limit]

    cols = 1 + len(gen_dirs)

    rows: List[str] = []
    for i, name in enumerate(names, start=1):
        orig_p = _resolve_image_by_stem(original_dir, name)

        orig_items = _qa_lookup(qa_index, name)
        orig_qa_html = _qa_html_block(orig_items, f"not found in original dataset for filename: {name}")

        gen_cells: List[str] = []
        for lab, d in zip(gen_labels, gen_dirs):
            p = _resolve_image_by_stem(d, name)
            lab_items = _qa_lookup(qa_by_label.get(lab, qa_index), name)
            lab_qa_html = _qa_html_block(lab_items, f"not found in QA index for label '{lab}', filename: {name}")
            gen_cells.append(
                f"""
    <div class="cell">
      <div class="label">{html.escape(lab)}</div>
      {_img_tag(p, out_parent)}
      {lab_qa_html}
    </div>
""".rstrip()
            )

        rows.append(
            f"""
<div class="row" data-name="{html.escape(name, quote=True)}">
  <div class="meta">
    <div class="idx">#{i}</div>
    <div class="fname">{html.escape(name)}</div>
  </div>

  <div class="grid" style="grid-template-columns: repeat({cols}, minmax(240px, 1fr));">
    <div class="cell">
      <div class="label">Original</div>
      {_img_tag(orig_p, out_parent)}
      {orig_qa_html}
    </div>

{os.linesep.join(gen_cells)}
  </div>
</div>
""".strip()
        )

    css = f"""
body {{
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 12px;
}}

h1 {{ margin: 0 0 10px 0; }}
.sub {{ color:#444; margin: 0 0 12px 0; }}

.toolbar {{
  display:flex;
  align-items:center;
  gap:12px;
  flex-wrap:wrap;
  margin: 10px 0 14px 0;
}}

#searchInput {{
  padding: 6px 10px;
  min-width: 320px;
}}

button {{
  padding: 6px 10px;
}}

.count {{
  font-weight: 700;
  color:#333;
}}

.row {{
  border-top: 1px solid #ddd;
  padding: 10px 0 14px 0;
}}

.meta {{
  display:flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}}

.idx {{
  font-weight: 700;
  color:#666;
  min-width: 52px;
}}

.fname {{
  font-weight: 700;
  color:#222;
  word-break: break-all;
}}

.qa {{
  border: 1px solid #e5e5e5;
  background: #fff;
  border-radius: 12px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}}
.qa-item + .qa-item {{
  border-top: 1px solid #eee;
  margin-top: 8px;
  padding-top: 8px;
}}
.qa-line {{
  display:flex;
  gap: 8px;
  margin: 2px 0;
}}
.qa-k {{
  font-weight: 700;
  color:#333;
  min-width: 22px;
}}
.qa-v {{
  color:#222;
  white-space: pre-wrap;
}}
.missing-qa .qa-v {{
  color:#a00;
}}

.grid {{
  display:grid;
  gap: 12px;
  align-items: start;
}}

.cell {{
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 8px 10px;
  background: #fafafa;
}}

.label {{
  font-weight: 700;
  font-size: 12px;
  color:#444;
  margin: 0 0 6px 0;
}}

.viz-img {{
  width: 100%;
  height: auto;
  max-height: 340px;
  object-fit: contain;
  background: white;
  border-radius: 8px;
}}

.missing {{
  color:#a00;
  background: #fff;
  border: 1px dashed #e0b4b4;
  border-radius: 8px;
  padding: 18px 10px;
  font-size: 12px;
}}

@media (max-width: 1400px) {{
  .grid {{ grid-template-columns: repeat(3, minmax(240px, 1fr)) !important; }}
}}
@media (max-width: 900px) {{
  .grid {{ grid-template-columns: repeat(2, minmax(240px, 1fr)) !important; }}
}}
@media (max-width: 700px) {{
  .grid {{ grid-template-columns: 1fr !important; }}
}}

{_modal_css_block()}
"""

    rows_html = "\n".join(rows)

    gen_lines = "\n".join(
        f"  <b>{html.escape(lab)}:</b> {html.escape(str(d))}<br/>"
        for lab, d in zip(gen_labels, gen_dirs)
    )

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Chartographer compare ({html.escape(split)})</title>
<style>{css}</style>
</head>
<body>
<h1>Chartographer image compare</h1>
<p class="sub">
  <b>Split:</b> {html.escape(split)} •
  <b>Original:</b> {html.escape(str(original_dir))}<br/>
{gen_lines}
</p>

<div class="toolbar">
  <span class="count" id="countShown">Showing: {len(names)}</span>

  <label for="searchInput"><b>Search filename:</b></label>
  <input id="searchInput" type="text" placeholder="e.g., 000123.png or bar" />
  <button id="clearSearch" type="button">Clear</button>
</div>

{rows_html}

{_modal_html_block()}

<script>
document.addEventListener("DOMContentLoaded", function() {{

{_modal_js_block()}

  const searchInput = document.getElementById('searchInput');
  const clearBtn = document.getElementById('clearSearch');
  const countEl = document.getElementById('countShown');

  function updateCountShown() {{
    const shown = Array.from(document.querySelectorAll('.row')).filter(r => r.style.display !== 'none').length;
    if (countEl) countEl.textContent = 'Showing: ' + shown;
  }}

  function applyFilters() {{
    const q = (searchInput && searchInput.value ? searchInput.value : '').toLowerCase().trim();
    document.querySelectorAll('.row').forEach(function(row) {{
      const name = (row.getAttribute('data-name') || '').toLowerCase();
      const matchesName = !q || name.indexOf(q) !== -1;
      row.style.display = matchesName ? 'block' : 'none';
    }});
    updateCountShown();
  }}

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
</html>
"""

    out_path.write_text(html_doc, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset key/HF alias used in results and QA lookup.")
    p.add_argument("--split", required=True, help="e.g., train/val/test")
    p.add_argument(
        "--results_root",
        default=str(REPO_ROOT / "results" / "chartographer"),
        help="Root that contains <dataset>_{split} results directories.",
    )
    p.add_argument(
        "--original_dir",
        default=None,
        help="Directory with original dataset images. Defaults to ../data/<dataset>/images.",
    )
    p.add_argument(
        "--charts",
        nargs="+",
        required=True,
        help="List of rendered variant names under images/ (for example: reconstruction seed_0).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output HTML path (default: <split_dir>/visualize_chartographer.html)",
    )
    p.add_argument(
        "--prefer",
        choices=["original", "generated", "union"],
        default="generated",
        help="Which filenames to index by (default: generated)",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results_root = Path(args.results_root)
    dataset_cfg = get_dataset_config(args.dataset)
    split_dir, used_name = resolve_split_dir(results_root, str(dataset_cfg["local_dir"]), args.split)

    original_dir = (
        Path(args.original_dir)
        if args.original_dir
        else REPO_ROOT / "data" / str(dataset_cfg["local_dir"]) / "images"
    )
    if args.original_dir is None:
        split_original_dir = original_dir / args.split
        if split_original_dir.exists():
            original_dir = split_original_dir

    # Resolve rendered variant directories under images/.
    gen_labels = list(args.charts)
    gen_dirs = [split_dir / "images" / name for name in gen_labels]

    out_path = Path(args.out) if args.out else (split_dir / "visualize_chartographer.html")

    if not split_dir.exists():
        print(f"[WARN] split dir not found: {split_dir} (using name '{used_name}')")
    for d in gen_dirs:
        if not d.exists():
            print(f"[WARN] missing generated dir: {d}")
    if not original_dir.exists():
        print(f"[WARN] missing original_dir: {original_dir}")

    # Build filename -> (question, answer) index
    qa_index = build_qa_index(args.dataset, args.split)
    print(f"[OK] built QA index for {args.dataset} split={args.split}: {len(qa_index)} items")

    qa_by_label: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    for lab in gen_labels:
        qa_path = qa_file_for_chart_label(split_dir, lab)
        if qa_path is None:
            qa_by_label[lab] = {}
            print(f"[WARN] no QA filename rule for label '{lab}', leaving QA empty for that variation")
            continue
        if not qa_path.exists():
            qa_by_label[lab] = {}
            print(f"[WARN] missing QA file for label '{lab}': {qa_path}")
            continue
        try:
            qa_by_label[lab] = build_qa_index_from_json(qa_path, dataset_cfg)
            print(f"[OK] loaded QA for '{lab}': {qa_path} ({len(qa_by_label[lab])} items)")
        except Exception as e:
            qa_by_label[lab] = {}
            print(f"[WARN] failed loading QA for '{lab}' from {qa_path}: {e}")

    render_html(
        split=args.split,
        out_path=out_path,
        original_dir=original_dir,
        gen_dirs=gen_dirs,
        gen_labels=gen_labels,
        prefer=args.prefer,
        limit=args.limit,
        qa_index=qa_index,
        qa_by_label=qa_by_label,
    )
    print(f"[OK] wrote: {out_path}")
