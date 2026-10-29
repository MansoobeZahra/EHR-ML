"""Add clickable links to the LibreOffice-rendered PDF (LibreOffice 7.3 drops them on export).

The docx already contains every link; this reproduces them in the PDF:
  * each in-text citation [n]  -> reference entry n
  * each "Fig. N" / "Table X"   -> that figure or table caption
  * each reference entry        -> its DOI or proceedings page (REF_URL in build_paper_v3.py)
Usage: python add_pdf_links.py <in.pdf> <out.pdf>
"""
import re
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
src = open(HERE / 'build_paper_v3.py', encoding='utf-8').read()
REF_URL = eval(src[src.index('REF_URL = {') + len('REF_URL = '):src.index('assert sorted(REF_URL)')])
assert sorted(REF_URL) == list(range(1, 23))

doc = pymupdf.open(sys.argv[1])
pages_words = [p.get_text('words', sort=False) for p in doc]   # (x0, y0, x1, y1, text, block, line, wordno)

# ---- locate the reference list and each entry label
ref_page = ref_y = None
for pi, words in enumerate(pages_words):
    for w in words:
        if w[4] == 'REFERENCES':
            ref_page, ref_y, ref_x = pi, w[1], w[0]
assert ref_page is not None
mid = doc[0].rect.width / 2


def in_refs(pi, w):
    """True if the word lies in the reference list (after the REFERENCES heading)."""
    if pi < ref_page:
        return False
    if pi > ref_page:
        return True
    col_of_heading = 0 if ref_x < mid else 1
    col = 0 if w[0] < mid else 1
    return (col > col_of_heading) or (col == col_of_heading and w[1] > ref_y)


labels = {}
for pi, words in enumerate(pages_words):
    for w in words:
        m = re.fullmatch(r'\[(\d+)\]', w[4])
        if m and in_refs(pi, w) and w[7] == 0:          # first word of a line = entry label
            labels.setdefault(int(m.group(1)), (pi, w))
assert sorted(labels) == list(range(1, 23)), sorted(labels)

# ---- caption anchors
caps = {}
for pi, words in enumerate(pages_words):
    for i, w in enumerate(words[:-1]):
        nxt = words[i + 1][4]
        if w[4] == 'Fig.' and re.fullmatch(r'\d\.', nxt) and w[7] == 0:
            caps.setdefault(f'Fig. {nxt[0]}', (pi, w))
        if w[4] == 'TABLE' and re.fullmatch(r'I{1,3}\.', nxt):
            caps.setdefault(f'Table {nxt[:-1]}', (pi, w))
assert sorted(caps) == ['Fig. 1', 'Fig. 2', 'Fig. 3', 'Table I'], caps


def goto(page, src_rect, dest):
    dpi, dw = dest
    page.insert_link({'kind': pymupdf.LINK_GOTO, 'from': pymupdf.Rect(src_rect[:4]),
                      'page': dpi, 'to': pymupdf.Point(dw[0], max(0, dw[1] - 4)), 'zoom': 0})


n_cite = n_fig = n_url = 0
for pi, words in enumerate(pages_words):
    page = doc[pi]
    for i, w in enumerate(words):
        if in_refs(pi, w):
            continue
        for m in re.finditer(r'\[(\d+)\]', w[4]):          # "[3]", "[7],", "[15],"
            n = int(m.group(1))
            if n in labels:
                goto(page, w, labels[n]); n_cite += 1
        if i + 1 < len(words):
            nxt = re.sub(r'[^\w]', '', words[i + 1][4])
            key = None
            if w[4] == 'Fig.' and nxt.isdigit():
                key = f'Fig. {nxt}'
            elif w[4] == 'Table' and nxt in ('I', 'II', 'III'):
                key = f'Table {nxt}'
            if key and key in caps and (pi, w) != caps[key]:
                cap_pi, cap_w = caps[key]
                if not (cap_pi == pi and abs(cap_w[1] - w[1]) < 1 and abs(cap_w[0] - w[0]) < 1):
                    r = pymupdf.Rect(w[:4]) | pymupdf.Rect(words[i + 1][:4])
                    goto(page, r, caps[key]); n_fig += 1

# ---- external link over each reference entry
order = sorted(labels)
for n in order:
    pi, lw = labels[n]
    words = pages_words[pi]
    col = 0 if lw[0] < mid else 1
    nxt = labels.get(n + 1)
    rects = []
    for w in words:
        wcol = 0 if w[0] < mid else 1
        if wcol != col or w[1] < lw[1] - 1:
            continue
        if nxt and nxt[0] == pi and (0 if nxt[1][0] < mid else 1) == col and w[1] >= nxt[1][1] - 1:
            continue
        if not in_refs(pi, w):
            continue
        rects.append(pymupdf.Rect(w[:4]))
    # entry continues in the next column/page if the next label is elsewhere: link the first part only
    lines = {}
    for r in rects:
        key = round(r.y0)
        lines[key] = lines[key] | r if key in lines else r
    for r in lines.values():
        doc[pi].insert_link({'kind': pymupdf.LINK_URI, 'from': r, 'uri': REF_URL[n]})
        n_url += 1

doc.save(sys.argv[2], garbage=3, deflate=True)
print(f'links added: {n_cite} citations, {n_fig} figure/table mentions, {n_url} reference-line links')
