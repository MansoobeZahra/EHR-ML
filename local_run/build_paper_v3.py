"""Build EHR-ML_revision_v3.docx from the v2 docx (formatting templates) and the
new results. Table values are read from the result CSVs; text numbers were checked
against results/SUMMARY.md and results/SUMMARY_ext_tables.md.

Usage: python build_paper_v3.py <unzipped_v2_dir> <out_docx>
"""
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

SRC = Path(sys.argv[1])
OUT_DOCX = Path(sys.argv[2]).resolve()
HERE = Path(__file__).resolve().parent
RES = HERE.parent
EXT = HERE / 'out_ext'
FIG = RES / 'figures'
WORK = SRC.parent / 'v3_build'
if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(SRC, WORK)
DOC = WORK / 'word' / 'document.xml'
x = open(DOC, encoding='utf-8').read()
b0 = x.index('<w:body>') + len('<w:body>')
b1 = x.rindex('</w:body>')
PRE, BODY, POST = x[:b0], x[b0:b1], x[b1:]


# ------------------------------------------------------------ split body items
def split_items(body):
    items, depth, start, tag = [], 0, None, None
    for m in re.finditer(r'<(/?)w:(p|tbl|sectPr)\b[^>]*?(/?)>', body):
        close, name, selfc = m.group(1), m.group(2), m.group(3)
        if depth == 0 and not close:
            start, tag = m.start(), name
            if selfc:
                items.append(body[start:m.end()])
                continue
            depth = 1
            continue
        if depth > 0 and name == tag:
            if not close and not selfc:
                depth += 1
            elif close:
                depth -= 1
                if depth == 0:
                    items.append(body[start:m.end()])
    return items


IT = split_items(BODY)


def txt(s):
    import html
    return html.unescape(''.join(re.findall(r'<w:t[^>]*>([^<]*)', s)))


assert txt(IT[0]).startswith('Explainable EHR-ML') and 'Abstract' in txt(IT[7])
assert txt(IT[93]).startswith('[1]') and txt(IT[111]).startswith('[19]')
TITLE_BLOCK = IT[0:7]                       # title, authors, affiliation, section break
OLD_TITLE = 'Explainable EHR-ML for Sepsis Outcome Prediction: An Ablation-Based Study Using MIMIC-IV and eICU'
NEW_TITLE = ('Stacked Ensembles Versus a Single Gradient-Boosted Model for Sepsis Outcome Prediction: '
             'A Controlled and Explainable Replication on MIMIC-IV and eICU')
assert TITLE_BLOCK[0].count(OLD_TITLE) == 1
TITLE_BLOCK[0] = TITLE_BLOCK[0].replace(OLD_TITLE, NEW_TITLE)
_core = WORK / 'docProps' / 'core.xml'
_c = open(_core, encoding='utf-8').read()
open(_core, 'w', encoding='utf-8').write(re.sub(r'<dc:title>[^<]*</dc:title>', f'<dc:title>{NEW_TITLE}</dc:title>', _c))
SECT_END_2COL = IT[22]                      # paragraph whose sectPr closes a 2-column section
SECT_END_1COL = IT[25]                      # closes a full-width section
FINAL_SECTPR = IT[-1]
DRAW_TPL = IT[23]

FONT = ('<w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" '
        'w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>')
R_PLAIN = f'<w:rPr>{FONT}<w:i w:val="false"/><w:iCs w:val="false"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
R_LINK = (f'<w:rPr>{FONT}<w:b w:val="false"/><w:bCs w:val="false"/><w:i w:val="false"/>'
          '<w:iCs w:val="false"/><w:color w:val="1155CC"/><w:sz w:val="20"/><w:szCs w:val="20"/>'
          '<w:u w:val="single"/></w:rPr>')
R_LEAD = f'<w:rPr>{FONT}<w:b/><w:bCs/><w:i/><w:iCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
P_BODY = '<w:pPr><w:spacing w:after="50" w:line="234"/><w:jc w:val="both"/></w:pPr>'

BM = [1000]


def bm_id():
    BM[0] += 1
    return BM[0]


# Compound terms that must not break at their hyphens: a hyphen at a line end is dropped by
# PDF text extraction (giving "metalearner", "length-ofstay"). Word's non-breaking hyphen keeps
# the term on one line and renders as a normal hyphen.
NOBREAK = ['meta-learners', 'meta-learner', 'length-of-stay', 'Leakage-safe', 'leakage-safe',
           'calibration-in-the-large', 'out-of-fold', 'in-hospital', 'fixed-configuration',
           'well-configured', 'gradient-boosted', 'best-model', 'top-K', 'k-nearest', 'Level-1',
           'Level-2', 'Nadeau-Bengio', 'EHR-ML', 'MIMIC-IV', 'early-death', 'feature-selection',
           'cross-validation', 'machine-learning', 'non-significant', 'day-2', 'one-step']
_NB_RE = re.compile('|'.join(sorted(map(re.escape, NOBREAK), key=len, reverse=True)))


def _t(t):
    return f'<w:t xml:space="preserve">{escape(t)}</w:t>'


def run(t, rpr=R_PLAIN):
    parts, pos = [], 0
    for m in _NB_RE.finditer(t):
        if m.start() > pos:
            parts.append(_t(t[pos:m.start()]))
        pieces = m.group(0).split('-')
        for i, piece in enumerate(pieces):
            if i:
                parts.append('<w:noBreakHyphen/>')
            parts.append(_t(piece))
        pos = m.end()
    if pos < len(t):
        parts.append(_t(t[pos:]))
    return f'<w:r>{rpr}{"".join(parts) or _t("")}</w:r>'


ROMAN = {'I': 'tableI', 'II': 'tableII', 'III': 'tableIII', 'IV': 'tableIV'}
LINK_RE = re.compile(r'(\[\d+\])|(Fig\. \d)|(Table (?:IV|III|II|I)\b)')


def linked_runs(text, rpr=R_PLAIN, rlink=R_LINK):
    out, pos = [], 0
    for m in LINK_RE.finditer(text):
        if m.start() > pos:
            out.append(run(text[pos:m.start()], rpr))
        tok = m.group(0)
        if tok.startswith('['):
            anchor = 'ref' + tok[1:-1]
        elif tok.startswith('Fig'):
            anchor = 'fig' + tok[-1]
        else:
            anchor = ROMAN[tok.split()[1]]
        out.append(f'<w:hyperlink w:history="1" w:anchor="{anchor}">{run(tok, rlink)}</w:hyperlink>')
        pos = m.end()
    if pos < len(text):
        out.append(run(text[pos:], rpr))
    return ''.join(out)


def para(text):
    return f'<w:p>{P_BODY}{linked_runs(text)}</w:p>'


def lead_para(lead, text):
    return f'<w:p>{P_BODY}{run(lead, R_LEAD)}{linked_runs(text)}</w:p>'


R_SEC = f'<w:rPr>{FONT}<w:b/><w:bCs/><w:caps/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
R_SUB = f'<w:rPr>{FONT}<w:b/><w:bCs/><w:i/><w:iCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
R_FOOT = f'<w:rPr>{FONT}<w:i/><w:iCs/><w:sz w:val="14"/><w:szCs w:val="14"/></w:rPr>'
R_CELL_H = f'<w:rPr>{FONT}<w:b/><w:bCs/><w:sz w:val="14"/><w:szCs w:val="14"/></w:rPr>'
R_CELL = f'<w:rPr>{FONT}<w:b w:val="false"/><w:bCs w:val="false"/><w:sz w:val="14"/><w:szCs w:val="14"/></w:rPr>'


def h_sec(t):
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:after="60" w:before="90"/><w:jc w:val="center"/></w:pPr>'
            + run(t, R_SEC) + '</w:p>')


def h_sub(t):
    return '<w:p><w:pPr><w:keepNext/><w:spacing w:after="50" w:before="80"/></w:pPr>' + run(t, R_SUB) + '</w:p>'


R_CAP = f'<w:rPr>{FONT}<w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>'


def caption(label, rest, bookmark, table=False):
    i = bm_id()
    ppr = ('<w:pPr><w:keepNext/><w:spacing w:after="80" w:before="40"/><w:jc w:val="center"/></w:pPr>' if table
           else '<w:pPr><w:spacing w:after="120"/><w:jc w:val="center"/></w:pPr>')
    return (f'<w:p>{ppr}<w:bookmarkStart w:name="{bookmark}" w:id="{i}"/>{run(label, R_CAP)}'
            f'<w:bookmarkEnd w:id="{i}"/>{run(" " + rest, R_CAP)}</w:p>')


R_FOOT_LINK = (f'<w:rPr>{FONT}<w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="14"/>'
               '<w:szCs w:val="14"/><w:u w:val="single"/></w:rPr>')


def footnote(t):
    return ('<w:p><w:pPr><w:spacing w:after="120" w:before="20"/><w:jc w:val="both"/></w:pPr>'
            + linked_runs(t, R_FOOT, R_FOOT_LINK) + '</w:p>')


# ------------------------------------------------------------------- figures
RELS = WORK / 'word' / '_rels' / 'document.xml.rels'
rels = open(RELS, encoding='utf-8').read()
DOCPR = [500]
EMU = 914400


def png_size(p):
    d = open(p, 'rb').read(24)
    return struct.unpack('>II', d[16:24])


def figure(png, width_in, rid):
    w, h = png_size(png)
    cx = int(width_in * EMU)
    cy = int(cx * h / w)
    shutil.copy(png, WORK / 'word' / 'media' / f'{rid}.png')
    global rels
    rels = rels.replace('</Relationships>',
                        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/'
                        f'2006/relationships/image" Target="media/{rid}.png"/></Relationships>')
    s = DRAW_TPL
    s = re.sub(r'r:embed="[^"]+"', f'r:embed="{rid}"', s)
    s = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{cx}" cy="{cy}"/>', s)
    s = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{cx}" cy="{cy}"/>', s)
    DOCPR[0] += 1
    s = re.sub(r'<wp:docPr id="\d+"', f'<wp:docPr id="{DOCPR[0]}"', s)
    s = s.replace('w:before="120"', 'w:before="60"')
    s = s.replace('<w:pPr><w:spacing', '<w:pPr><w:keepNext/><w:spacing', 1)
    return s


# -------------------------------------------------------------------- tables
TC_MAR = ('<w:tcMar><w:top w:type="dxa" w:w="20"/><w:left w:type="dxa" w:w="45"/>'
          '<w:bottom w:type="dxa" w:w="20"/><w:right w:type="dxa" w:w="45"/></w:tcMar>')


def cell(text, w, header=False, span=1, left=False):
    lines = text.split('\n')
    rpr = R_CELL_H if header else R_CELL
    paras = ''.join(f'<w:p><w:pPr><w:keepNext/><w:spacing w:after="0" w:line="200" w:lineRule="auto"/>'
                    f'<w:jc w:val="{"left" if left else "center"}"/></w:pPr>{run(l, rpr)}</w:p>'
                    for l in lines)
    shd = '<w:shd w:fill="D9D9D9" w:val="clear"/>' if header else ''
    gs = f'<w:gridSpan w:val="{span}"/>' if span > 1 else ''
    return (f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/>{gs}{shd}{TC_MAR}'
            f'<w:vAlign w:val="center"/></w:tcPr>{paras}</w:tc>')


def table(widths, header_rows, rows, first_left=True):
    tw = sum(widths)
    grid = ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{tw}"/><w:jc w:val="center"/><w:tblBorders>'
           '<w:top w:val="single" w:color="000000" w:sz="4"/><w:left w:val="none"/>'
           '<w:bottom w:val="single" w:color="000000" w:sz="4"/><w:right w:val="none"/>'
           '<w:insideH w:val="single" w:color="999999" w:sz="2"/><w:insideV w:val="none"/>'
           f'</w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
    for hr in header_rows:          # list of (text, span)
        cells, k = [], 0
        for t, span in hr:
            cells.append(cell(t, sum(widths[k:k + span]), header=True, span=span))
            k += span
        out.append('<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' + ''.join(cells) + '</w:tr>')
    for r in rows:
        out.append('<w:tr><w:trPr><w:cantSplit/></w:trPr>' +
                   ''.join(cell(t, widths[i], left=(first_left and i == 0)) for i, t in enumerate(r)) +
                   '</w:tr>')
    out.append('</w:tbl>')
    return ''.join(out)


def f3(v):
    return f'{v:+.3f}'.replace('-', '−') if v != 0 else '0.000'


def ci3(d, lo, hi, newline=False):
    sep = '\n' if newline else ' '
    return f'{f3(d)}{sep}({f3(lo)}, {f3(hi)})'


exec(open(HERE / 'paper_v3_text.py').read())

# ---------------------------------------------------------------- references
REF_ITEMS = {int(re.match(r'\[(\d+)\]', txt(s)).group(1)): s for s in IT[93:112]}
OLD = {k: re.sub(r'^\[\d+\]\s*', '', txt(v)) for k, v in REF_ITEMS.items()}   # v2 numbering
REF_TEXT = {1: OLD[1], 2: OLD[2], 3: OLD[3], 4: OLD[4], 5: OLD[5], 6: OLD[6], 7: OLD[7], 8: OLD[8],
            9: OLD[9], 10: OLD[10],
            11: ('G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly '
                 'efficient gradient boosting decision tree," in Advances in Neural Information Processing '
                 'Systems 30 (NeurIPS), 2017, pp. 3146\u20133154.'),
            12: OLD[11], 13: OLD[12],
            14: OLD[13].replace('H. Shanmugam, L. Airen, and S. Rawat', 'S. Rawat, H. Shanmugam, and L. Airen'),
            15: OLD[14],
            16: OLD[17].replace(', Apr. 2025, doi: 10.1016/j.ijmedinf.2025.105816.', ', 2025.'),
            17: OLD[19], 18: ('C. Nadeau and Y. Bengio, "Inference for the generalization error," Machine '
                              'Learning, vol. 52, no. 3, pp. 239\u2013281, 2003.'),
            19: OLD[18],
            20: ('F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., '
                 'vol. 12, pp. 2825\u20132830, 2011.'),
            21: OLD[15], 22: OLD[16]}
assert 'S. Rawat, H. Shanmugam' in REF_TEXT[14] and REF_TEXT[16].endswith('2025.')
assert 'Wang' in REF_TEXT[12] and 'Lei' in REF_TEXT[13] and 'EHR-ML' in REF_TEXT[16]
assert 'Guo' in REF_TEXT[17] and 'CatBoost' in REF_TEXT[19] and 'Van Calster' in REF_TEXT[21]
assert 'de Hond' in REF_TEXT[22] and 'Bomrah' in REF_TEXT[15]
REF_HEAD = IT[92]
assert txt(REF_HEAD) == 'References'

body_xml = ''.join(B)
order = []
for m in re.finditer(r'w:anchor="ref(\d+)"', body_xml):
    n = int(m.group(1))
    if n not in order:
        order.append(n)
assert order == list(range(1, 23)), order
remap = {old: new + 1 for new, old in enumerate(order)}
body_xml = re.sub(r'w:anchor="ref(\d+)"(>.*?<w:t xml:space="preserve">)\[(\d+)\]',
                  lambda m: f'w:anchor="ref{remap[int(m.group(1))]}"{m.group(2)}[{remap[int(m.group(3))]}]',
                  body_xml)
R_REF = f'<w:rPr>{FONT}<w:sz w:val="16"/><w:szCs w:val="18"/></w:rPr>'
# Each reference entry links to its DOI or official proceedings page (all checked against
# Crossref, arXiv, NeurIPS, PMLR or JMLR records on 26 Oct 2026).
REF_URL = {
    1: 'https://doi.org/10.1001/jama.2016.0287', 2: 'https://doi.org/10.1001/jama.1993.03510240069035',
    3: 'https://doi.org/10.1038/s41597-022-01899-x', 4: 'https://doi.org/10.1038/sdata.2018.178',
    5: 'https://doi.org/10.1145/2939672.2939785', 6: 'https://doi.org/10.48550/arXiv.2502.17978',
    7: 'https://doi.org/10.1016/S0893-6080(05)80023-1', 8: 'https://doi.org/10.1097/MD.0000000000037634',
    9: 'https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html',
    10: 'https://doi.org/10.1186/s40001-023-01593-7',
    11: 'https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html',
    12: 'https://doi.org/10.1007/s11739-024-03732-2', 13: 'https://doi.org/10.2196/66733',
    14: 'https://doi.org/10.5005/jp-journals-10071-24986', 15: 'https://doi.org/10.1186/s13054-024-04948-6',
    16: 'https://doi.org/10.1016/j.ijmedinf.2025.105816', 17: 'https://proceedings.mlr.press/v70/guo17a.html',
    18: 'https://doi.org/10.1023/A:1024068626366',
    19: 'https://papers.nips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html',
    20: 'https://jmlr.org/papers/v12/pedregosa11a.html', 21: 'https://doi.org/10.1186/s12916-023-02779-w',
    22: 'https://doi.org/10.1038/s41746-023-00832-9'}
assert sorted(REF_URL) == list(range(1, 23))
ref_xml = [h_sec('References')]
for old in order:
    new = remap[old]
    body_txt = REF_TEXT[old]
    i = bm_id()
    rid = f'rIdRefLink{new}'
    rels = rels.replace('</Relationships>',
                        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                        f'relationships/hyperlink" Target="{escape(REF_URL[new])}" TargetMode="External"/>'
                        '</Relationships>')
    ref_xml.append('<w:p><w:pPr><w:spacing w:after="30" w:line="228"/><w:ind w:left="240" w:hanging="240"/>'
                   f'<w:jc w:val="both"/></w:pPr><w:bookmarkStart w:name="ref{new}" w:id="{i}"/>'
                   f'{run(f"[{new}]", R_REF)}<w:bookmarkEnd w:id="{i}"/>'
                   f'<w:hyperlink r:id="{rid}" w:history="1">{run(" " + body_txt, R_REF)}</w:hyperlink></w:p>')

NEW_BODY = ''.join(TITLE_BLOCK) + body_xml + ''.join(ref_xml) + FINAL_SECTPR
assert not re.search(r"\{f\[|\{nd\[|\{tw\[|\{ab\[|\{neg\(|\{f3\(|\{p2\(", NEW_BODY), 'unformatted placeholder in text'
open(DOC, 'w', encoding='utf-8').write(PRE + NEW_BODY + POST)

# drop image relationships that are no longer used, and their files
used = set(re.findall(r'r:embed="([^"]+)"', NEW_BODY))
keep = []
for m in re.finditer(r'<Relationship [^>]*/>', rels):
    r = m.group(0)
    if 'relationships/image' in r:
        rid = re.search(r'Id="([^"]+)"', r).group(1)
        if rid not in used:
            tgt = re.search(r'Target="([^"]+)"', r).group(1)
            (WORK / 'word' / tgt).unlink(missing_ok=True)
            continue
    keep.append(r)
head = rels[:rels.index('<Relationship ')]
open(RELS, 'w', encoding='utf-8').write(head + ''.join(keep) + '</Relationships>')

# remove the leftover comments part if nothing references it
if 'commentReference' not in NEW_BODY:
    pass
OUT_DOCX.unlink(missing_ok=True)
subprocess.run(['zip', '-qXr', str(OUT_DOCX), '.'], cwd=WORK, check=True)
print('wrote', OUT_DOCX, 'refs order', order)
print('words:', len(' '.join(re.findall(r'<w:t[^>]*>([^<]*)', NEW_BODY)).split()))
