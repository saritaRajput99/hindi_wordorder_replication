"""
Generate architecture diagram for PPT — clean, large text, high contrast.

Run:
    cd D:\MTP_WO\hindi_wordorder_replication
    python scripts/make_diagram.py

Output: scripts/word_order_diagram.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), 'word_order_diagram.png')

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 22, 14
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor('#FFFFFF')

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = '#FFFFFF'
C_TITLE  = '#1A237E'   # dark navy
C_S      = '#1565C0'   # blue  — Surprisal
C_D      = '#2E7D32'   # green — DL
C_I      = '#6A1B9A'   # purple — IS
C_METHOD = '#E3F2FD'   # very light blue method boxes
C_MERGE  = '#FF6F00'   # orange — preferred order
C_PAIR   = '#37474F'   # dark grey — pairwise
C_LR     = '#B71C1C'   # red — logistic regression
ARROW    = '#212121'
WHITE    = '#FFFFFF'
DARK     = '#212121'

# ── Helper: rounded box ───────────────────────────────────────────────────────
def rbox(cx, cy, w, h, fc, ec=DARK, lw=2, radius=0.25, alpha=1.0):
    p = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=3
    )
    ax.add_patch(p)

def txt(x, y, s, size=13, color=DARK, bold=False, va='center'):
    ax.text(x, y, s, ha='center', va=va,
            fontsize=size, color=color,
            fontweight='bold' if bold else 'normal', zorder=5)

def arrow(x1, y1, x2, y2, lw=2.5, color=ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    mutation_scale=20,
                    connectionstyle='arc3,rad=0.0'),
                zorder=6)

def hline(x1, x2, y, lw=2.5, color=ARROW):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, zorder=4)

def vline(x, y1, y2, lw=2.5, color=ARROW):
    ax.plot([x, x], [y1, y2], color=color, lw=lw, zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Title
# ═══════════════════════════════════════════════════════════════════════════════
rbox(11, 13.1, 16, 1.1, C_TITLE, ec='#1A237E', lw=3)
txt(11, 13.45, 'What Drives Hindi Word Order?', size=22, color=WHITE, bold=True)
txt(11, 12.82, 'Replication of Ranjan & van Schijndel (2024)', size=13, color='#BBDEFB')

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Three pillars
# ═══════════════════════════════════════════════════════════════════════════════
Y2 = 11.2
pillar_y  = 11.2
pillar_h  = 1.1
pillar_w  = 5.8

XL, XM, XR = 4.0, 11.0, 18.0   # left / middle / right column centres

# ── branch from title ─────────────────────────────────────────────────────────
vline(11, 12.55, 11.85)
hline(XL, XR, 11.85)
for xb in [XL, XM, XR]:
    vline(xb, 11.85, 11.75)
    ax.annotate('', xy=(xb, Y2 + pillar_h/2),
                xytext=(xb, 11.75),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=2.5,
                                mutation_scale=18), zorder=6)

# ── Pillar boxes ──────────────────────────────────────────────────────────────
rbox(XL, pillar_y, pillar_w, pillar_h, C_S, ec=C_S, lw=2.5)
txt(XL, pillar_y + 0.2, 'Surprisal', size=17, color=WHITE, bold=True)
txt(XL, pillar_y - 0.25, 'Predictability in Context', size=12, color='#BBDEFB')

rbox(XM, pillar_y, pillar_w, pillar_h, C_D, ec=C_D, lw=2.5)
txt(XM, pillar_y + 0.2, 'Dependency Length', size=17, color=WHITE, bold=True)
txt(XM, pillar_y - 0.25, 'Memory / Processing Cost', size=12, color='#C8E6C9')

rbox(XR, pillar_y, pillar_w, pillar_h, C_I, ec=C_I, lw=2.5)
txt(XR, pillar_y + 0.2, 'Information Structure', size=17, color=WHITE, bold=True)
txt(XR, pillar_y - 0.25, 'Given  →  New  (Givenness)', size=12, color='#E1BEE7')

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Method boxes
# ═══════════════════════════════════════════════════════════════════════════════
Y3     = 9.2
meth_h = 1.3
meth_w = 5.8

for xb in [XL, XM, XR]:
    arrow(xb, pillar_y - pillar_h/2, xb, Y3 + meth_h/2)

# Left: Language Models
rbox(XL, Y3, meth_w, meth_h, '#E3F2FD', ec=C_S, lw=2)
txt(XL, Y3 + 0.38, 'Language Models', size=15, color=C_S, bold=True)
txt(XL, Y3 + 0.02, 'Trigram  ·  PCFG  ·  Adaptive LSTM', size=12, color=DARK)
txt(XL, Y3 - 0.38, 'P(sentence) from each model', size=11, color='#555555')

# Middle: DL formula
rbox(XM, Y3, meth_w, meth_h, '#E8F5E9', ec=C_D, lw=2)
txt(XM, Y3 + 0.38, 'DL Computation', size=15, color=C_D, bold=True)
txt(XM, Y3 + 0.02, 'DL = Σ | pos(dep) − pos(head) |', size=13, color=DARK)
txt(XM, Y3 - 0.38, 'For all non-root tokens in sentence', size=11, color='#555555')

# Right: IS Scoring
rbox(XR, Y3, meth_w, meth_h, '#F3E5F5', ec=C_I, lw=2)
txt(XR, Y3 + 0.38, 'IS Scoring', size=15, color=C_I, bold=True)
txt(XR, Y3 + 0.02, 'Given = +1   ·   New = −1   ·   Unknown = 0', size=12, color=DARK)
txt(XR, Y3 - 0.38, 'Based on discourse context', size=11, color='#555555')

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Preferred word order (convergence)
# ═══════════════════════════════════════════════════════════════════════════════
Y4     = 7.2
box4_h = 1.0
box4_w = 13

# Converging arrows
arrow(XL, Y3 - meth_h/2, 8.5,  Y4 + box4_h/2)
arrow(XM, Y3 - meth_h/2, 11.0, Y4 + box4_h/2)
arrow(XR, Y3 - meth_h/2, 13.5, Y4 + box4_h/2)

rbox(11, Y4, box4_w, box4_h, C_MERGE, ec='#E65100', lw=3)
txt(11, Y4 + 0.2, 'Preferred Word Order', size=18, color=WHITE, bold=True)
txt(11, Y4 - 0.22, 'Corpus sentence  is preferred over  Generated variant', size=13, color='#FFF8E1')

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Pairwise transform
# ═══════════════════════════════════════════════════════════════════════════════
Y5     = 5.6
box5_h = 1.1
box5_w = 13

arrow(11, Y4 - box4_h/2, 11, Y5 + box5_h/2)

rbox(11, Y5, box5_w, box5_h, C_PAIR, ec='#263238', lw=3)
txt(11, Y5 + 0.28, 'Pairwise Ranking Transform  (Joachims 2002)', size=15, color=WHITE, bold=True)
txt(11, Y5 - 0.08, 'Δ features  =  φ(corpus) − φ(variant)      Label = 1', size=13, color='#CFD8DC')
txt(11, Y5 - 0.42, 'Balanced dataset: 50% corpus preferred  ·  50% variant preferred', size=11.5, color='#90A4AE')

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6 — Logistic Regression result
# ═══════════════════════════════════════════════════════════════════════════════
Y6     = 3.85
box6_h = 1.5
box6_w = 16

arrow(11, Y5 - box5_h/2, 11, Y6 + box6_h/2)

rbox(11, Y6, box6_w, box6_h, C_LR, ec='#7F0000', lw=3)
txt(11, Y6 + 0.52,  'Logistic Regression  ·  10-fold Cross-Validation', size=17, color=WHITE, bold=True)
txt(11, Y6 + 0.09,  'All: 94.0%     Canonical: 94.6%     IO-fronted: 94.8%     O-fronted: 84.6%     DO-fronted: 76.8%', size=13, color='#FFCDD2')
txt(11, Y6 - 0.3,   'Surprisal (Trigram) = strongest predictor  (coef −2.08, p < 0.001)', size=12.5, color='#FFAB91')
txt(11, Y6 - 0.62,  'DL significant for DO-fronted only  ·  IS score significant for DO-fronted', size=12, color='#FFCCBC')

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 7 — Finding summary
# ═══════════════════════════════════════════════════════════════════════════════
Y7     = 2.2
box7_h = 0.85
box7_w = 16

arrow(11, Y6 - box6_h/2, 11, Y7 + box7_h/2)

rbox(11, Y7, box7_w, box7_h, '#F57F17', ec='#E65100', lw=3)
txt(11, Y7 + 0.17, 'Key Finding: Surprisal is the dominant driver of non-canonical Hindi word order', size=14, color=WHITE, bold=True)
txt(11, Y7 - 0.22, 'DL & Information Structure provide additional but secondary explanatory power', size=12.5, color='#FFF8E1')

# ── Footer ────────────────────────────────────────────────────────────────────
txt(11, 1.15,
    'Data: UD Hindi-HDTB (HUTB)  ·  13,306 raw sentences  →  2,320 filtered  →  90,520 variants',
    size=11, color='#757575')

plt.tight_layout(pad=0.3)
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f'Saved: {OUT_PATH}')
