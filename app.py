import streamlit as st
import numpy as np
import pandas as pd
import io

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="LJK Scanner",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
  --navy:    #111844;
  --mid:     #4B5694;
  --steel:   #7288AE;
  --cream:   #EAE0CF;
  --cream2:  #F5F0E8;
  --white:   #FFFFFF;
  --success: #3D8B6E;
  --warn:    #C47C2B;
  --danger:  #B34040;
  --text:    #EAE0CF;
  --text-dim:#7288AE;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--navy) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--cream) !important;
}
[data-testid="stAppViewContainer"] > .main {
  background: transparent !important;
}
[data-testid="block-container"] {
  padding: 2rem 2.5rem 3rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #0C1235 !important;
  border-right: 1px solid rgba(74,86,148,0.3) !important;
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }
[data-testid="stSidebar"] .stButton > button {
  background: rgba(74,86,148,0.15) !important;
  border: 1px solid rgba(74,86,148,0.4) !important;
  color: var(--cream) !important;
  width: 100%;
  border-radius: 8px;
  font-size: 0.8rem;
  padding: 6px 10px;
  transition: all .2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(74,86,148,0.35) !important;
  border-color: var(--steel) !important;
}

/* ── Typography ── */
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: var(--cream) !important; }
.serif-title {
  font-family: 'DM Serif Display', serif;
  font-size: 2.4rem;
  color: var(--cream);
  line-height: 1.15;
  letter-spacing: -0.02em;
}
.serif-title span { color: var(--cream); font-style: italic; background: linear-gradient(135deg, #EAE0CF, #7288AE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #9BAABF;
  margin-bottom: 0.4rem;
}
.page-subtitle {
  color: var(--steel);
  font-size: 0.95rem;
  font-weight: 300;
  margin-top: 0.3rem;
}

/* ── Step Nav ── */
.step-nav {
  display: flex;
  gap: 0;
  margin-bottom: 2rem;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(74,86,148,0.3);
}
.step-item {
  flex: 1;
  padding: 12px 8px;
  text-align: center;
  font-size: 0.78rem;
  font-weight: 500;
  background: rgba(255,255,255,0.02);
  color: var(--text-dim);
  transition: all .2s;
  border-right: 1px solid rgba(74,86,148,0.2);
  cursor: default;
}
.step-item:last-child { border-right: none; }
.step-item.active {
  background: linear-gradient(180deg, rgba(74,86,148,0.3), rgba(17,24,68,0.4));
  color: var(--cream);
  font-weight: 600;
  border-bottom: 2px solid #EAE0CF;
}
.step-item.done {
  background: rgba(74,86,148,0.12);
  color: #7288AE;
}
.step-icon { font-size: 1rem; display: block; margin-bottom: 3px; }

/* ── Cards ── */
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(74,86,148,0.25);
  border-top: 2px solid rgba(234,224,207,0.3);
  border-radius: 16px;
  padding: 1.2rem 1.5rem 0.8rem;
  margin-bottom: 1rem;
}
.card-sm {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(74,86,148,0.2);
  border-radius: 12px;
  padding: 1rem 1.2rem;
}

/* ── Metric Tiles ── */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin: 1.2rem 0;
}
.metric-tile {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(74,86,148,0.2);
  border-top: 2px solid rgba(234,224,207,0.2);
  border-radius: 14px;
  padding: 18px 14px;
  text-align: center;
  transition: transform .2s, border-color .2s, border-top-color .2s;
}
.metric-tile:hover { transform: translateY(-2px); border-color: var(--steel); border-top-color: rgba(234,224,207,0.55); }
.metric-tile .t-val {
  font-family: 'DM Serif Display', serif;
  font-size: 2rem;
  line-height: 1;
  margin-bottom: 6px;
}
.metric-tile .t-lbl {
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
}
.c-blue   { color: #7CA4D4; }
.c-green  { color: #6DBF9E; }
.c-red    { color: #E07575; }
.c-grey   { color: #7288AE; }
.c-amber  { color: #D4A96A; }
.c-cream  { color: var(--cream); }

/* ── Answer Grid ── */
.answer-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(68px, 1fr));
  gap: 6px;
  margin-top: 0.8rem;
}
.ans-cell {
  border-radius: 8px;
  padding: 6px 4px;
  text-align: center;
  font-size: 0.72rem;
  font-family: 'JetBrains Mono', monospace;
  border: 1px solid rgba(74,86,148,0.2);
  background: rgba(255,255,255,0.02);
}
.ans-cell.correct { background: rgba(61,139,110,0.15); border-color: #3D8B6E; color: #6DBF9E; }
.ans-cell.wrong   { background: rgba(179,64,64,0.15);  border-color: #B34040; color: #E07575; }
.ans-cell.empty   { opacity: 0.45; }

/* ── Streamlit Overrides ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox select {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(74,86,148,0.35) !important;
  border-radius: 10px !important;
  color: var(--cream) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--steel) !important;
  box-shadow: 0 0 0 2px rgba(114,136,174,0.15) !important;
}
label, .stSelectbox label, .stTextInput label,
.stNumberInput label, .stTextArea label {
  color: var(--steel) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
}
.stButton > button {
  background: linear-gradient(135deg, #4B5694, #3d4878) !important;
  color: #EAE0CF !important;
  border: 1px solid rgba(234,224,207,0.15) !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  padding: 10px 24px !important;
  transition: all .2s !important;
  letter-spacing: 0.02em !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #7288AE, #4B5694) !important;
  border-color: rgba(234,224,207,0.35) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(17,24,68,0.5) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.03) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 4px !important;
  border: 1px solid rgba(74,86,148,0.2) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-dim) !important;
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(74,86,148,0.35) !important;
  color: var(--cream) !important;
}
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(74,86,148,0.2) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
  color: var(--cream) !important;
  font-weight: 500 !important;
}
.stSuccess, .stInfo, .stWarning, .stError {
  border-radius: 10px !important;
}
[data-testid="stDataFrame"] {
  border-radius: 12px !important;
  overflow: hidden !important;
}
.stFileUploader {
  background: rgba(255,255,255,0.02) !important;
  border: 2px dashed rgba(74,86,148,0.4) !important;
  border-radius: 14px !important;
  transition: border-color .2s !important;
}
.stFileUploader:hover {
  border-color: var(--steel) !important;
}
div[data-testid="stFileUploaderDropzone"] {
  background: transparent !important;
}
.stDownloadButton > button {
  background: rgba(61,139,110,0.2) !important;
  border: 1px solid rgba(61,139,110,0.5) !important;
  color: #6DBF9E !important;
}
.stDownloadButton > button:hover {
  background: rgba(61,139,110,0.35) !important;
}

/* ── Divider ── */
hr { border-color: rgba(74,86,148,0.2) !important; }

/* ── Heatmap container ── */
.heatmap-wrap {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(74,86,148,0.2);
  border-radius: 14px;
  padding: 1.2rem;
}
.heatmap-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--steel);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.8rem;
}

/* ── Badge ── */
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.badge-blue  { background: rgba(114,136,174,0.2); color: #7CA4D4; border: 1px solid rgba(114,136,174,0.3); }
.badge-green { background: rgba(61,139,110,0.2);  color: #6DBF9E; border: 1px solid rgba(61,139,110,0.3); }
.badge-amber { background: rgba(196,124,43,0.2);  color: #D4A96A; border: 1px solid rgba(196,124,43,0.3); }
.badge-red   { background: rgba(179,64,64,0.2);   color: #E07575; border: 1px solid rgba(179,64,64,0.3); }

/* ── ML Metric Cards ── */
.ml-metric {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(74,86,148,0.22);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.ml-metric .m-name {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.ml-metric .m-val {
  font-family: 'DM Serif Display', serif;
  font-size: 1.4rem;
}

/* ── Progress bar ── */
.prog-bar-wrap { margin: 6px 0; }
.prog-bar-label {
  display: flex; justify-content: space-between;
  font-size: 0.72rem; color: var(--text-dim); margin-bottom: 3px;
}
.prog-bar-bg {
  background: rgba(74,86,148,0.15);
  border-radius: 4px; height: 6px; overflow: hidden;
}
.prog-bar-fill { height: 100%; border-radius: 4px; transition: width .6s ease; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,86,148,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────
def init_state():
    defaults = {
        'answer_key':   {},
        'total_soal':   50,
        'sesi_nama':    '',
        'kode_kelas':   '',
        'kode_dosen':   '',
        'scoring':      'standard',
        'records':      [],
        'ml_results':   None,
        'step':         'setup',
        'key_text':     '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

DEFAULT_50 = ['B','C','A','D','E','A','B','C','D','A','E','B','C','A','D','B','E','A','C','D',
              'A','B','E','C','D','B','A','D','C','E','A','C','B','D','E','C','A','B','D','E',
              'B','D','A','C','E','A','D','B','E','C']

def make_key_text(n):
    lines = []
    for i in range(1, n+1):
        ans = DEFAULT_50[i-1] if i <= 50 else 'A'
        lines.append(f"{i}. {ans}")
    return "\n".join(lines)

# Init key_text if empty
if not st.session_state.key_text:
    st.session_state.key_text = make_key_text(st.session_state.total_soal)

# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 16px">
      <div style="font-family:'DM Serif Display',serif; font-size:1.9rem; color:#EAE0CF; line-height:1.1; letter-spacing:-0.02em">
        LJK Scanner
      </div>
      <div style="font-size:0.82rem; color:#7288AE; margin-top:6px; letter-spacing:0.1em; text-transform:uppercase; font-weight:500">
        Computer Vision Project
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Step indicators
    steps = [
        ('setup',   '⚙', 'Setup'),
        ('scan',    '📸', 'Scan'),
        ('results', '📊', 'Hasil'),
        ('ml',      '🤖', 'Model ML'),
    ]
    cur = st.session_state.step
    cur_idx = [s[0] for s in steps].index(cur)

    for i, (s, icon, lbl) in enumerate(steps):
        if i < cur_idx:
            cls = "done"; dot = "✓"
        elif i == cur_idx:
            cls = "active"; dot = icon
        else:
            cls = ""; dot = icon
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;'
            f'border-radius:8px;margin-bottom:4px;'
            f'background:{"rgba(74,86,148,0.2)" if cls=="active" else "rgba(61,139,110,0.1)" if cls=="done" else "transparent"};'
            f'border:1px solid {"rgba(74,86,148,0.4)" if cls=="active" else "rgba(61,139,110,0.25)" if cls=="done" else "transparent"}">'
            f'<span style="font-size:1rem">{dot}</span>'
            f'<span style="font-size:0.92rem;color:{"#EAE0CF" if cls in ["active","done"] else "#4B5694"};'
            f'font-weight:{"600" if cls=="active" else "400"}">{lbl}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # Session stats
    records = st.session_state.records
    if records:
        scores = [r['score'] for r in records]
        st.markdown(f'<div class="section-label">Sesi Aktif</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card-sm" style="margin-bottom:8px">
          <div style="font-size:0.78rem;color:#7288AE;margin-bottom:8px">
            {st.session_state.sesi_nama or "—"}<br>
            <span style="font-size:0.68rem;font-family:monospace">{st.session_state.kode_kelas}</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
            <div style="text-align:center">
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#EAE0CF">{len(records)}</div>
              <div style="font-size:0.65rem;color:#4B5694;text-transform:uppercase;letter-spacing:0.1em">Scanned</div>
            </div>
            <div style="text-align:center">
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#7CA4D4">{np.mean(scores):.1f}</div>
              <div style="font-size:0.65rem;color:#4B5694;text-transform:uppercase;letter-spacing:0.1em">Avg</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Grade distribution mini
        grade_counts = {'A':0,'B':0,'C':0,'D':0,'E':0}
        for s in scores:
            if s>=80: grade_counts['A']+=1
            elif s>=70: grade_counts['B']+=1
            elif s>=60: grade_counts['C']+=1
            elif s>=50: grade_counts['D']+=1
            else: grade_counts['E']+=1
        grade_colors = {'A':'#6DBF9E','B':'#7CA4D4','C':'#D4A96A','D':'#E07575','E':'#9B7E7E'}
        for g, cnt in grade_counts.items():
            if cnt == 0: continue
            pct = cnt / len(records) * 100
            st.markdown(f"""
            <div class="prog-bar-wrap">
              <div class="prog-bar-label"><span>Grade {g}</span><span>{cnt}</span></div>
              <div class="prog-bar-bg">
                <div class="prog-bar-fill" style="width:{pct}%;background:{grade_colors[g]}"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:#4B5694;font-size:0.88rem;padding:8px 0;font-style:italic">Belum ada data scan</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("↺  Reset Sesi"):
        for k in ['answer_key','records','ml_results','step','sesi_nama','kode_kelas','kode_dosen','key_text']:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

# ─── STEP INDICATOR (top of main) ───────────────────────────
step_html = '<div class="step-nav">'
for i, (s, icon, lbl) in enumerate(steps):
    if i < cur_idx:
        cls = "done"; badge_icon = "✓"
    elif i == cur_idx:
        cls = "active"; badge_icon = icon
    else:
        cls = ""; badge_icon = icon
    step_html += f'<div class="step-item {cls}"><span class="step-icon">{badge_icon}</span>{lbl}</div>'
step_html += '</div>'
st.markdown(step_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  STEP 1 — SETUP
# ══════════════════════════════════════════════════════════════
if st.session_state.step == 'setup':
    st.markdown('<div class="section-label">Langkah 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Setup <span>Sesi Ujian</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Konfigurasikan parameter ujian dan input kunci jawaban sebelum memulai scan.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('''
        <div class="card">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;
            padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)">
            <div style="width:3px;height:22px;background:linear-gradient(180deg,#EAE0CF,#7288AE);
              border-radius:2px;flex-shrink:0"></div>
            <div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.05rem;color:#EAE0CF">Informasi Sesi</div>
              <div style="font-size:0.68rem;color:#4B5694;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px">Parameter ujian</div>
            </div>
          </div>
        ''', unsafe_allow_html=True)

        st.session_state.sesi_nama  = st.text_input("Nama Sesi / Mata Kuliah", value=st.session_state.sesi_nama or "Computer Vision UAS")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.kode_kelas = st.text_input("Kode Kelas", value=st.session_state.kode_kelas or "LK01")
        with c2:
            st.session_state.kode_dosen = st.text_input("Kode Dosen", value=st.session_state.kode_dosen or "DS123")

        new_total = st.number_input("Jumlah Soal", min_value=1, max_value=100,
                                     value=st.session_state.total_soal)
        if new_total != st.session_state.total_soal:
            st.session_state.total_soal = new_total
            st.session_state.key_text = make_key_text(new_total)
            st.rerun()

        st.session_state.scoring = st.selectbox(
            "Metode Penilaian",
            ["standard", "penalty"],
            format_func=lambda x: "Standar — (Benar / Total) × 100"
                                   if x == "standard"
                                   else "Penalty — Benar − (Salah × 0.25)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="card">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;
            padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)">
            <div style="width:3px;height:22px;background:linear-gradient(180deg,#EAE0CF,#7288AE);
              border-radius:2px;flex-shrink:0"></div>
            <div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.05rem;color:#EAE0CF">Kunci Jawaban</div>
              <div style="font-size:0.68rem;color:#4B5694;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px">Format: 1. A, 2. B, ...</div>
            </div>
          </div>
        ''', unsafe_allow_html=True)

        key_text = st.text_area(
            "Format: `1. A`, `2. B`, ...",
            value=st.session_state.key_text,
            height=280,
            label_visibility="collapsed"
        )
        st.session_state.key_text = key_text

        # Parse
        answer_key = {}; errors = []
        for line in key_text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            if '. ' in line:
                parts = line.split('. ', 1)
            else:
                parts = line.split(',', 1)
            if len(parts) != 2: errors.append(f"Format salah: `{line}`"); continue
            try:
                q = int(parts[0].strip()); ans = parts[1].strip().upper()
                if ans not in ['A','B','C','D','E']:
                    errors.append(f"Jawaban tidak valid di soal {q}: `{ans}`"); continue
                answer_key[q] = ans
            except ValueError:
                errors.append(f"Nomor tidak valid: `{line}`")

        if errors:
            for e in errors[:2]: st.error(e)
        else:
            total = st.session_state.total_soal
            filled = len(answer_key)
            pct = filled / total * 100 if total > 0 else 0
            color = "#6DBF9E" if filled == total else "#D4A96A"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
              background:rgba(61,139,110,0.08);border:1px solid rgba(61,139,110,0.25);
              border-radius:10px;margin-top:8px">
              <span style="font-size:1.1rem">{'✅' if filled==total else '⚠️'}</span>
              <div>
                <div style="font-size:0.85rem;color:{color};font-weight:600">
                  {filled} / {total} kunci jawaban valid
                </div>
                <div style="font-size:0.7rem;color:#4B5694">{pct:.0f}% terisi</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Mulai Scan  →", disabled=len(answer_key) == 0):
            st.session_state.answer_key = answer_key
            st.session_state.step = 'scan'
            st.rerun()

# ══════════════════════════════════════════════════════════════
#  STEP 2 — SCAN
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 'scan':
    # Header
    st.markdown('<div class="section-label">Langkah 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Scan <span>Lembar Jawaban</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{st.session_state.sesi_nama} &nbsp;·&nbsp; {st.session_state.kode_kelas} &nbsp;·&nbsp; {st.session_state.total_soal} soal &nbsp;·&nbsp; {len(st.session_state.records)} lembar ter-scan</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Nav
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("← Setup"):
            st.session_state.step = 'setup'; st.rerun()
    with c2:
        if st.button("Lihat Hasil →", disabled=len(st.session_state.records) == 0):
            st.session_state.step = 'results'; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload area — Tab: upload file OR take photo
    st.markdown("""
    <div style="margin-bottom:0.6rem">
      <div style="font-family:'DM Serif Display',serif;font-size:1.05rem;color:#EAE0CF;margin-bottom:4px">
        Upload atau Foto Langsung
      </div>
      <div style="font-size:0.82rem;color:#7288AE">Pilih salah satu cara di bawah untuk memasukkan LJK.</div>
    </div>
    """, unsafe_allow_html=True)

    upload_tab, camera_tab = st.tabs(["📁  Upload File", "📷  Ambil Foto (Webcam)"])

    uploaded_files = []
    camera_image = None

    with upload_tab:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _files = st.file_uploader(
            "Pilih foto LJK (JPG / PNG) — bisa lebih dari satu",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="visible",
        )
        if _files:
            uploaded_files = _files

    with camera_tab:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.8rem;color:#7288AE;margin-bottom:8px">'
            'Arahkan kamera ke LJK, pastikan semua sudut terlihat jelas.</div>',
            unsafe_allow_html=True
        )
        camera_image = st.camera_input("Ambil foto LJK", label_visibility="collapsed")
        if camera_image is not None:
            from PIL import Image as PILImage
            uploaded_files = [camera_image]  # treat same as uploaded

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if uploaded_files:
        for uploaded in uploaded_files:
            fname = getattr(uploaded, 'name', None) or 'webcam_capture.jpg'
            if any(r['filename'] == fname for r in st.session_state.records):
                st.warning(f"⚠️ `{fname}` sudah di-scan, dilewati.")
                continue

            with st.expander(f"📄  {fname}", expanded=True):
                from PIL import Image
                img_pil = Image.open(uploaded)

                # ── Import scanner (dibuat oleh anggota lain) ──
                try:
                    import scanner
                    warped_np, ok = scanner.warp_ljk(img_pil)
                except ImportError:
                    st.warning("⚠️ `scanner.py` belum tersedia. Menampilkan preview saja.")
                    st.image(img_pil, caption="Preview", use_container_width=True)
                    continue

                col_orig, col_warp = st.columns(2, gap="medium")
                with col_orig:
                    st.markdown('<div class="section-label">Input Asli</div>', unsafe_allow_html=True)
                    st.image(img_pil, use_container_width=True)
                with col_warp:
                    st.markdown('<div class="section-label">Setelah Warp</div>', unsafe_allow_html=True)
                    if ok:
                        st.image(warped_np, use_container_width=True)
                    else:
                        st.error("❌ Gagal mendeteksi 4 sudut LJK.")
                        continue

                with st.spinner("Mendeteksi nama, NIM, dan jawaban..."):
                    nama    = scanner.detect_nama(warped_np)
                    nim     = scanner.detect_nim(warped_np)
                    tanggal = scanner.detect_tanggal(warped_np)
                    answers = scanner.detect_answers(warped_np, st.session_state.total_soal)
                    heatmap = scanner.get_heatmap(warped_np) if hasattr(scanner, 'get_heatmap') else None

                correct, wrong, unanswered = 0, 0, 0
                for q, key in st.session_state.answer_key.items():
                    s = answers.get(q)
                    if s is None: unanswered += 1
                    elif s == key: correct += 1
                    else: wrong += 1

                scoring = st.session_state.scoring
                total_soal = st.session_state.total_soal
                if scoring == "standard":
                    score = round(correct / total_soal * 100, 2) if total_soal > 0 else 0
                else:
                    score = round(max(0, (correct - wrong * 0.25) / total_soal * 100), 2)

                # Grade badge
                if score >= 80: grade, grade_cls = 'A', 'badge-green'
                elif score >= 70: grade, grade_cls = 'B', 'badge-blue'
                elif score >= 60: grade, grade_cls = 'C', 'badge-amber'
                else: grade, grade_cls = 'D/E', 'badge-red'

                # ── Metric tiles ──
                st.markdown(f"""
                <div class="metrics-row">
                  <div class="metric-tile">
                    <div class="t-val c-blue" style="font-size:2.2rem">{score}</div>
                    <div class="t-lbl">Skor</div>
                  </div>
                  <div class="metric-tile">
                    <div class="t-val c-green">{correct}</div>
                    <div class="t-lbl">Benar</div>
                  </div>
                  <div class="metric-tile">
                    <div class="t-val c-red">{wrong}</div>
                    <div class="t-lbl">Salah</div>
                  </div>
                  <div class="metric-tile">
                    <div class="t-val c-grey">{unanswered}</div>
                    <div class="t-lbl">Kosong</div>
                  </div>
                  <div class="metric-tile">
                    <span class="badge {grade_cls}" style="font-size:1.4rem;padding:6px 16px">{grade}</span>
                    <div class="t-lbl" style="margin-top:8px">Grade</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Identity row ──
                st.markdown(f"""
                <div class="card-sm" style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap">
                  <div>
                    <div class="section-label">Nama</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#EAE0CF">{nama}</div>
                  </div>
                  <div>
                    <div class="section-label">NIM</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;color:#7CA4D4">{nim}</div>
                  </div>
                  <div>
                    <div class="section-label">Tanggal</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:#EAE0CF">{tanggal}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Heatmap + Answer Grid ──
                col_heat, col_ans = st.columns([1, 1], gap="medium")

                with col_heat:
                    st.markdown('<div class="section-label" style="margin-top:12px">Heatmap Bubble</div>', unsafe_allow_html=True)
                    if heatmap is not None:
                        st.markdown('<div class="heatmap-wrap">', unsafe_allow_html=True)
                        st.image(heatmap, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="heatmap-wrap" style="min-height:180px;display:flex;
                          align-items:center;justify-content:center;flex-direction:column;gap:8px">
                          <div style="font-size:2rem;opacity:0.3">🗺</div>
                          <div style="font-size:0.78rem;color:#4B5694">
                            Heatmap tersedia setelah<br><code>scanner.get_heatmap()</code> diimplementasikan
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_ans:
                    st.markdown('<div class="section-label" style="margin-top:12px">Detail Jawaban</div>', unsafe_allow_html=True)
                    key = st.session_state.answer_key
                    grid_html = '<div class="answer-grid">'
                    for q in range(1, total_soal + 1):
                        s_ans = answers.get(q)
                        k_ans = key.get(q, '?')
                        if s_ans is None:
                            cls = "empty"; txt = f"{q}. —"
                        elif s_ans == k_ans:
                            cls = "correct"; txt = f"{q}. {s_ans}"
                        else:
                            cls = "wrong"; txt = f"{q}. {s_ans}"
                        grid_html += f'<div class="ans-cell {cls}">{txt}</div>'
                    grid_html += '</div>'
                    st.markdown(grid_html, unsafe_allow_html=True)

                # Save
                record = {
                    'filename': fname, 'nama': nama, 'nim': nim,
                    'tanggal': tanggal, 'correct': correct, 'wrong': wrong,
                    'unanswered': unanswered, 'score': score,
                    'total_soal': total_soal,
                    'student_answers': {str(k): v for k, v in answers.items()},
                    'answer_key': st.session_state.answer_key,
                }
                st.session_state.records.append(record)
                st.session_state.ml_results = None
                st.success(f"✅  `{fname}` berhasil disimpan.")

# ══════════════════════════════════════════════════════════════
#  STEP 3 — RESULTS
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 'results':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    records = st.session_state.records
    if not records:
        st.warning("Belum ada data. Silakan scan LJK terlebih dahulu.")
        if st.button("← Scan"): st.session_state.step = 'scan'; st.rerun()
        st.stop()

    st.markdown('<div class="section-label">Langkah 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Hasil & <span>Analitik Kelas</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{st.session_state.sesi_nama} &nbsp;·&nbsp; {st.session_state.kode_kelas}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    with c1:
        if st.button("← Scan Lagi"): st.session_state.step = 'scan'; st.rerun()
    with c2:
        if st.button("Model ML →"): st.session_state.step = 'ml'; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    scores = [r['score'] for r in records]
    total_soal = records[0]['total_soal']

    # Summary metrics
    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-tile">
        <div class="t-val c-cream">{len(records)}</div>
        <div class="t-lbl">Mahasiswa</div>
      </div>
      <div class="metric-tile">
        <div class="t-val c-blue">{np.mean(scores):.1f}</div>
        <div class="t-lbl">Rata-rata</div>
      </div>
      <div class="metric-tile">
        <div class="t-val c-green">{max(scores):.1f}</div>
        <div class="t-lbl">Tertinggi</div>
      </div>
      <div class="metric-tile">
        <div class="t-val c-red">{min(scores):.1f}</div>
        <div class="t-lbl">Terendah</div>
      </div>
      <div class="metric-tile">
        <div class="t-val c-amber">{np.std(scores):.2f}</div>
        <div class="t-lbl">Std Dev</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋  Tabel Nilai", "📈  Grafik & Distribusi", "💾  Export"])

    COLORS = {
        'navy': '#111844', 'mid': '#4B5694', 'steel': '#7288AE',
        'cream': '#EAE0CF', 'green': '#6DBF9E', 'red': '#E07575',
        'blue': '#7CA4D4', 'amber': '#D4A96A',
    }

    def style_axes(ax):
        ax.set_facecolor('#0C1235')
        for sp in ax.spines.values(): sp.set_color(COLORS['mid'])
        ax.tick_params(colors=COLORS['steel'], labelsize=8)
        ax.xaxis.label.set_color(COLORS['steel'])
        ax.yaxis.label.set_color(COLORS['steel'])
        ax.title.set_color(COLORS['cream'])
        ax.grid(axis='y', color=COLORS['mid'], alpha=0.2, linewidth=0.5)

    # Tab 1 — Table
    with tab1:
        df = pd.DataFrame([{
            'Nama': r['nama'], 'NIM': r['nim'], 'Tanggal': r['tanggal'],
            'Benar': r['correct'], 'Salah': r['wrong'], 'Kosong': r['unanswered'],
            'Skor': r['score'],
            'Grade': 'A' if r['score']>=80 else 'B' if r['score']>=70 else 'C' if r['score']>=60 else 'D' if r['score']>=50 else 'E'
        } for r in records])
        st.dataframe(df, use_container_width=True, height=400)

    # Tab 2 — Charts
    with tab2:
        col_ch1, col_ch2 = st.columns(2, gap="large")
        with col_ch1:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#111844')
            style_axes(ax)
            n, bins, patches = ax.hist(scores, bins=range(0, 105, 10),
                                        edgecolor='#111844', linewidth=0.8)
            for patch in patches:
                patch.set_facecolor(COLORS['mid'])
                patch.set_alpha(0.85)
            ax.axvline(np.mean(scores), color=COLORS['amber'], lw=1.5,
                       linestyle='--', label=f'Mean = {np.mean(scores):.1f}')
            ax.set_xlabel("Nilai"); ax.set_ylabel("Jumlah")
            ax.set_title("Distribusi Nilai", fontsize=11, fontweight='bold', pad=10)
            ax.legend(facecolor='#0C1235', labelcolor=COLORS['cream'],
                      fontsize=8, framealpha=0.8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close(fig)

        with col_ch2:
            grades = {'A (≥80)':0,'B (70-79)':0,'C (60-69)':0,'D (50-59)':0,'E (<50)':0}
            for s in scores:
                if s>=80: grades['A (≥80)']+=1
                elif s>=70: grades['B (70-79)']+=1
                elif s>=60: grades['C (60-69)']+=1
                elif s>=50: grades['D (50-59)']+=1
                else: grades['E (<50)']+=1
            lp = [k for k,v in grades.items() if v>0]
            sp = [v for v in grades.values() if v>0]
            pie_colors = [COLORS['green'],COLORS['blue'],COLORS['amber'],COLORS['red'],'#9B7E7E'][:len(lp)]
            fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#111844')
            ax2.set_facecolor('#111844')
            wedges, texts, autotexts = ax2.pie(
                sp, labels=lp, colors=pie_colors,
                autopct='%1.0f%%', startangle=90,
                textprops={'color': COLORS['cream'], 'fontsize': 8},
                wedgeprops={'linewidth': 2, 'edgecolor': '#111844'}
            )
            for at in autotexts: at.set_color('#111844'); at.set_fontweight('bold')
            ax2.set_title("Distribusi Grade", color=COLORS['cream'], fontsize=11,
                          fontweight='bold', pad=10)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig2); plt.close(fig2)

        if len(records) > 1:
            st.markdown('<div class="section-label" style="margin-top:1rem">Tingkat Kesulitan per Soal</div>', unsafe_allow_html=True)
            rates = [sum(1 for r in records if r['student_answers'].get(str(q)) == r['answer_key'].get(q))
                     / len(records) * 100 for q in range(1, total_soal+1)]
            bar_colors = [COLORS['green'] if v>=70 else COLORS['amber'] if v>=40 else COLORS['red'] for v in rates]
            fig3, ax3 = plt.subplots(figsize=(14, 3), facecolor='#111844')
            style_axes(ax3)
            ax3.bar(range(1, total_soal+1), rates, color=bar_colors, width=0.7, edgecolor='none')
            ax3.set_xlabel("Nomor Soal"); ax3.set_ylabel("% Benar")
            ax3.set_title("Persentase Benar per Soal", fontsize=11, fontweight='bold', pad=8)
            ax3.set_ylim(0, 108)
            ax3.axhline(70, color=COLORS['green'], lw=0.8, linestyle=':', alpha=0.5)
            ax3.axhline(40, color=COLORS['amber'], lw=0.8, linestyle=':', alpha=0.5)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig3); plt.close(fig3)

            # Legend
            st.markdown("""
            <div style="display:flex;gap:16px;margin-top:4px">
              <span style="font-size:0.72rem;color:#6DBF9E">■ Mudah (≥70%)</span>
              <span style="font-size:0.72rem;color:#D4A96A">■ Sedang (40-69%)</span>
              <span style="font-size:0.72rem;color:#E07575">■ Sulit (&lt;40%)</span>
            </div>
            """, unsafe_allow_html=True)

    # Tab 3 — Export
    with tab3:
        st.markdown('<div class="section-label">Download Hasil</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        df_rekap = pd.DataFrame([{
            'Nama': r['nama'], 'NIM': r['nim'], 'Tanggal': r['tanggal'],
            'Benar': r['correct'], 'Salah': r['wrong'], 'Kosong': r['unanswered'],
            'Skor': r['score'],
            'Grade': 'A' if r['score']>=80 else 'B' if r['score']>=70 else 'C' if r['score']>=60 else 'D' if r['score']>=50 else 'E'
        } for r in records])

        rows_detail = []
        for r in records:
            row = {'NIM': r['nim'], 'Nama': r['nama'], 'Skor': r['score']}
            for q in range(1, total_soal+1):
                row[f'Q{q}'] = r['student_answers'].get(str(q), '-')
                row[f'Q{q}_kunci'] = r['answer_key'].get(q, '?')
            rows_detail.append(row)
        df_detail = pd.DataFrame(rows_detail)

        df_stats = pd.DataFrame([{'Metrik': m, 'Nilai': v} for m, v in {
            'Total Mahasiswa': len(records),
            'Rata-rata': round(np.mean(scores), 2),
            'Tertinggi': max(scores),
            'Terendah': min(scores),
            'Std Deviasi': round(np.std(scores), 2),
            'Varians': round(np.var(scores), 2),
        }.items()])

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_rekap.to_excel(writer, sheet_name='Rekap Nilai', index=False)
            df_detail.to_excel(writer, sheet_name='Detail Jawaban', index=False)
            df_stats.to_excel(writer, sheet_name='Statistik Kelas', index=False)
        buf.seek(0)

        fname = f"hasil_{st.session_state.kode_kelas}_{st.session_state.sesi_nama.replace(' ','_')}.xlsx"
        st.download_button("⬇  Download Excel (3 Sheet)", data=buf, file_name=fname,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("<br>", unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.download_button("⬇  CSV Rekap Nilai",
                               df_rekap.to_csv(index=False).encode(),
                               file_name=f"rekap_{st.session_state.kode_kelas}.csv",
                               mime="text/csv")
        with cc2:
            st.download_button("⬇  CSV Detail Jawaban",
                               df_detail.to_csv(index=False).encode(),
                               file_name=f"detail_{st.session_state.kode_kelas}.csv",
                               mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  STEP 4 — ML
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 'ml':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    records = st.session_state.records

    st.markdown('<div class="section-label">Langkah 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Model <span>Machine Learning</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Score prediction menggunakan Linear Regression & Support Vector Regression.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("← Hasil"): st.session_state.step = 'results'; st.rerun()

    if not records:
        st.warning("Belum ada data scan."); st.stop()
    if len(records) < 3:
        st.info(f"Model ML butuh minimal **3 data**. Saat ini ada **{len(records)}** data."); st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # Train
    if st.session_state.ml_results is None or st.button("↺  Re-train Model"):
        with st.spinner("Melatih model..."):
            # features
            scores_arr = [r['score'] for r in records]
            class_mean = np.mean(scores_arr); class_var = np.var(scores_arr); class_std = np.std(scores_arr)
            total = records[0]['total_soal']
            n_students = len(records)
            correct_counts = np.zeros(total)
            for r in records:
                for q, ans in r['student_answers'].items():
                    if ans == r['answer_key'].get(int(q)):
                        idx = int(q)-1
                        if 0 <= idx < total: correct_counts[idx] += 1
            avg_diff = float(np.mean(correct_counts / max(n_students, 1)))
            feats = [[r['correct'], r['wrong'], r['unanswered'],
                      class_mean, class_var, class_std, avg_diff] for r in records]
            X = np.array(feats); y = np.array(scores_arr)
            scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
            lr = LinearRegression(); lr.fit(X_sc, y); y_lr = lr.predict(X_sc)
            svr = SVR(kernel='rbf', C=10, epsilon=0.5); svr.fit(X_sc, y); y_svr = svr.predict(X_sc)
            def met(pred):
                return {'MAE': round(mean_absolute_error(y, pred), 3),
                        'RMSE': round(np.sqrt(mean_squared_error(y, pred)), 3),
                        'R2': round(r2_score(y, pred) if len(y)>1 else 0, 3)}
            st.session_state.ml_results = {
                'actuals': y.tolist(), 'pred_lr': y_lr.tolist(), 'pred_svr': y_svr.tolist(),
                'metrics_lr': met(y_lr), 'metrics_svr': met(y_svr),
                'lr_coef': lr.coef_.tolist(),
                'feature_names': ['n_correct','n_wrong','n_unanswered','class_mean','class_var','class_std','avg_difficulty']
            }

    ml = st.session_state.ml_results
    COLORS = {'navy':'#111844','mid':'#4B5694','steel':'#7288AE','cream':'#EAE0CF',
              'green':'#6DBF9E','red':'#E07575','blue':'#7CA4D4','amber':'#D4A96A'}
    THR = {'MAE': 5.0, 'RMSE': 7.0, 'R2': 0.80}

    # Metrics comparison
    st.markdown('<div class="section-label">Evaluasi Model</div>', unsafe_allow_html=True)
    col_lr, col_svr = st.columns(2, gap="large")

    for col, name, met_key, pred_key, color in [
        (col_lr,  "Linear Regression",        'metrics_lr',  'pred_lr',  COLORS['blue']),
        (col_svr, "Support Vector Regression", 'metrics_svr', 'pred_svr', COLORS['amber']),
    ]:
        with col:
            st.markdown(f"""
            <div style="border-left:3px solid {color};padding-left:12px;margin-bottom:12px">
              <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:{color}">{name}</div>
            </div>
            """, unsafe_allow_html=True)
            for metric, val in ml[met_key].items():
                ok = val >= THR[metric] if metric == 'R2' else val <= THR[metric]
                val_color = COLORS['green'] if ok else COLORS['amber']
                icon = "✓" if ok else "⚠"
                thr_txt = f"{'≥' if metric=='R2' else '≤'}{THR[metric]}"
                st.markdown(f"""
                <div class="ml-metric">
                  <div>
                    <div class="m-name">{metric}</div>
                    <div style="font-size:0.7rem;color:#4B5694">{icon} Target {thr_txt}</div>
                  </div>
                  <div class="m-val" style="color:{val_color}">{val}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Scatter plots
    st.markdown('<div class="section-label">Aktual vs Prediksi</div>', unsafe_allow_html=True)

    def style_axes(ax):
        ax.set_facecolor('#0C1235')
        for sp in ax.spines.values(): sp.set_color(COLORS['mid'])
        ax.tick_params(colors=COLORS['steel'], labelsize=8)
        ax.xaxis.label.set_color(COLORS['steel'])
        ax.yaxis.label.set_color(COLORS['steel'])
        ax.title.set_color(COLORS['cream'])
        ax.grid(color=COLORS['mid'], alpha=0.15, linewidth=0.5)

    names = [r['nim'] or f"S{i+1}" for i, r in enumerate(records)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#111844')
    for ax, preds, title, c in [
        (axes[0], ml['pred_lr'],  "Linear Regression",        COLORS['blue']),
        (axes[1], ml['pred_svr'], "Support Vector Regression", COLORS['amber']),
    ]:
        style_axes(ax)
        actuals = ml['actuals']
        ax.scatter(actuals, preds, color=c, s=80, alpha=0.85, zorder=3,
                   edgecolors='white', linewidths=0.4)
        mv = min(min(actuals), min(preds)); xv = max(max(actuals), max(preds))
        ax.plot([mv,xv],[mv,xv], color='white', lw=1, alpha=0.25, linestyle='--')
        for i,(a,p) in enumerate(zip(actuals,preds)):
            ax.annotate(names[i],(a,p), textcoords='offset points', xytext=(5,3),
                        fontsize=6.5, color=COLORS['steel'])
        ax.set_xlabel("Nilai Aktual"); ax.set_ylabel("Nilai Prediksi")
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close(fig)

    # Table
    st.markdown('<div class="section-label" style="margin-top:1.5rem">Tabel Prediksi</div>', unsafe_allow_html=True)
    df_ml = pd.DataFrame({
        'Nama':       [r['nama'] for r in records],
        'NIM':        [r['nim']  for r in records],
        'Aktual':     ml['actuals'],
        'Pred LR':    [round(p,1) for p in ml['pred_lr']],
        'Pred SVR':   [round(p,1) for p in ml['pred_svr']],
        'Δ LR':       [round(abs(a-p),1) for a,p in zip(ml['actuals'],ml['pred_lr'])],
        'Δ SVR':      [round(abs(a-p),1) for a,p in zip(ml['actuals'],ml['pred_svr'])],
    })
    st.dataframe(df_ml, use_container_width=True)

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    buf_ml = io.BytesIO()
    with pd.ExcelWriter(buf_ml, engine='openpyxl') as w:
        df_ml.to_excel(w, sheet_name='Prediksi ML', index=False)
        pd.DataFrame([{'Model':'LR','MAE':ml['metrics_lr']['MAE'],'RMSE':ml['metrics_lr']['RMSE'],'R2':ml['metrics_lr']['R2']},
                      {'Model':'SVR','MAE':ml['metrics_svr']['MAE'],'RMSE':ml['metrics_svr']['RMSE'],'R2':ml['metrics_svr']['R2']}]
                    ).to_excel(w, sheet_name='Metrik', index=False)
    buf_ml.seek(0)
    st.download_button("⬇  Download Hasil ML (Excel)", data=buf_ml,
                       file_name=f"ml_{st.session_state.kode_kelas}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
