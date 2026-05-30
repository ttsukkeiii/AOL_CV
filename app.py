import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="LJK Scanner — CV Project",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  .metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    color: #f1f5f9;
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; color: #38bdf8; }
  .metric-card .lbl { font-size: .8rem; color: #94a3b8; margin-top: 2px; }
  .correct   { color: #22c55e; font-weight: 700; }
  .wrong     { color: #ef4444; font-weight: 700; }
  .unanswered{ color: #94a3b8; }
  .stButton>button {
    background: #0ea5e9;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
  }
  .stButton>button:hover { background: #0284c7; }
  div[data-testid="stExpander"] summary { font-weight: 600; }
  .answer-grid {
    display: grid;
    grid-template-columns: repeat(10, 1fr);
    gap: 4px;
    font-size: .78rem;
  }
  .answer-cell {
    border-radius: 6px;
    padding: 4px 2px;
    text-align: center;
    background: #1e293b;
    border: 1px solid #334155;
  }
</style>
""", unsafe_allow_html=True)

# ─── CV HELPERS (ported from notebook) ──────────────────────

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def get_bubble_range(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rp = np.sum(bw, axis=1) / 255
    w = gray.shape[1]
    b0 = next((i for i, v in enumerate(rp) if v > w * 0.03), 0)
    b1 = next((i for i in range(len(rp)-1, 0, -1) if rp[i] > w * 0.03), gray.shape[0])
    return b0, b1

def scan_grid(gray, num_cols, num_rows, labels, z_thresh=1.2, z_gap=0.5, per_row=False):
    eq  = apply_clahe(gray)
    inv = cv2.bitwise_not(eq)
    h, w = gray.shape
    b0, b1 = get_bubble_range(gray)
    bh = b1 - b0
    row_h = bh / num_rows
    col_w = w  / num_cols
    raw = np.zeros((num_rows, num_cols), dtype=float)
    for r in range(num_rows):
        for c in range(num_cols):
            y0 = b0 + int(r * row_h) + 2
            y1b = b0 + int((r+1) * row_h) - 2
            x0 = int(c * col_w) + 2
            x1b = int((c+1) * col_w) - 2
            cell = inv[y0:y1b, x0:x1b]
            if cell.size == 0: continue
            ch, cw = cell.shape
            cx = cell[ch//4:3*ch//4, cw//4:3*cw//4]
            raw[r, c] = float(np.mean(cx)) if cx.size > 0 else 0.0
    density_map = np.zeros((num_rows, num_cols), dtype=float)
    results = []
    if not per_row:
        for c in range(num_cols):
            arr = raw[:, c]
            mean, std = arr.mean(), arr.std() + 1e-6
            z = (arr - mean) / std
            density_map[:, c] = z
            best = int(np.argmax(z))
            bz = z[best]; sz = sorted(z)[-2]
            if bz > z_thresh and (bz - sz) > z_gap and best < len(labels):
                results.append(labels[best])
            else:
                results.append(None)
    else:
        for r in range(num_rows):
            arr = raw[r, :]
            mean, std = arr.mean(), arr.std() + 1e-6
            z = (arr - mean) / std
            density_map[r, :] = z
            best = int(np.argmax(z))
            bz = z[best]; sz = sorted(z)[-2]
            if bz > z_thresh and (bz - sz) > z_gap and best < len(labels):
                results.append(labels[best])
            else:
                results.append(None)
    return results, density_map, b0, b1, row_h, col_w

def warp_ljk(img):
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0: continue
        solidity = area / float(w * h)
        aspect = w / float(h)
        min_a = img_w * img_h * 0.0003
        max_a = img_w * img_h * 0.04
        if min_a < area < max_a and 0.4 < aspect < 2.5 and solidity >= 0.65:
            candidates.append({'area': area, 'x': x, 'y': y, 'w': w, 'h': h,
                                'cx': x + w//2, 'cy': y + h//2})
    corners_def = {'TL': (0, 0), 'TR': (img_w, 0), 'BR': (img_w, img_h), 'BL': (0, img_h)}
    selected = {}
    for label, (tx, ty) in corners_def.items():
        if not candidates: break
        best = min(candidates, key=lambda a: (a['cx']-tx)**2 + (a['cy']-ty)**2)
        selected[label] = best
    if len(selected) < 4:
        return None, False
    pts = np.array([[selected['TL']['cx'], selected['TL']['cy']],
                    [selected['TR']['cx'], selected['TR']['cy']],
                    [selected['BR']['cx'], selected['BR']['cy']],
                    [selected['BL']['cx'], selected['BL']['cy']]], dtype='float32')
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype='float32')
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    maxW = max(int(np.linalg.norm(rect[2]-rect[3])), int(np.linalg.norm(rect[1]-rect[0])))
    maxH = max(int(np.linalg.norm(rect[1]-rect[2])), int(np.linalg.norm(rect[0]-rect[3])))
    dst = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))
    warped = cv2.resize(warped, (1000, 1414))
    return warped, True

def detect_nama(warped):
    ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    X1, Y1, X2, Y2 = 40, 270, 520, 850
    roi = cv2.cvtColor(warped[Y1:Y2, X1:X2], cv2.COLOR_BGR2GRAY)
    results, *_ = scan_grid(roi, num_cols=20, num_rows=26, labels=ALPHABET, per_row=False)
    nama = ''.join(r or '_' for r in results).replace('_', ' ').strip()
    return nama

def detect_nim(warped):
    x1, y1, x2, y2 = 550, 270, 790, 500
    roi = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    results, *_ = scan_grid(roi, num_cols=10, num_rows=10,
                             labels=[str(i) for i in range(10)], per_row=False)
    return ''.join(r or '_' for r in results)

def detect_tanggal(warped):
    x1, y1, x2, y2 = 820, 270, 970, 500
    roi = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    results, *_ = scan_grid(roi, num_cols=6, num_rows=10,
                             labels=[str(i) for i in range(10)], per_row=False)
    raw = ''.join(r or '_' for r in results)
    if len(raw) >= 6:
        return f"{raw[0:2]}/{raw[2:4]}/{raw[4:6]}"
    return raw

def detect_answers(warped, total_soal=100):
    CHOICES = ['A', 'B', 'C', 'D', 'E']
    ROI_JAWABAN = [
        ('1-10',    70,  930, 190, 1150,  1),
        ('11-20',   70, 1160, 190, 1390, 11),
        ('21-30',  250,  930, 380, 1150, 21),
        ('31-40',  259, 1160, 380, 1390, 31),
        ('41-50',  450,  930, 580, 1150, 41),
        ('51-60',  450, 1160, 580, 1390, 51),
        ('61-70',  650,  930, 780, 1150, 61),
        ('71-80',  650, 1160, 780, 1390, 71),
        ('81-90',  830,  930, 950, 1150, 81),
        ('91-100', 830, 1160, 950, 1390, 91),
    ]
    all_answers = {}
    soal_done = 0
    for label, x1, y1, x2, y2, q_start in ROI_JAWABAN:
        if soal_done >= total_soal:
            break
        soal_di_blok = min(10, total_soal - soal_done)
        roi_gray = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        r_blok, *_ = scan_grid(roi_gray, num_cols=5, num_rows=10,
                                labels=CHOICES, per_row=True)
        for i, ans in enumerate(r_blok[:soal_di_blok]):
            all_answers[q_start + i] = ans
        soal_done += soal_di_blok
    return all_answers

def score_answers(student_answers, answer_key):
    correct = wrong = unanswered = 0
    for q, key in answer_key.items():
        student = student_answers.get(q)
        if student is None:
            unanswered += 1
        elif student == key:
            correct += 1
        else:
            wrong += 1
    return correct, wrong, unanswered

def compute_score(correct, total_soal, scoring_method="standard"):
    if scoring_method == "standard":
        return round((correct / total_soal) * 100, 2) if total_soal > 0 else 0
    elif scoring_method == "penalty":
        wrong = total_soal - correct
        raw = correct - (wrong * 0.25)
        return round(max(0, raw / total_soal * 100), 2)
    return 0

# ─── ML HELPERS ─────────────────────────────────────────────

def extract_features(record, class_data):
    """Extract feature vector for ML model."""
    n_correct = record['correct']
    n_wrong   = record['wrong']
    n_unans   = record['unanswered']
    total     = record['total_soal']

    if class_data:
        scores = [r['score'] for r in class_data]
        class_mean = np.mean(scores)
        class_var  = np.var(scores)
        class_std  = np.std(scores)
        # Difficulty distribution: proportion of students who got each question right
        n_students = len(class_data)
        correct_counts = np.zeros(total)
        for r in class_data:
            for q, ans in r['student_answers'].items():
                if ans == r['answer_key'].get(q):
                    idx = int(q) - 1
                    if 0 <= idx < total:
                        correct_counts[idx] += 1
        difficulty = correct_counts / max(n_students, 1)  # fraction correct per question
        avg_difficulty = float(np.mean(difficulty))
    else:
        class_mean = n_correct / total * 100 if total > 0 else 0
        class_var  = 0
        class_std  = 0
        avg_difficulty = 0.5

    return [n_correct, n_wrong, n_unans, class_mean, class_var, class_std, avg_difficulty]

def train_and_predict(records, answer_key, total_soal):
    """Train LR & SVR, return predictions and metrics."""
    if len(records) < 2:
        return None

    features = []
    actuals  = []

    for rec in records:
        feat = extract_features(rec, records)
        features.append(feat)
        actuals.append(rec['score'])

    X = np.array(features)
    y = np.array(actuals)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # ── Linear Regression ──
    lr = LinearRegression()
    lr.fit(X_sc, y)
    y_pred_lr = lr.predict(X_sc)

    # ── SVR ──
    svr = SVR(kernel='rbf', C=10, epsilon=0.5)
    svr.fit(X_sc, y)
    y_pred_svr = svr.predict(X_sc)

    def metrics(pred):
        mae  = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        r2   = r2_score(y, pred) if len(y) > 1 else 0.0
        return {'MAE': round(mae, 3), 'RMSE': round(rmse, 3), 'R2': round(r2, 3)}

    return {
        'actuals':       y.tolist(),
        'pred_lr':       y_pred_lr.tolist(),
        'pred_svr':      y_pred_svr.tolist(),
        'metrics_lr':    metrics(y_pred_lr),
        'metrics_svr':   metrics(y_pred_svr),
        'lr_model':      lr,
        'svr_model':     svr,
        'scaler':        scaler,
        'feature_names': ['n_correct','n_wrong','n_unanswered',
                          'class_mean','class_var','class_std','avg_difficulty'],
    }

# ─── SESSION STATE ───────────────────────────────────────────
def init_state():
    defaults = {
        'answer_key':   {},
        'total_soal':   20,
        'sesi_nama':    '',
        'kode_kelas':   '',
        'kode_dosen':   '',
        'scoring':      'standard',
        'records':      [],          # list of per-student dicts
        'ml_results':   None,
        'step':         'setup',     # setup | scan | results | ml
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📝 LJK Scanner")
    st.markdown("**Computer Vision Project**")
    st.divider()

    step = st.session_state.step
    steps = ['setup', 'scan', 'results', 'ml']
    labels_sidebar = ['⚙️ Setup Sesi', '📸 Scan LJK', '📊 Hasil', '🤖 Model ML']
    for i, (s, lbl) in enumerate(zip(steps, labels_sidebar)):
        active = "→ " if s == step else "   "
        st.markdown(f"`{active}`{lbl}")

    st.divider()
    if st.session_state.records:
        st.markdown(f"**{len(st.session_state.records)}** lembar ter-scan")
        scores = [r['score'] for r in st.session_state.records]
        st.markdown(f"Rata-rata: **{np.mean(scores):.1f}**")
        st.markdown(f"Tertinggi: **{max(scores):.1f}**")
        st.markdown(f"Terendah: **{min(scores):.1f}**")
    st.divider()
    if st.button("🔄 Reset Sesi"):
        for k in ['answer_key','records','ml_results','step','sesi_nama','kode_kelas','kode_dosen']:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# ─── MAIN ────────────────────────────────────────────────────

# ══════════════════════════════════════════
#  STEP 1 — SETUP SESI
# ══════════════════════════════════════════
if st.session_state.step == 'setup':
    st.title("⚙️ Setup Sesi Ujian")
    st.markdown("Konfigurasikan parameter ujian dan masukkan kunci jawaban sebelum memulai scan.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informasi Sesi")
        st.session_state.sesi_nama  = st.text_input("Nama Sesi / Mata Kuliah", value=st.session_state.sesi_nama or "Computer Vision UAS")
        st.session_state.kode_kelas = st.text_input("Kode Kelas", value=st.session_state.kode_kelas or "LK01")
        st.session_state.kode_dosen = st.text_input("Kode Dosen", value=st.session_state.kode_dosen or "DS123")
        st.session_state.total_soal = st.number_input("Jumlah Soal (1–100)", min_value=1, max_value=100,
                                                        value=st.session_state.total_soal)
        st.session_state.scoring    = st.selectbox("Metode Penilaian",
                                                    ["standard", "penalty"],
                                                    format_func=lambda x: "Standar (benar/total × 100)" if x == "standard"
                                                                          else "Penalty (-0.25 per salah)")

    with col2:
        st.subheader("Kunci Jawaban")
        total = st.session_state.total_soal
        choices = ['A', 'B', 'C', 'D', 'E']

        st.markdown("**Input kunci jawaban (satu baris per soal: nomor,jawaban)**")
        default_key_text = "\n".join(f"{i},A" for i in range(1, total+1))
        if 'key_text' not in st.session_state:
            st.session_state.key_text = default_key_text

        key_text = st.text_area("Format: `1,A` `2,B` dll.", value=st.session_state.key_text,
                                 height=300, label_visibility="collapsed")
        st.session_state.key_text = key_text

        # Parse
        answer_key = {}
        errors = []
        for line in key_text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) != 2:
                errors.append(f"Format salah: `{line}`")
                continue
            try:
                q = int(parts[0].strip())
                ans = parts[1].strip().upper()
                if ans not in choices:
                    errors.append(f"Jawaban tidak valid di soal {q}: `{ans}`")
                    continue
                answer_key[q] = ans
            except ValueError:
                errors.append(f"Nomor soal tidak valid: `{line}`")

        if errors:
            for e in errors[:3]:
                st.error(e)
        else:
            st.success(f"✅ {len(answer_key)} kunci jawaban valid")
            st.session_state.answer_key = answer_key

    st.divider()
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("▶ Mulai Scan", disabled=len(answer_key) == 0):
            st.session_state.answer_key = answer_key
            st.session_state.step = 'scan'
            st.rerun()

# ══════════════════════════════════════════
#  STEP 2 — SCAN LJK
# ══════════════════════════════════════════
elif st.session_state.step == 'scan':
    st.title("📸 Scan Lembar Jawaban (LJK)")

    col_nav1, col_nav2 = st.columns([1, 5])
    with col_nav1:
        if st.button("← Setup"):
            st.session_state.step = 'setup'
            st.rerun()

    st.info(f"**Sesi:** {st.session_state.sesi_nama} | **Kelas:** {st.session_state.kode_kelas} | "
            f"**Soal:** {st.session_state.total_soal} | **Ter-scan:** {len(st.session_state.records)} lembar")

    uploaded_files = st.file_uploader(
        "Upload foto LJK (JPG / PNG) — bisa lebih dari satu",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="ljk_upload"
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            already = any(r['filename'] == uploaded.name for r in st.session_state.records)
            if already:
                st.warning(f"⚠️ `{uploaded.name}` sudah di-scan sebelumnya, dilewati.")
                continue

            with st.expander(f"🔍 Memproses: `{uploaded.name}`", expanded=True):
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                img_orig   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img_orig is None:
                    st.error("❌ Gagal membaca gambar.")
                    continue

                # ── Warp ──
                warped, ok = warp_ljk(img_orig)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB),
                             caption="Input asli", use_container_width=True)
                with col_b:
                    if ok:
                        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB),
                                 caption="Setelah warp/perspective correction", use_container_width=True)
                    else:
                        st.error("❌ Gagal mendeteksi 4 sudut LJK. Pastikan foto jelas dan sudut terlihat.")
                        continue

                # ── Detect fields ──
                with st.spinner("Mendeteksi data..."):
                    nama    = detect_nama(warped)
                    nim     = detect_nim(warped)
                    tanggal = detect_tanggal(warped)
                    answers = detect_answers(warped, st.session_state.total_soal)

                correct, wrong, unanswered = score_answers(answers, st.session_state.answer_key)
                score = compute_score(correct, st.session_state.total_soal, st.session_state.scoring)

                # ── Show extracted info ──
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.markdown(f'<div class="metric-card"><div class="val">{score}</div><div class="lbl">Skor</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><div class="val" style="color:#22c55e">{correct}</div><div class="lbl">Benar</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-card"><div class="val" style="color:#ef4444">{wrong}</div><div class="lbl">Salah</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="metric-card"><div class="val" style="color:#94a3b8">{unanswered}</div><div class="lbl">Kosong</div></div>', unsafe_allow_html=True)
                with m5:
                    st.markdown(f'<div class="metric-card"><div class="val" style="font-size:1rem;color:#f59e0b">{nim}</div><div class="lbl">NIM</div></div>', unsafe_allow_html=True)

                st.markdown(f"**Nama:** {nama}  |  **Tanggal:** {tanggal}")

                # ── Answer grid ──
                with st.expander("📋 Detail Jawaban per Soal"):
                    total_soal = st.session_state.total_soal
                    key = st.session_state.answer_key
                    cols_grid = st.columns(10)
                    for q in range(1, total_soal + 1):
                        student_ans = answers.get(q)
                        correct_ans = key.get(q, '?')
                        col_idx = (q - 1) % 10
                        with cols_grid[col_idx]:
                            if student_ans is None:
                                label = f"**{q}**: –"
                                color = "#374151"
                            elif student_ans == correct_ans:
                                label = f"**{q}**: {student_ans} ✓"
                                color = "#14532d"
                            else:
                                label = f"**{q}**: ~~{correct_ans}~~ {student_ans}"
                                color = "#450a0a"
                            st.markdown(
                                f'<div style="background:{color};border-radius:4px;padding:3px 5px;'
                                f'font-size:0.7rem;margin-bottom:4px;text-align:center">{label}</div>',
                                unsafe_allow_html=True)

                # ── Save record ──
                record = {
                    'filename':       uploaded.name,
                    'nama':           nama,
                    'nim':            nim,
                    'tanggal':        tanggal,
                    'correct':        correct,
                    'wrong':          wrong,
                    'unanswered':     unanswered,
                    'score':          score,
                    'total_soal':     st.session_state.total_soal,
                    'student_answers': {str(k): v for k, v in answers.items()},
                    'answer_key':      st.session_state.answer_key,
                }
                st.session_state.records.append(record)
                st.session_state.ml_results = None  # reset ML on new data
                st.success(f"✅ `{uploaded.name}` berhasil di-scan dan disimpan.")

    st.divider()
    col_nav_a, col_nav_b = st.columns([1, 1])
    with col_nav_a:
        if st.button("📊 Lihat Hasil", disabled=len(st.session_state.records) == 0):
            st.session_state.step = 'results'
            st.rerun()

# ══════════════════════════════════════════
#  STEP 3 — RESULTS
# ══════════════════════════════════════════
elif st.session_state.step == 'results':
    st.title("📊 Hasil & Analitik Kelas")

    records = st.session_state.records
    if not records:
        st.warning("Belum ada data. Silakan scan LJK terlebih dahulu.")
        if st.button("← Kembali ke Scan"):
            st.session_state.step = 'scan'
            st.rerun()
        st.stop()

    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
    with col_nav1:
        if st.button("← Scan Lagi"):
            st.session_state.step = 'scan'
            st.rerun()
    with col_nav2:
        if st.button("🤖 Analisis ML"):
            st.session_state.step = 'ml'
            st.rerun()

    # ── Summary stats ──
    scores = [r['score'] for r in records]
    corrects = [r['correct'] for r in records]
    total_soal = records[0]['total_soal']

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label, color in [
        (c1, len(records), "Total Mahasiswa", "#38bdf8"),
        (c2, f"{np.mean(scores):.1f}", "Rata-rata Nilai", "#a78bfa"),
        (c3, f"{max(scores):.1f}", "Nilai Tertinggi", "#22c55e"),
        (c4, f"{min(scores):.1f}", "Nilai Terendah", "#ef4444"),
        (c5, f"{np.std(scores):.2f}", "Std Deviasi", "#f59e0b"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="val" style="color:{color}">{val}</div><div class="lbl">{label}</div></div>',
                     unsafe_allow_html=True)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📋 Tabel Nilai", "📈 Distribusi & Grafik", "💾 Export"])

    # ── TAB 1 — Table ──
    with tab1:
        df = pd.DataFrame([{
            'Nama':       r['nama'],
            'NIM':        r['nim'],
            'Tanggal':    r['tanggal'],
            'Benar':      r['correct'],
            'Salah':      r['wrong'],
            'Kosong':     r['unanswered'],
            'Skor':       r['score'],
            'File':       r['filename'],
        } for r in records])

        def highlight_score(val):
            if val >= 80: return 'background-color:#14532d;color:#bbf7d0'
            elif val >= 60: return 'background-color:#713f12;color:#fde68a'
            else: return 'background-color:#450a0a;color:#fecaca'

        styled = df.style.applymap(highlight_score, subset=['Skor'])
        st.dataframe(styled, use_container_width=True, height=400)

    # ── TAB 2 — Charts ──
    with tab2:
        col_ch1, col_ch2 = st.columns(2)

        with col_ch1:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0f172a')
            ax.set_facecolor('#1e293b')
            bins_edges = list(range(0, 105, 10))
            ax.hist(scores, bins=bins_edges, color='#0ea5e9', edgecolor='#0f172a', alpha=0.9)
            ax.set_xlabel("Rentang Nilai", color='#e2e8f0')
            ax.set_ylabel("Jumlah Mahasiswa", color='#e2e8f0')
            ax.set_title("Distribusi Nilai", color='#f1f5f9', fontweight='bold')
            ax.axvline(np.mean(scores), color='#f59e0b', lw=2, linestyle='--', label=f'Mean={np.mean(scores):.1f}')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values(): spine.set_color('#334155')
            ax.legend(facecolor='#1e293b', labelcolor='#f1f5f9')
            st.pyplot(fig)

        with col_ch2:
            # Grade distribution pie
            grades = {'A (≥80)': 0, 'B (70-79)': 0, 'C (60-69)': 0, 'D (50-59)': 0, 'E (<50)': 0}
            for s in scores:
                if s >= 80: grades['A (≥80)'] += 1
                elif s >= 70: grades['B (70-79)'] += 1
                elif s >= 60: grades['C (60-69)'] += 1
                elif s >= 50: grades['D (50-59)'] += 1
                else: grades['E (<50)'] += 1

            labels_pie = [k for k, v in grades.items() if v > 0]
            sizes_pie  = [v for v in grades.values() if v > 0]
            colors_pie = ['#22c55e', '#84cc16', '#f59e0b', '#f97316', '#ef4444'][:len(labels_pie)]

            fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#0f172a')
            ax2.set_facecolor('#1e293b')
            wedges, texts, autotexts = ax2.pie(
                sizes_pie, labels=labels_pie, colors=colors_pie,
                autopct='%1.0f%%', startangle=90,
                textprops={'color': '#e2e8f0', 'fontsize': 9})
            for at in autotexts: at.set_color('#0f172a')
            ax2.set_title("Distribusi Grade", color='#f1f5f9', fontweight='bold')
            st.pyplot(fig2)

        # Per-question difficulty
        if len(records) > 1:
            st.subheader("Tingkat Kesulitan per Soal")
            correct_rates = []
            for q in range(1, total_soal + 1):
                n_correct_q = sum(
                    1 for r in records
                    if r['student_answers'].get(str(q)) == r['answer_key'].get(q)
                )
                correct_rates.append(n_correct_q / len(records) * 100)

            fig3, ax3 = plt.subplots(figsize=(14, 3), facecolor='#0f172a')
            ax3.set_facecolor('#1e293b')
            colors_bar = ['#22c55e' if v >= 70 else '#f59e0b' if v >= 40 else '#ef4444'
                          for v in correct_rates]
            ax3.bar(range(1, total_soal+1), correct_rates, color=colors_bar, width=0.7)
            ax3.set_xlabel("Nomor Soal", color='#e2e8f0')
            ax3.set_ylabel("% Benar", color='#e2e8f0')
            ax3.set_title("Persentase Jawaban Benar per Soal  |  🟢 Mudah  🟡 Sedang  🔴 Sulit",
                          color='#f1f5f9', fontweight='bold')
            ax3.tick_params(colors='#94a3b8')
            for spine in ax3.spines.values(): spine.set_color('#334155')
            ax3.set_ylim(0, 105)
            st.pyplot(fig3)

    # ── TAB 3 — Export ──
    with tab3:
        st.subheader("💾 Export Hasil ke CSV")

        # Sheet 1: Rekap Nilai
        df_rekap = pd.DataFrame([{
            'Nama': r['nama'], 'NIM': r['nim'], 'Tanggal': r['tanggal'],
            'Benar': r['correct'], 'Salah': r['wrong'], 'Kosong': r['unanswered'],
            'Skor': r['score'], 'File': r['filename'],
        } for r in records])

        # Sheet 2: Detail Jawaban
        rows_detail = []
        for r in records:
            row = {'NIM': r['nim'], 'Nama': r['nama'], 'Skor': r['score']}
            for q in range(1, total_soal + 1):
                student = r['student_answers'].get(str(q))
                key_ans = r['answer_key'].get(q, '?')
                row[f'Q{q}_jawaban']    = student or '-'
                row[f'Q{q}_kunci']      = key_ans
                row[f'Q{q}_benar']      = 'Y' if student == key_ans else 'N'
            rows_detail.append(row)
        df_detail = pd.DataFrame(rows_detail)

        # Sheet 3: Statistik Kelas
        df_stats = pd.DataFrame([{
            'Metrik': m, 'Nilai': v
        } for m, v in {
            'Total Mahasiswa': len(records),
            'Total Soal': total_soal,
            'Rata-rata Nilai': round(np.mean(scores), 2),
            'Nilai Tertinggi': max(scores),
            'Nilai Terendah': min(scores),
            'Std Deviasi': round(np.std(scores), 2),
            'Varians': round(np.var(scores), 2),
            'A (≥80)': sum(1 for s in scores if s >= 80),
            'B (70-79)': sum(1 for s in scores if 70 <= s < 80),
            'C (60-69)': sum(1 for s in scores if 60 <= s < 70),
            'D (50-59)': sum(1 for s in scores if 50 <= s < 60),
            'E (<50)':   sum(1 for s in scores if s < 50),
        }.items()])

        # Write to Excel-like multi-sheet (using CSV)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_rekap.to_excel(writer, sheet_name='Rekap Nilai', index=False)
            df_detail.to_excel(writer, sheet_name='Detail Jawaban', index=False)
            df_stats.to_excel(writer, sheet_name='Statistik Kelas', index=False)
        buf.seek(0)

        filename_export = f"hasil_{st.session_state.kode_kelas}_{st.session_state.sesi_nama.replace(' ','_')}.xlsx"
        st.download_button(
            label="⬇️ Download Excel (Multi-Sheet)",
            data=buf,
            file_name=filename_export,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            csv_rekap = df_rekap.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ CSV Rekap Nilai", csv_rekap,
                               file_name=f"rekap_{st.session_state.kode_kelas}.csv",
                               mime="text/csv")
        with col_e2:
            csv_detail = df_detail.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ CSV Detail Jawaban", csv_detail,
                               file_name=f"detail_{st.session_state.kode_kelas}.csv",
                               mime="text/csv")

# ══════════════════════════════════════════
#  STEP 4 — ML MODEL
# ══════════════════════════════════════════
elif st.session_state.step == 'ml':
    st.title("🤖 Model Machine Learning — Score Prediction")

    records = st.session_state.records
    if not records:
        st.warning("Belum ada data scan. Silakan scan LJK terlebih dahulu.")
        if st.button("← Scan LJK"):
            st.session_state.step = 'scan'
            st.rerun()
        st.stop()

    col_nav1, col_nav2 = st.columns([1, 5])
    with col_nav1:
        if st.button("← Hasil"):
            st.session_state.step = 'results'
            st.rerun()

    if len(records) < 3:
        st.info(f"Model ML membutuhkan minimal 3 data. Saat ini ada **{len(records)}** data.")
        st.stop()

    # ── Train ──
    if st.session_state.ml_results is None or st.button("🔄 Re-train Model"):
        with st.spinner("Melatih model Linear Regression & SVR..."):
            ml = train_and_predict(records, st.session_state.answer_key, st.session_state.total_soal)
            st.session_state.ml_results = ml

    ml = st.session_state.ml_results
    if ml is None:
        st.error("Training gagal. Pastikan data cukup.")
        st.stop()

    # ── Metrics comparison ──
    st.subheader("📏 Perbandingan Metrik Evaluasi")

    THRESHOLD = {'MAE': 5.0, 'RMSE': 7.0, 'R2': 0.80}

    def check(val, metric, reverse=False):
        thr = THRESHOLD[metric]
        if metric == 'R2':
            ok = val >= thr
        else:
            ok = val <= thr
        return "✅" if ok else "⚠️"

    col_m1, col_m2 = st.columns(2)

    for col, model_name, metrics in [(col_m1, "Linear Regression", ml['metrics_lr']),
                                      (col_m2, "Support Vector Regression", ml['metrics_svr'])]:
        with col:
            st.markdown(f"**{model_name}**")
            for metric, val in metrics.items():
                icon = check(val, metric)
                thr_label = f"(target {'≤' if metric != 'R2' else '≥'}{THRESHOLD[metric]})"
                if metric == 'R2':
                    color = '#22c55e' if val >= THRESHOLD[metric] else '#f59e0b'
                else:
                    color = '#22c55e' if val <= THRESHOLD[metric] else '#f59e0b'
                st.markdown(
                    f'<div class="metric-card" style="margin-bottom:8px">'
                    f'<div class="val" style="color:{color}">{val}</div>'
                    f'<div class="lbl">{icon} {metric} {thr_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Actual vs Predicted chart ──
    st.subheader("📈 Nilai Aktual vs Prediksi")

    actuals   = ml['actuals']
    pred_lr   = ml['pred_lr']
    pred_svr  = ml['pred_svr']
    names     = [r['nim'] or r['nama'][:8] or f"S{i+1}" for i, r in enumerate(records)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f172a')
    for ax, preds, title, color in [
        (axes[0], pred_lr,  "Linear Regression", '#38bdf8'),
        (axes[1], pred_svr, "SVR", '#a78bfa'),
    ]:
        ax.set_facecolor('#1e293b')
        ax.scatter(actuals, preds, color=color, alpha=0.8, s=80, zorder=3)
        min_v = min(min(actuals), min(preds))
        max_v = max(max(actuals), max(preds))
        ax.plot([min_v, max_v], [min_v, max_v], 'w--', lw=1.5, alpha=0.4, label='Perfect prediction')
        for i, (a, p) in enumerate(zip(actuals, preds)):
            ax.annotate(names[i], (a, p), textcoords='offset points', xytext=(5, 3),
                        fontsize=7, color='#94a3b8')
        ax.set_xlabel("Nilai Aktual", color='#e2e8f0')
        ax.set_ylabel("Nilai Prediksi", color='#e2e8f0')
        ax.set_title(title, color='#f1f5f9', fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values(): spine.set_color('#334155')
        ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    # ── Prediction table ──
    st.subheader("📋 Tabel Prediksi per Mahasiswa")
    df_ml = pd.DataFrame({
        'Nama':           [r['nama'] for r in records],
        'NIM':            [r['nim']  for r in records],
        'Nilai Aktual':   actuals,
        'Pred LR':        [round(p, 1) for p in pred_lr],
        'Pred SVR':       [round(p, 1) for p in pred_svr],
        'Selisih LR':     [round(abs(a-p), 1) for a, p in zip(actuals, pred_lr)],
        'Selisih SVR':    [round(abs(a-p), 1) for a, p in zip(actuals, pred_svr)],
    })

    def color_diff(val):
        if val <= 5: return 'color:#22c55e'
        elif val <= 10: return 'color:#f59e0b'
        return 'color:#ef4444'

    st.dataframe(
        df_ml.style.applymap(color_diff, subset=['Selisih LR', 'Selisih SVR']),
        use_container_width=True
    )

    # ── Feature importance (LR coefficients) ──
    with st.expander("🔬 Feature Importance (Linear Regression Coefficients)"):
        coef_df = pd.DataFrame({
            'Fitur':      ml['feature_names'],
            'Koefisien':  [round(c, 4) for c in ml['lr_model'].coef_],
        }).sort_values('Koefisien', key=abs, ascending=False)

        fig_coef, ax_coef = plt.subplots(figsize=(8, 3), facecolor='#0f172a')
        ax_coef.set_facecolor('#1e293b')
        colors_coef = ['#22c55e' if v > 0 else '#ef4444' for v in coef_df['Koefisien']]
        ax_coef.barh(coef_df['Fitur'], coef_df['Koefisien'], color=colors_coef)
        ax_coef.axvline(0, color='white', lw=0.8, alpha=0.5)
        ax_coef.set_xlabel("Koefisien", color='#e2e8f0')
        ax_coef.set_title("Pengaruh Fitur terhadap Prediksi Nilai", color='#f1f5f9', fontweight='bold')
        ax_coef.tick_params(colors='#94a3b8')
        for spine in ax_coef.spines.values(): spine.set_color('#334155')
        st.pyplot(fig_coef)

    # ── Export ML results ──
    st.subheader("💾 Export Hasil ML")
    buf_ml = io.BytesIO()
    with pd.ExcelWriter(buf_ml, engine='openpyxl') as writer:
        df_ml.to_excel(writer, sheet_name='Prediksi ML', index=False)
        metrics_df = pd.DataFrame({
            'Model': ['Linear Regression', 'SVR'],
            'MAE':   [ml['metrics_lr']['MAE'], ml['metrics_svr']['MAE']],
            'RMSE':  [ml['metrics_lr']['RMSE'], ml['metrics_svr']['RMSE']],
            'R2':    [ml['metrics_lr']['R2'], ml['metrics_svr']['R2']],
        })
        metrics_df.to_excel(writer, sheet_name='Metrik Evaluasi', index=False)
        coef_df.to_excel(writer, sheet_name='Feature Importance', index=False)
    buf_ml.seek(0)
    st.download_button(
        "⬇️ Download Hasil ML (Excel)",
        data=buf_ml,
        file_name=f"ml_results_{st.session_state.kode_kelas}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
