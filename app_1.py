import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import pandas as pd
import json
import io
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from skimage import filters, morphology, measure, transform, exposure
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings("ignore")

DEFAULT_50 = ['B','C','A','D','E','A','B','C','D','A','E','B','C','A','D','B','E','A','C','D','A','B','E','C','D','B','A','D','C','E','A','C','B','D','E','C','A','B','D','E','B','D','A','C','E','A','D','B','E','C']
def make_key_text(n):
    lines = []
    for i in range(1, n+1):
        ans = DEFAULT_50[i-1] if i <= 50 else 'A'
        lines.append(f"{i}. {ans}")
    return "\n".join(lines)


# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="LJK Scanner — CV Project",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  .metric-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 16px 20px;
    text-align: center; color: #f1f5f9;
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; color: #38bdf8; }
  .metric-card .lbl { font-size: .8rem; color: #94a3b8; margin-top: 2px; }
  .stButton>button {
    background: #0ea5e9; color: white; border: none;
    border-radius: 8px; font-weight: 600; padding: 8px 20px;
  }
  .stButton>button:hover { background: #0284c7; }
</style>
""", unsafe_allow_html=True)

# ─── IMAGE PROCESSING HELPERS (no cv2) ──────────────────────

def pil_to_np(img_pil):
    return np.array(img_pil)

def np_to_pil(arr):
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def to_gray_np(img_np):
    if img_np.ndim == 3:
        return (rgb2gray(img_np) * 255).astype(np.uint8)
    return img_np

def apply_clahe_np(gray_np):
    return (exposure.equalize_adapthist(gray_np / 255.0, clip_limit=0.02) * 255).astype(np.uint8)

def threshold_otsu_np(gray_np):
    thresh = filters.threshold_otsu(gray_np)
    return (gray_np < thresh).astype(np.uint8) * 255

def get_bubble_range(gray_np):
    bw = threshold_otsu_np(gray_np)
    row_proj = bw.sum(axis=1) / 255
    w = gray_np.shape[1]
    h = gray_np.shape[0]
    # Try progressively looser thresholds
    for frac in (0.03, 0.015, 0.005):
        b0 = next((i for i, v in enumerate(row_proj) if v > w * frac), None)
        b1 = next((i for i in range(len(row_proj)-1, 0, -1) if row_proj[i] > w * frac), None)
        if b0 is not None and b1 is not None and b1 > b0:
            return b0, b1
    return 0, h

def scan_grid(gray_np, num_cols, num_rows, labels, z_thresh=0.6, z_gap=0.3, per_row=False):
    eq  = apply_clahe_np(gray_np)
    inv = 255 - eq
    h, w = gray_np.shape
    b0, b1 = get_bubble_range(gray_np)
    bh = b1 - b0
    row_h = bh / num_rows
    col_w = w  / num_cols
    raw = np.zeros((num_rows, num_cols), dtype=float)
    for r in range(num_rows):
        for c in range(num_cols):
            y0  = b0 + int(r * row_h) + 2
            y1b = b0 + int((r+1) * row_h) - 2
            x0  = int(c * col_w) + 2
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
            results.append(labels[best] if bz > z_thresh and (bz - sz) > z_gap and best < len(labels) else None)
    else:
        for r in range(num_rows):
            arr = raw[r, :]
            mean, std = arr.mean(), arr.std() + 1e-6
            z = (arr - mean) / std
            density_map[r, :] = z
            best = int(np.argmax(z))
            bz = z[best]; sz = sorted(z)[-2]
            results.append(labels[best] if bz > z_thresh and (bz - sz) > z_gap and best < len(labels) else None)
    return results, density_map, b0, b1, row_h, col_w

def find_corner_squares(gray_np):
    """Detect 4 corner squares using skimage."""
    img_h, img_w = gray_np.shape
    thresh = (gray_np < 80).astype(np.uint8)
    # morphological open
    selem = morphology.square(3)
    cleaned = morphology.binary_opening(thresh, selem)
    labeled = measure.label(cleaned)
    props = measure.regionprops(labeled)

    candidates = []
    for p in props:
        area = p.area
        bb = p.bbox  # (min_row, min_col, max_row, max_col)
        h = bb[2] - bb[0]; w = bb[3] - bb[1]
        if w == 0 or h == 0: continue
        solidity = area / (w * h)
        aspect = w / h
        min_a = img_w * img_h * 0.0003
        max_a = img_w * img_h * 0.04
        if min_a < area < max_a and 0.4 < aspect < 2.5 and solidity >= 0.65:
            candidates.append({'cx': (bb[1]+bb[3])//2, 'cy': (bb[0]+bb[2])//2})

    if not candidates:
        return None

    corners = {'TL': (0,0), 'TR': (img_w,0), 'BR': (img_w,img_h), 'BL': (0,img_h)}
    selected = {}
    for label, (tx, ty) in corners.items():
        best = min(candidates, key=lambda a: (a['cx']-tx)**2 + (a['cy']-ty)**2)
        selected[label] = best
    return selected

def warp_ljk(img_pil):
    img_np = pil_to_np(img_pil.convert('RGB'))
    gray   = to_gray_np(img_np)
    selected = find_corner_squares(gray)
    if selected is None or len(selected) < 4:
        return None, False

    pts = np.array([
        [selected['TL']['cx'], selected['TL']['cy']],
        [selected['TR']['cx'], selected['TR']['cy']],
        [selected['BR']['cx'], selected['BR']['cy']],
        [selected['BL']['cx'], selected['BL']['cy']],
    ], dtype='float32')

    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype='float32')
    rect[0] = pts[np.argmin(s)];    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]

    maxW = int(max(np.linalg.norm(rect[2]-rect[3]), np.linalg.norm(rect[1]-rect[0])))
    maxH = int(max(np.linalg.norm(rect[1]-rect[2]), np.linalg.norm(rect[0]-rect[3])))

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype='float32')
    tform = transform.ProjectiveTransform()
    tform.estimate(dst, rect)
    warped_np = transform.warp(img_np, tform, output_shape=(maxH, maxW))
    warped_np = (warped_np * 255).astype(np.uint8)

    warped_pil = Image.fromarray(warped_np).resize((1000, 1414), Image.LANCZOS)
    warped_np  = pil_to_np(warped_pil)
    return warped_np, True

def crop_gray(warped_np, y1, y2, x1, x2):
    roi = warped_np[y1:y2, x1:x2]
    return to_gray_np(roi)

def detect_nama(warped_np):
    """
    LJK Sunib: NAMA MAHASISWA area
    Rows = A-Z (26 rows), Cols = 20 character positions
    Located roughly top-left of the form, below header
    """
    ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    # ROI: (y1, y2, x1, x2) after warp to 1000x1414
    roi = crop_gray(warped_np, 200, 720, 25, 490)
    results, *_ = scan_grid(roi, num_cols=20, num_rows=26, labels=ALPHABET, per_row=False)
    nama = ''.join(r or ' ' for r in results).strip()
    # Clean up multiple spaces
    import re
    nama = re.sub(r' +', ' ', nama)
    return nama or 'UNKNOWN'

def detect_nim(warped_np):
    """
    LJK Sunib: NOMOR INDUK MAHASISWA
    10 columns, rows = 0-9
    Located top-right area
    """
    roi = crop_gray(warped_np, 200, 530, 510, 760)
    results, *_ = scan_grid(roi, num_cols=10, num_rows=10,
                             labels=[str(i) for i in range(10)], per_row=False)
    return ''.join(r or '_' for r in results)

def detect_tanggal(warped_np):
    """
    LJK Sunib: TANGGAL — 6 columns (DD MM YY), rows = 0-9
    Located top-right corner
    """
    roi = crop_gray(warped_np, 200, 530, 780, 970)
    results, *_ = scan_grid(roi, num_cols=6, num_rows=10,
                             labels=[str(i) for i in range(10)], per_row=False)
    raw = ''.join(r or '_' for r in results)
    if len(raw) >= 6:
        return f"{raw[0:2]}/{raw[2:4]}/{raw[4:6]}"
    return raw

def detect_answers(warped_np, total_soal=100):
    """
    LJK Sunib JAWABAN layout:
    5 columns per answer block (A,B,C,D,E), read per ROW
    Blocks arranged as:
      Col1: Q1-10 (top), Q11-20 (bottom)
      Col2: Q21-30 (top), Q31-40 (bottom)
      Col3: Q41-50 (top), Q51-60 (bottom)
      Col4: Q61-70 (top), Q71-80 (bottom)
      Col5: Q81-90 (top), Q91-100 (bottom)
    All in one big JAWABAN section at bottom of LJK
    Warped image is 1000x1414
    """
    CHOICES = ['A', 'B', 'C', 'D', 'E']
    # JAWABAN section starts around y=760, ends around y=1380
    # 5 column-groups, each with 2 sub-blocks of 10 rows
    # x positions of each column group (5 groups across width)
    # Each group: answer bubbles for 5 choices (A-E)
    # Layout (x1, y1_top, x2, y2_top, y1_bot, y2_bot, q_top_start, q_bot_start)
    # Koordinat dikalibrasi ulang:
    # - x1 digeser lebih ke KIRI agar kolom A tertangkap penuh (bubble A tidak terpotong)
    # - setiap block lebar ~195px, gap antar block ~5px
    # - y-range disesuaikan agar 10 baris terdeteksi dengan benar
    BLOCKS = [
        # x1,  y1_top, x2,  y2_top, y1_bot, y2_bot, q_top, q_bot
        (  5,  755, 198, 1058,  1058, 1375,   1,  11),
        (198,  755, 393, 1058,  1058, 1375,  21,  31),
        (393,  755, 588, 1058,  1058, 1375,  41,  51),
        (588,  755, 783, 1058,  1058, 1375,  61,  71),
        (783,  755, 995, 1058,  1058, 1375,  81,  91),
    ]
    all_answers = {}
    soal_done = 0
    for x1, y1t, x2, y2t, y1b, y2b, q_top, q_bot in BLOCKS:
        # Top sub-block (10 questions)
        if soal_done < total_soal:
            n = min(10, total_soal - soal_done)
            roi = crop_gray(warped_np, y1t, y2t, x1, x2)
            r, *_ = scan_grid(roi, num_cols=5, num_rows=10, labels=CHOICES, per_row=True)
            for i, ans in enumerate(r[:n]):
                all_answers[q_top + i] = ans
            soal_done += n
        # Bottom sub-block (10 questions)
        if soal_done < total_soal:
            n = min(10, total_soal - soal_done)
            roi = crop_gray(warped_np, y1b, y2b, x1, x2)
            r, *_ = scan_grid(roi, num_cols=5, num_rows=10, labels=CHOICES, per_row=True)
            for i, ans in enumerate(r[:n]):
                all_answers[q_bot + i] = ans
            soal_done += n
    return all_answers

def visualize_answer_grid(warped_np, total_soal=100):
    """
    Tampilkan overlay grid JAWABAN di atas warped image untuk debugging koordinat.
    Menggambar kotak hijau di setiap ROI dan label nomor blok.
    """
    from PIL import ImageDraw, ImageFont
    vis = Image.fromarray(warped_np).copy()
    draw = ImageDraw.Draw(vis)

    BLOCKS = [
        (  5,  755, 198, 1058,  1058, 1375,   1,  11),
        (198,  755, 393, 1058,  1058, 1375,  21,  31),
        (393,  755, 588, 1058,  1058, 1375,  41,  51),
        (588,  755, 783, 1058,  1058, 1375,  61,  71),
        (783,  755, 995, 1058,  1058, 1375,  81,  91),
    ]
    colors = ['#00FF00','#00FFFF','#FF00FF','#FFFF00','#FF8800']
    CHOICES = ['A','B','C','D','E']

    for idx, (x1, y1t, x2, y2t, y1b, y2b, q_top, q_bot) in enumerate(BLOCKS):
        col = colors[idx % len(colors)]
        # Draw top block outline
        draw.rectangle([x1, y1t, x2, y2t], outline=col, width=2)
        draw.text((x1+2, y1t+2), f"Q{q_top}-{q_top+9}", fill=col)
        # Draw bottom block outline
        draw.rectangle([x1, y1b, x2, y2b], outline=col, width=2)
        draw.text((x1+2, y1b+2), f"Q{q_bot}-{q_bot+9}", fill=col)

        # Draw column lines for A-E within each block (top)
        bw = x2 - x1
        col_w = bw / 5
        for ci in range(6):
            lx = int(x1 + ci * col_w)
            draw.line([(lx, y1t), (lx, y2t)], fill=col, width=1)
            draw.line([(lx, y1b), (lx, y2b)], fill=col, width=1)
            if ci < 5:
                draw.text((int(x1 + ci*col_w + col_w/2 - 4), y1t + 5), CHOICES[ci], fill=col)
                draw.text((int(x1 + ci*col_w + col_w/2 - 4), y1b + 5), CHOICES[ci], fill=col)

        # Draw row lines for 10 rows (top block)
        bh_t = y2t - y1t
        row_h_t = bh_t / 10
        for ri in range(11):
            ry = int(y1t + ri * row_h_t)
            draw.line([(x1, ry), (x2, ry)], fill=col, width=1)
        bh_b = y2b - y1b
        row_h_b = bh_b / 10
        for ri in range(11):
            ry = int(y1b + ri * row_h_b)
            draw.line([(x1, ry), (x2, ry)], fill=col, width=1)

    return vis


def detect_answers_debug(warped_np, total_soal=100):
    """Sama dengan detect_answers tapi mengembalikan density maps untuk debugging."""
    CHOICES = ['A', 'B', 'C', 'D', 'E']
    BLOCKS = [
        (  5,  755, 198, 1058,  1058, 1375,   1,  11),
        (198,  755, 393, 1058,  1058, 1375,  21,  31),
        (393,  755, 588, 1058,  1058, 1375,  41,  51),
        (588,  755, 783, 1058,  1058, 1375,  61,  71),
        (783,  755, 995, 1058,  1058, 1375,  81,  91),
    ]
    all_answers = {}
    debug_info = []
    soal_done = 0
    for x1, y1t, x2, y2t, y1b, y2b, q_top, q_bot in BLOCKS:
        if soal_done < total_soal:
            n = min(10, total_soal - soal_done)
            roi = crop_gray(warped_np, y1t, y2t, x1, x2)
            r, dmap, b0, b1, row_h, col_w = scan_grid(roi, num_cols=5, num_rows=10, labels=CHOICES, per_row=True)
            for i, ans in enumerate(r[:n]):
                all_answers[q_top + i] = ans
            debug_info.append({'label': f"Q{q_top}-{q_top+n-1}", 'density': dmap, 'answers': r[:n], 'q_start': q_top})
            soal_done += n
        if soal_done < total_soal:
            n = min(10, total_soal - soal_done)
            roi = crop_gray(warped_np, y1b, y2b, x1, x2)
            r, dmap, b0, b1, row_h, col_w = scan_grid(roi, num_cols=5, num_rows=10, labels=CHOICES, per_row=True)
            for i, ans in enumerate(r[:n]):
                all_answers[q_bot + i] = ans
            debug_info.append({'label': f"Q{q_bot}-{q_bot+n-1}", 'density': dmap, 'answers': r[:n], 'q_start': q_bot})
            soal_done += n
    return all_answers, debug_info


def score_answers(student_answers, answer_key):
    correct = wrong = unanswered = 0
    for q, key in answer_key.items():
        student = student_answers.get(q)
        if student is None: unanswered += 1
        elif student == key: correct += 1
        else: wrong += 1
    return correct, wrong, unanswered

def compute_score(correct, total_soal, method="standard"):
    if method == "standard":
        return round((correct / total_soal) * 100, 2) if total_soal > 0 else 0
    wrong = total_soal - correct
    return round(max(0, (correct - wrong * 0.25) / total_soal * 100), 2)

# ─── ML HELPERS ─────────────────────────────────────────────

def extract_features(record, class_data):
    n_correct = record['correct']; n_wrong = record['wrong']
    n_unans = record['unanswered']; total = record['total_soal']
    if class_data:
        scores = [r['score'] for r in class_data]
        class_mean = np.mean(scores); class_var = np.var(scores); class_std = np.std(scores)
        n_students = len(class_data)
        correct_counts = np.zeros(total)
        for r in class_data:
            for q, ans in r['student_answers'].items():
                if ans == r['answer_key'].get(int(q)):
                    idx = int(q) - 1
                    if 0 <= idx < total: correct_counts[idx] += 1
        avg_difficulty = float(np.mean(correct_counts / max(n_students, 1)))
    else:
        class_mean = n_correct / total * 100 if total > 0 else 0
        class_var = class_std = avg_difficulty = 0
    return [n_correct, n_wrong, n_unans, class_mean, class_var, class_std, avg_difficulty]

def train_and_predict(records):
    if len(records) < 2: return None
    features = [extract_features(r, records) for r in records]
    actuals  = [r['score'] for r in records]
    X = np.array(features); y = np.array(actuals)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    lr = LinearRegression(); lr.fit(X_sc, y); y_lr = lr.predict(X_sc)
    svr = SVR(kernel='rbf', C=10, epsilon=0.5); svr.fit(X_sc, y); y_svr = svr.predict(X_sc)
    def metrics(pred):
        return {'MAE': round(mean_absolute_error(y, pred), 3),
                'RMSE': round(np.sqrt(mean_squared_error(y, pred)), 3),
                'R2': round(r2_score(y, pred) if len(y) > 1 else 0.0, 3)}
    return {'actuals': y.tolist(), 'pred_lr': y_lr.tolist(), 'pred_svr': y_svr.tolist(),
            'metrics_lr': metrics(y_lr), 'metrics_svr': metrics(y_svr),
            'lr_model': lr, 'feature_names': ['n_correct','n_wrong','n_unanswered','class_mean','class_var','class_std','avg_difficulty']}

# ─── SESSION STATE ───────────────────────────────────────────
def init_state():
    defaults = {'answer_key': {}, 'total_soal': 20, 'sesi_nama': '', 'kode_kelas': '',
                'kode_dosen': '', 'scoring': 'standard', 'records': [], 'ml_results': None, 'step': 'setup'}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
init_state()

# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📝 LJK Scanner")
    st.markdown("**Computer Vision Project**")
    st.markdown("*Binus University @Alam Sutera*")
    st.divider()
    step = st.session_state.step
    for s, lbl in [('setup','⚙️ Setup Sesi'),('scan','📸 Scan LJK'),('results','📊 Hasil'),('ml','🤖 Model ML')]:
        st.markdown(f"`{'→' if s == step else '  '}`{lbl}")
    st.divider()
    if st.session_state.records:
        scores = [r['score'] for r in st.session_state.records]
        st.markdown(f"**{len(scores)}** lembar ter-scan")
        st.markdown(f"Rata-rata: **{np.mean(scores):.1f}**")
        st.markdown(f"Tertinggi: **{max(scores):.1f}** | Terendah: **{min(scores):.1f}**")
    st.divider()
    if st.button("🔄 Reset Sesi"):
        for k in ['answer_key','records','ml_results','step','sesi_nama','kode_kelas','kode_dosen']:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

# ══════════════════════════════════════════
#  STEP 1 — SETUP
# ══════════════════════════════════════════
if st.session_state.step == 'setup':
    st.title("⚙️ Setup Sesi Ujian")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informasi Sesi")
        st.session_state.sesi_nama  = st.text_input("Nama Sesi / Mata Kuliah", value=st.session_state.sesi_nama or "Computer Vision UAS")
        st.session_state.kode_kelas = st.text_input("Kode Kelas", value=st.session_state.kode_kelas or "LK01")
        st.session_state.kode_dosen = st.text_input("Kode Dosen", value=st.session_state.kode_dosen or "DS123")
        new_total = st.number_input("Jumlah Soal (1–100)", min_value=1, max_value=100, value=st.session_state.total_soal)
        if new_total != st.session_state.total_soal:
            st.session_state.total_soal = new_total
            st.session_state.key_text = make_key_text(new_total)
            st.rerun()
        st.session_state.scoring    = st.selectbox("Metode Penilaian", ["standard","penalty"],
            format_func=lambda x: "Standar (benar/total × 100)" if x=="standard" else "Penalty (-0.25 per salah)")
    with col2:
        st.subheader("Kunci Jawaban")
        total = st.session_state.total_soal
        if 'key_text' not in st.session_state:
            st.session_state.key_text = make_key_text(total)
        key_text = st.text_area("Format: `1. A` `2. B` dll.", value=st.session_state.key_text, height=300, label_visibility="collapsed")
        st.session_state.key_text = key_text
        answer_key = {}; errors = []
        for line in key_text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            # support both "1. A" and "1,A"
            if '. ' in line:
                parts = line.split('. ', 1)
            else:
                parts = line.split(',', 1)
            if len(parts) != 2: errors.append(f"Format salah: `{line}`"); continue
            try:
                q = int(parts[0].strip()); ans = parts[1].strip().upper()
                if ans not in ['A','B','C','D','E']: errors.append(f"Jawaban tidak valid di soal {q}"); continue
                answer_key[q] = ans
            except ValueError: errors.append(f"Nomor soal tidak valid: `{line}`")
        if errors:
            for e in errors[:3]: st.error(e)
        else:
            st.success(f"✅ {len(answer_key)} kunci jawaban valid")
    st.divider()
    if st.button("▶ Mulai Scan", disabled=len(answer_key)==0):
        st.session_state.answer_key = answer_key
        st.session_state.step = 'scan'; st.rerun()

# ══════════════════════════════════════════
#  STEP 2 — SCAN
# ══════════════════════════════════════════
elif st.session_state.step == 'scan':
    st.title("📸 Scan Lembar Jawaban (LJK)")
    col1, _ = st.columns([1,5])
    with col1:
        if st.button("← Setup"): st.session_state.step='setup'; st.rerun()
    st.info(f"**Sesi:** {st.session_state.sesi_nama} | **Kelas:** {st.session_state.kode_kelas} | **Soal:** {st.session_state.total_soal} | **Ter-scan:** {len(st.session_state.records)}")
    uploaded_files = st.file_uploader("Upload foto LJK (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded in uploaded_files:
            if any(r['filename']==uploaded.name for r in st.session_state.records):
                st.warning(f"⚠️ `{uploaded.name}` sudah di-scan, dilewati."); continue
            with st.expander(f"🔍 Memproses: `{uploaded.name}`", expanded=True):
                img_pil = Image.open(uploaded)
                with st.spinner("Mendeteksi sudut & melakukan warp..."):
                    warped_np, ok = warp_ljk(img_pil)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(img_pil, caption="Input asli", use_container_width=True)
                with col_b:
                    if ok:
                        st.image(warped_np, caption="Setelah warp", use_container_width=True)
                    else:
                        st.error("❌ Gagal mendeteksi 4 sudut LJK. Pastikan foto jelas."); continue
                # ── Debug grid visualization ──
                show_debug = st.checkbox("🔍 Tampilkan debug grid koordinat", key=f"dbg_{uploaded.name}")
                if show_debug and ok:
                    grid_vis = visualize_answer_grid(warped_np, st.session_state.total_soal)
                    st.image(grid_vis, caption="Debug: Grid ROI JAWABAN (setiap warna = 1 blok, A–E per kolom, 10 baris per sub-blok)", use_container_width=True)

                with st.spinner("Mendeteksi nama, NIM, jawaban..."):
                    nama    = detect_nama(warped_np)
                    nim     = detect_nim(warped_np)
                    tanggal = detect_tanggal(warped_np)
                    if show_debug:
                        answers, dbg_info = detect_answers_debug(warped_np, st.session_state.total_soal)
                        with st.expander("🧪 Debug: Density Z-score per Blok", expanded=False):
                            CHOICES = ['A','B','C','D','E']
                            for blk in dbg_info:
                                ans_list = [f"Q{blk['q_start']+i}={a}" for i,a in enumerate(blk['answers'])]
                                st.markdown(f"**{blk['label']}** → {ans_list}")
                                fig_h, ax_h = plt.subplots(figsize=(5,2), facecolor='#0f172a')
                                ax_h.set_facecolor('#1e293b')
                                n_rows = len(blk['answers'])
                                im = ax_h.imshow(blk['density'][:n_rows, :5], aspect='auto', cmap='RdYlGn', vmin=-2, vmax=2)
                                ax_h.set_xticks(range(5)); ax_h.set_xticklabels(CHOICES, color='#e2e8f0', fontsize=8)
                                ax_h.set_yticks(range(n_rows))
                                ax_h.set_yticklabels([f"Q{blk['q_start']+i}" for i in range(n_rows)], color='#e2e8f0', fontsize=7)
                                plt.colorbar(im, ax=ax_h, shrink=0.8)
                                ax_h.set_title("Z-score (hijau=tinggi=bubble terisi)", color='#f1f5f9', fontsize=8)
                                st.pyplot(fig_h)
                    else:
                        answers = detect_answers(warped_np, st.session_state.total_soal)
                correct, wrong, unanswered = score_answers(answers, st.session_state.answer_key)
                score = compute_score(correct, st.session_state.total_soal, st.session_state.scoring)
                m1,m2,m3,m4,m5 = st.columns(5)
                for col, val, lbl, color in [
                    (m1,score,"Skor","#38bdf8"),(m2,correct,"Benar","#22c55e"),
                    (m3,wrong,"Salah","#ef4444"),(m4,unanswered,"Kosong","#94a3b8"),
                    (m5,nim,"NIM","#f59e0b")]:
                    col.markdown(f'<div class="metric-card"><div class="val" style="color:{color};font-size:{"2rem" if lbl!="NIM" else "1rem"}">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
                st.markdown(f"**Nama:** {nama}  |  **Tanggal:** {tanggal}")
                with st.expander("📋 Detail Jawaban per Soal"):
                    cols_grid = st.columns(10)
                    for q in range(1, st.session_state.total_soal+1):
                        s_ans = answers.get(q); k_ans = st.session_state.answer_key.get(q,'?')
                        cidx = (q-1)%10
                        with cols_grid[cidx]:
                            if s_ans is None: lbl,bg = f"**{q}**: –","#374151"
                            elif s_ans==k_ans: lbl,bg = f"**{q}**: {s_ans} ✓","#14532d"
                            else: lbl,bg = f"**{q}**: {s_ans} ✗","#450a0a"
                            st.markdown(f'<div style="background:{bg};border-radius:4px;padding:3px 5px;font-size:0.7rem;margin-bottom:4px;text-align:center">{lbl}</div>', unsafe_allow_html=True)
                record = {'filename':uploaded.name,'nama':nama,'nim':nim,'tanggal':tanggal,
                          'correct':correct,'wrong':wrong,'unanswered':unanswered,'score':score,
                          'total_soal':st.session_state.total_soal,
                          'student_answers':{str(k):v for k,v in answers.items()},
                          'answer_key':st.session_state.answer_key}
                st.session_state.records.append(record)
                st.session_state.ml_results = None
                st.success(f"✅ `{uploaded.name}` berhasil disimpan.")
    st.divider()
    if st.button("📊 Lihat Hasil", disabled=len(st.session_state.records)==0):
        st.session_state.step='results'; st.rerun()

# ══════════════════════════════════════════
#  STEP 3 — RESULTS
# ══════════════════════════════════════════
elif st.session_state.step == 'results':
    st.title("📊 Hasil & Analitik Kelas")
    records = st.session_state.records
    if not records:
        st.warning("Belum ada data.")
        if st.button("← Scan"): st.session_state.step='scan'; st.rerun()
        st.stop()
    c1,c2 = st.columns([1,1])
    with c1:
        if st.button("← Scan Lagi"): st.session_state.step='scan'; st.rerun()
    with c2:
        if st.button("🤖 Analisis ML"): st.session_state.step='ml'; st.rerun()
    scores = [r['score'] for r in records]
    total_soal = records[0]['total_soal']
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,val,lbl,color in [(c1,len(records),"Total Mahasiswa","#38bdf8"),
        (c2,f"{np.mean(scores):.1f}","Rata-rata","#a78bfa"),(c3,f"{max(scores):.1f}","Tertinggi","#22c55e"),
        (c4,f"{min(scores):.1f}","Terendah","#ef4444"),(c5,f"{np.std(scores):.2f}","Std Dev","#f59e0b")]:
        col.markdown(f'<div class="metric-card"><div class="val" style="color:{color}">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.divider()
    tab1,tab2,tab3 = st.tabs(["📋 Tabel Nilai","📈 Grafik","💾 Export"])
    with tab1:
        df = pd.DataFrame([{'Nama':r['nama'],'NIM':r['nim'],'Tanggal':r['tanggal'],
            'Benar':r['correct'],'Salah':r['wrong'],'Kosong':r['unanswered'],'Skor':r['score']} for r in records])
        st.dataframe(df, use_container_width=True)
    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(6,4),facecolor='#0f172a'); ax.set_facecolor('#1e293b')
            ax.hist(scores,bins=range(0,105,10),color='#0ea5e9',edgecolor='#0f172a',alpha=0.9)
            ax.axvline(np.mean(scores),color='#f59e0b',lw=2,linestyle='--',label=f'Mean={np.mean(scores):.1f}')
            ax.set_xlabel("Nilai",color='#e2e8f0'); ax.set_ylabel("Jumlah",color='#e2e8f0')
            ax.set_title("Distribusi Nilai",color='#f1f5f9',fontweight='bold')
            ax.tick_params(colors='#94a3b8'); ax.legend(facecolor='#1e293b',labelcolor='#f1f5f9')
            for sp in ax.spines.values(): sp.set_color('#334155')
            st.pyplot(fig)
        with col2:
            grades = {'A(≥80)':0,'B(70-79)':0,'C(60-69)':0,'D(50-59)':0,'E(<50)':0}
            for s in scores:
                if s>=80: grades['A(≥80)']+=1
                elif s>=70: grades['B(70-79)']+=1
                elif s>=60: grades['C(60-69)']+=1
                elif s>=50: grades['D(50-59)']+=1
                else: grades['E(<50)']+=1
            lbl_pie=[k for k,v in grades.items() if v>0]; sz=[v for v in grades.values() if v>0]
            fig2,ax2=plt.subplots(figsize=(5,4),facecolor='#0f172a'); ax2.set_facecolor('#1e293b')
            ax2.pie(sz,labels=lbl_pie,colors=['#22c55e','#84cc16','#f59e0b','#f97316','#ef4444'][:len(lbl_pie)],
                    autopct='%1.0f%%',textprops={'color':'#e2e8f0','fontsize':9})
            ax2.set_title("Distribusi Grade",color='#f1f5f9',fontweight='bold')
            st.pyplot(fig2)
        if len(records)>1:
            st.subheader("Tingkat Kesulitan per Soal")
            rates=[sum(1 for r in records if r['student_answers'].get(str(q))==r['answer_key'].get(q))/len(records)*100 for q in range(1,total_soal+1)]
            fig3,ax3=plt.subplots(figsize=(14,3),facecolor='#0f172a'); ax3.set_facecolor('#1e293b')
            ax3.bar(range(1,total_soal+1),rates,color=['#22c55e' if v>=70 else '#f59e0b' if v>=40 else '#ef4444' for v in rates])
            ax3.set_xlabel("Nomor Soal",color='#e2e8f0'); ax3.set_ylabel("% Benar",color='#e2e8f0')
            ax3.set_title("Tingkat Kesulitan per Soal",color='#f1f5f9',fontweight='bold')
            ax3.tick_params(colors='#94a3b8')
            for sp in ax3.spines.values(): sp.set_color('#334155')
            st.pyplot(fig3)
    with tab3:
        df_rekap=pd.DataFrame([{'Nama':r['nama'],'NIM':r['nim'],'Tanggal':r['tanggal'],
            'Benar':r['correct'],'Salah':r['wrong'],'Kosong':r['unanswered'],'Skor':r['score']} for r in records])
        rows_detail=[]
        for r in records:
            row={'NIM':r['nim'],'Nama':r['nama'],'Skor':r['score']}
            for q in range(1,total_soal+1):
                row[f'Q{q}']=r['student_answers'].get(str(q),'-'); row[f'Q{q}_kunci']=r['answer_key'].get(q,'?')
            rows_detail.append(row)
        df_detail=pd.DataFrame(rows_detail)
        df_stats=pd.DataFrame([{'Metrik':m,'Nilai':v} for m,v in {
            'Total Mahasiswa':len(records),'Rata-rata':round(np.mean(scores),2),
            'Tertinggi':max(scores),'Terendah':min(scores),'Std Dev':round(np.std(scores),2)}.items()])
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine='openpyxl') as w:
            df_rekap.to_excel(w,sheet_name='Rekap Nilai',index=False)
            df_detail.to_excel(w,sheet_name='Detail Jawaban',index=False)
            df_stats.to_excel(w,sheet_name='Statistik Kelas',index=False)
        buf.seek(0)
        st.download_button("⬇️ Download Excel (Multi-Sheet)",data=buf,
            file_name=f"hasil_{st.session_state.kode_kelas}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════
#  STEP 4 — ML
# ══════════════════════════════════════════
elif st.session_state.step == 'ml':
    st.title("🤖 Model Machine Learning — Score Prediction")
    records = st.session_state.records
    if not records:
        st.warning("Belum ada data scan.")
        if st.button("← Scan"): st.session_state.step='scan'; st.rerun()
        st.stop()
    if st.button("← Hasil"): st.session_state.step='results'; st.rerun()
    if len(records)<3:
        st.info(f"Model ML butuh minimal 3 data. Saat ini ada **{len(records)}** data."); st.stop()
    if st.session_state.ml_results is None or st.button("🔄 Re-train"):
        with st.spinner("Melatih LR & SVR..."):
            st.session_state.ml_results = train_and_predict(records)
    ml = st.session_state.ml_results
    if ml is None: st.error("Training gagal."); st.stop()
    THRESHOLD = {'MAE':5.0,'RMSE':7.0,'R2':0.80}
    st.subheader("📏 Metrik Evaluasi")
    c1,c2 = st.columns(2)
    for col,name,met in [(c1,"Linear Regression",ml['metrics_lr']),(c2,"SVR",ml['metrics_svr'])]:
        with col:
            st.markdown(f"**{name}**")
            for k,v in met.items():
                ok = v>=THRESHOLD[k] if k=='R2' else v<=THRESHOLD[k]
                color='#22c55e' if ok else '#f59e0b'
                thr=f"({'≥' if k=='R2' else '≤'}{THRESHOLD[k]})"
                st.markdown(f'<div class="metric-card" style="margin-bottom:8px"><div class="val" style="color:{color}">{v}</div><div class="lbl">{"✅" if ok else "⚠️"} {k} {thr}</div></div>', unsafe_allow_html=True)
    st.subheader("📈 Aktual vs Prediksi")
    names=[r['nim'] or f"S{i+1}" for i,r in enumerate(records)]
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor='#0f172a')
    for ax,preds,title,color in [(axes[0],ml['pred_lr'],"Linear Regression",'#38bdf8'),(axes[1],ml['pred_svr'],"SVR",'#a78bfa')]:
        ax.set_facecolor('#1e293b')
        ax.scatter(ml['actuals'],preds,color=color,alpha=0.8,s=80)
        mv=min(min(ml['actuals']),min(preds)); xv=max(max(ml['actuals']),max(preds))
        ax.plot([mv,xv],[mv,xv],'w--',lw=1.5,alpha=0.4)
        for i,(a,p) in enumerate(zip(ml['actuals'],preds)):
            ax.annotate(names[i],(a,p),textcoords='offset points',xytext=(5,3),fontsize=7,color='#94a3b8')
        ax.set_xlabel("Aktual",color='#e2e8f0'); ax.set_ylabel("Prediksi",color='#e2e8f0')
        ax.set_title(title,color='#f1f5f9',fontweight='bold'); ax.tick_params(colors='#94a3b8')
        for sp in ax.spines.values(): sp.set_color('#334155')
    plt.tight_layout(); st.pyplot(fig)
    st.subheader("📋 Tabel Prediksi")
    df_ml=pd.DataFrame({'Nama':[r['nama'] for r in records],'NIM':[r['nim'] for r in records],
        'Aktual':ml['actuals'],'Pred LR':[round(p,1) for p in ml['pred_lr']],
        'Pred SVR':[round(p,1) for p in ml['pred_svr']],
        'Selisih LR':[round(abs(a-p),1) for a,p in zip(ml['actuals'],ml['pred_lr'])],
        'Selisih SVR':[round(abs(a-p),1) for a,p in zip(ml['actuals'],ml['pred_svr'])]})
    st.dataframe(df_ml,use_container_width=True)
    buf_ml=io.BytesIO()
    with pd.ExcelWriter(buf_ml,engine='openpyxl') as w:
        df_ml.to_excel(w,sheet_name='Prediksi ML',index=False)
    buf_ml.seek(0)
    st.download_button("⬇️ Download Hasil ML",data=buf_ml,
        file_name=f"ml_{st.session_state.kode_kelas}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
