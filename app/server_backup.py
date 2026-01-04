# app/server.py
import os
import re
import pickle
from typing import List, Tuple

import numpy as np
import faiss

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")

FAISS_PATH = os.path.join(DATA_DIR, "faiss_index_e5.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "all_chunks_e5.pkl")

# logo/favicons
LOGO_PATH = os.path.join(DATA_DIR, "logo.png")  # ضعي شعارك هنا
FAVICON_PATH = LOGO_PATH  # نفس اللوغو

# اختياري: صور PNG شفافة للملصقات (إذا وضعتيها)
# ضعي ملفات PNG هنا: data/stickers/
STICKER_DIR = os.path.join(DATA_DIR, "stickers")

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 5
SEARCH_K = 80

# عتبة "لا أعلم" (إذا الثقة منخفضة)
DONT_KNOW_THRESHOLD = 0.72


# =========================
# LOAD
# =========================
e5 = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks: List[str] = pickle.load(f)


# =========================
# RETRIEVAL HELPERS
# =========================
AR_STOP = set([
    "ما","هي","هل","أين","من","في","على","إلى","عن","هذا","هذه","ذلك","تكون","يكون",
    "كم","متى","لماذا","كيف","ماهو","ماهي","هو","هي","؟"
])

def _keywords_ar(text: str) -> List[str]:
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    words = [w.strip() for w in text.split() if len(w.strip()) >= 3]
    words = [w for w in words if w not in AR_STOP]
    return list(dict.fromkeys(words))

def retrieve_chunks(question: str, top_k=TOP_K, search_k=SEARCH_K) -> Tuple[List[str], List[float]]:
    q = "query: " + question.strip()
    q_emb = e5.encode([q], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, search_k)

    candidates = [chunks[i] for i in idxs[0]]
    cand_scores = list(scores[0])

    kws = _keywords_ar(question)
    def kw_score(ch: str) -> int:
        return sum(1 for w in kws if w in ch)

    ranked = sorted(
        zip(candidates, cand_scores),
        key=lambda x: (kw_score(x[0]), x[1]),
        reverse=True
    )
    best = ranked[:top_k]
    return [b[0] for b in best], [float(b[1]) for b in best]

def build_answer(question: str, retrieved: List[str], scores: List[float]) -> Tuple[str, float]:
    best_score = scores[0] if scores else 0.0
    if best_score < DONT_KNOW_THRESHOLD:
        return ("لا أعلم، ليس لدي معلومات كافية عن هذا السؤال حاليًا، "
                "لكنني في مرحلة التطوير."), best_score

    text = retrieved[0].strip()

    # نأخذ أول جملة مفيدة (مع حماية)
    splitters = ["۔", ".", "!", "؟", "\n"]
    ans = text
    for s in splitters:
        if s in ans:
            ans = ans.split(s)[0].strip()
            break

    if len(ans) < 18:
        ans = text[:220].strip()

    return ans, best_score


# =========================
# FASTAPI
# =========================
app = FastAPI(title="smARtABIC Agent")

# static files: logo + stickers
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

class AskRequest(BaseModel):
    question: str


# =========================
# ROUTES
# =========================
@app.get("/favicon.ico")
def favicon():
    if os.path.exists(FAVICON_PATH):
        return FileResponse(FAVICON_PATH)
    # fallback: لا favicon
    return JSONResponse({"detail": "favicon not found"}, status_code=404)

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(_home_html())

@app.post("/ask")
def ask(req: AskRequest):
    question = (req.question or "").strip()
    retrieved, scores = retrieve_chunks(question)
    answer, best_score = build_answer(question, retrieved, scores)

    return JSONResponse({
        "question": question,
        "answer": answer,
        "best_score": float(best_score),
        "chunks": retrieved,
        "scores": scores
    })


# =========================
# HTML (سينمائي + parallax + stickers + galaxy bg)
# =========================
def _home_html() -> str:
    # أسماء صور PNG شفافة (اختياري). إذا لم تجديها، سيظهر Emoji تلقائيًا.
    # ضعي هذه الملفات إن أردت:
    # data/stickers/robot.png
    # data/stickers/search.png
    # data/stickers/book.png
    # data/stickers/bolt.png
    return f"""
<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>smARtABIC Agent</title>

  <style>
    :root {{
      --txt: rgba(255,255,255,.92);
      --muted: rgba(255,255,255,.70);
      --border: rgba(255,255,255,.18);

      /* الزجاج */
      --glass: rgba(10, 18, 40, .26);
      --glass2: rgba(255,255,255,.07);

      /* ألوان */
      --violet: #7c3aed;
      --sky: #38bdf8;
      --mint: #22c55e;
      --pink: #fb7185;
      --gold1: #ffd700;
      --gold2: #ffb703;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin:0;
      color: var(--txt);
      font-family: system-ui, -apple-system, Segoe UI, Arial, sans-serif;
      min-height:100vh;
      overflow-x:hidden;

      /* تدرج فضائي واضح (نيلي → سماوي → قطبي → أسود خفيف) */
      background:
        radial-gradient(1100px 700px at 15% 20%, rgba(124,58,237,.22), transparent 65%),
        radial-gradient(900px 600px at 80% 25%, rgba(56,189,248,.25), transparent 60%),
        radial-gradient(900px 700px at 60% 80%, rgba(34,197,94,.18), transparent 65%),
        radial-gradient(1000px 800px at 40% 75%, rgba(251,113,133,.12), transparent 68%),
        linear-gradient(180deg, #0b1026 0%, #0c2346 30%, #0e4a6e 55%, #071424 82%, #04040c 100%);
    }}

    /* ====== Parallax Layers (تتحرك مع الماوس) ====== */
    .layer {{
      position: fixed;
      inset: 0;
      pointer-events: none;
      transform: translate3d(var(--px,0px), var(--py,0px), 0);
      will-change: transform;
    }}

    /* سُدم / مجرات */
    .nebula {{
      z-index: -6;
      inset: -25%;
      background:
        radial-gradient(circle at 18% 25%, rgba(124,58,237,.45), transparent 58%),
        radial-gradient(circle at 78% 22%, rgba(56,189,248,.40), transparent 62%),
        radial-gradient(circle at 55% 70%, rgba(34,197,94,.30), transparent 66%),
        radial-gradient(circle at 30% 78%, rgba(251,113,133,.24), transparent 70%);
      filter: blur(46px) saturate(1.2);
      opacity: .95;
      animation: nebFloat 22s ease-in-out infinite alternate;
    }}
    @keyframes nebFloat {{
      from {{ transform: translate3d(-1%, -1%, 0) scale(1.03); }}
      to   {{ transform: translate3d(2%, 1%, 0) scale(1.10); }}
    }}

    /* نجوم (3 طبقات كثيفة) */
    .stars1, .stars2, .stars3 {{
      z-index: -5;
      background-repeat: repeat;
      background-size: 340px 340px;
      opacity: .85;
      animation: starsDrift linear infinite;
    }}

    .stars1 {{
      background-image:
        radial-gradient(2px 2px at 10% 20%, rgba(255,255,255,.95), transparent 55%),
        radial-gradient(1px 1px at 40% 80%, rgba(255,255,255,.85), transparent 55%),
        radial-gradient(2px 2px at 70% 30%, rgba(255,255,255,.95), transparent 55%),
        radial-gradient(1px 1px at 90% 60%, rgba(255,255,255,.85), transparent 55%),
        radial-gradient(1px 1px at 25% 55%, rgba(255,255,255,.9), transparent 55%),
        radial-gradient(2px 2px at 55% 15%, rgba(255,255,255,.95), transparent 55%);
      animation-duration: 190s;
    }}

    .stars2 {{
      opacity: .60;
      background-size: 560px 560px;
      background-image:
        radial-gradient(1px 1px at 20% 50%, rgba(255,255,255,.95), transparent 55%),
        radial-gradient(2px 2px at 60% 10%, rgba(255,255,255,.95), transparent 55%),
        radial-gradient(1px 1px at 80% 75%, rgba(255,255,255,.85), transparent 55%),
        radial-gradient(2px 2px at 35% 25%, rgba(255,255,255,.9), transparent 55%),
        radial-gradient(1px 1px at 75% 40%, rgba(255,255,255,.9), transparent 55%);
      animation-duration: 270s;
    }}

    .stars3 {{
      opacity: .45;
      background-size: 820px 820px;
      background-image:
        radial-gradient(2px 2px at 15% 70%, rgba(255,255,255,.95), transparent 55%),
        radial-gradient(1px 1px at 85% 35%, rgba(255,255,255,.90), transparent 55%),
        radial-gradient(2px 2px at 50% 50%, rgba(255,255,255,.80), transparent 55%);
      animation-duration: 380s;
    }}

    @keyframes starsDrift {{
      from {{ transform: translateY(0); }}
      to   {{ transform: translateY(-1400px); }}
    }}

    /* ====== شهب كثيرة ====== */
    .meteor {{
      position: fixed;
      width: 320px;
      height: 2px;
      background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,.95));
      filter: drop-shadow(0 0 10px rgba(56,189,248,.65));
      opacity: .0;
      z-index: -4;
      transform: rotate(18deg);
      animation: meteor 9.5s linear infinite;
      pointer-events:none;
    }}

    /* كل شهاب بإعدادات مختلفة */
    .m1 {{ top:-12%; left:-30%; animation-delay: 0s;  animation-duration: 9s;  }}
    .m2 {{ top: 8%;  left:-40%; animation-delay: 1.8s; animation-duration: 11s; }}
    .m3 {{ top: 24%; left:-45%; animation-delay: 3.2s; animation-duration: 10s; }}
    .m4 {{ top: 44%; left:-55%; animation-delay: 4.6s; animation-duration: 12s; }}
    .m5 {{ top: 62%; left:-45%; animation-delay: 6.2s; animation-duration: 10.5s; }}
    .m6 {{ top: 78%; left:-60%; animation-delay: 7.5s; animation-duration: 13s; }}
    .m7 {{ top: 90%; left:-55%; animation-delay: 8.6s; animation-duration: 11.5s; }}

    @keyframes meteor {{
      0%   {{ transform: translateX(0) translateY(0) rotate(18deg); opacity:0; }}
      8%   {{ opacity:.95; }}
      100% {{ transform: translateX(170vw) translateY(90vh) rotate(18deg); opacity:0; }}
    }}

    /* ====== LAYOUT: واجهة بالمنتصف ====== */
    .page {{
      position: relative;
      z-index: 5;
      padding: 26px 16px 44px;
      display:flex;
      justify-content:center;
    }}
    .container {{
      width: min(1120px, 100%);
    }}

    /* ====== TOPBAR ====== */
    .topbar {{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap: 14px;
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(10, 18, 40, .22);
      border: 1px solid var(--border);
      backdrop-filter: blur(14px);
      box-shadow: 0 18px 70px rgba(0,0,0,.30);
      margin-bottom: 14px;
    }}

    .brand {{
      display:flex;
      align-items:center;
      gap: 16px;
      min-width: 260px;
    }}

    /* تكبير اللوغو + وضوح تفاصيل */
    .brand img {{
      width: 118px;
      height: 118px;
      border-radius: 26px;
      border: 1px solid rgba(255,255,255,.16);
      box-shadow:
        0 0 28px rgba(255,255,255,.18),
        0 0 64px rgba(56,189,248,.35),
        0 0 80px rgba(124,58,237,.28);
      background: rgba(255,255,255,.05);
      object-fit: cover;
    }}

    .title h1 {{
      margin:0;
      font-size: 26px;
      line-height: 1.1;
      letter-spacing: .3px;
      font-weight: 900;
    }}
    .subtitle {{
      margin-top: 6px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 650;
    }}

    /* فضي لامع */
    .silver {{
      background: linear-gradient(180deg, #ffffff, #cfd3d8, #ffffff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-shadow: 0 0 12px rgba(255,255,255,.26);
    }}

    /* ذهبي مرصّع */
    .gold {{
      background: linear-gradient(180deg, #fff2b2, var(--gold1), var(--gold2), #fff2b2);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-shadow:
        0 0 14px rgba(255,215,0,.55),
        0 0 26px rgba(255,180,60,.45);
    }}

    .badge {{
      font-size: 12px;
      font-weight: 800;
      color: rgba(255,255,255,.9);
      padding: 10px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,.14);
      background: rgba(255,255,255,.06);
      backdrop-filter: blur(10px);
      white-space:nowrap;
    }}

    /* ====== ASK BAR ====== */
    .ask {{
      display:flex;
      gap: 10px;
      align-items:stretch;
      padding: 14px;
      border-radius: 18px;
      background: rgba(10, 18, 40, .20);
      border: 1px solid rgba(255,255,255,.14);
      backdrop-filter: blur(14px);
      box-shadow: 0 16px 56px rgba(0,0,0,.24);
      margin-bottom: 14px;
    }}

    input[type=text] {{
      flex:1;
      padding: 14px 14px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,.18);
      outline:none;
      background: rgba(255,255,255,.06);
      color: var(--txt);
      font-size: 15px;
    }}

    /* أزرار نابضة بالحياة */
    button {{
      min-width: 128px;
      border:none;
      border-radius: 14px;
      cursor:pointer;
      font-weight: 900;
      color: white;
      background: linear-gradient(135deg, var(--violet), var(--sky));
      box-shadow: 0 12px 40px rgba(124,58,237,.28);
      transition: transform .15s ease, filter .15s ease;
      position: relative;
      overflow:hidden;
    }}
    button::before {{
      content:"";
      position:absolute;
      inset:-60%;
      background: radial-gradient(circle, rgba(255,255,255,.35), transparent 60%);
      transform: translateX(-40%);
      animation: pulseGlow 2.2s ease-in-out infinite;
      opacity:.65;
    }}
    @keyframes pulseGlow {{
      0%{{ transform: translateX(-45%) scale(1); opacity:.35; }}
      50%{{ transform: translateX(5%) scale(1.15); opacity:.7; }}
      100%{{ transform: translateX(-45%) scale(1); opacity:.35; }}
    }}
    button:hover {{ transform: translateY(-1px); filter: brightness(1.06); }}
    button:active {{ transform: translateY(0px); }}

    /* ====== GRID ====== */
    .grid {{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}

    .card {{
      border-radius: 18px;
      background: rgba(10, 18, 40, .18); /* أخف ليظهر الفضاء */
      border: 1px solid rgba(255,255,255,.14);
      backdrop-filter: blur(12px);
      box-shadow: 0 16px 56px rgba(0,0,0,.22);
      padding: 14px;
      min-height: 440px;
      overflow:hidden;
      position:relative;
    }}

    /* مستطيلات منحنية تطفو على الجوانب داخل الكارد */
    .card::after {{
      content:"";
      position:absolute;
      inset:-40%;
      background:
        radial-gradient(circle at 30% 30%, rgba(124,58,237,.25), transparent 60%),
        radial-gradient(circle at 70% 55%, rgba(56,189,248,.22), transparent 62%),
        radial-gradient(circle at 55% 70%, rgba(34,197,94,.14), transparent 68%);
      filter: blur(22px);
      opacity:.55;
      animation: cardFloat 10s ease-in-out infinite alternate;
      pointer-events:none;
    }}
    @keyframes cardFloat {{
      from {{ transform: translate3d(-1%, -1%, 0) rotate(-1deg); }}
      to   {{ transform: translate3d(1.5%, 1%, 0) rotate(1deg); }}
    }}

    .card > * {{ position: relative; z-index: 2; }}

    .card h3 {{
      margin: 0 0 10px 0;
      font-size: 16px;
      display:flex;
      align-items:center;
      justify-content:space-between;
    }}

    .hint {{
      font-size: 12px;
      color: var(--muted);
      font-weight: 800;
    }}

    .box {{
      height: 380px;
      overflow:auto;
      padding: 12px;
      border-radius: 14px;
      background: rgba(255,255,255,.06);
      border: 1px solid rgba(255,255,255,.10);
    }}

    .chunk {{
      padding: 12px;
      border-radius: 14px;
      background: rgba(255,255,255,.05);
      border: 1px solid rgba(255,255,255,.10);
      margin-bottom: 10px;
      line-height: 1.8;
      color: rgba(255,255,255,.93);
    }}

    .score {{
      display:block;
      font-size: 12px;
      color: rgba(255,255,255,.72);
      margin-bottom: 6px;
      font-weight: 900;
    }}

    .answer {{
      white-space: pre-wrap;
      line-height: 1.9;
      font-size: 15px;
    }}

    .footer {{
      margin-top: 12px;
      text-align:center;
      color: rgba(255,255,255,.55);
      font-size: 12px;
      font-weight: 700;
    }}

    /* ====== Stickers 3D on sides (PNG + emoji) ====== */
    .stickers {{
      position: fixed;
      inset: 0;
      z-index: 4;
      pointer-events:none;
    }}

    .sticker {{
      position:absolute;
      width: 120px;
      height: 120px;
      border-radius: 28px;
      border: 1px solid rgba(255,255,255,.18);
      background: radial-gradient(circle at 30% 30%, rgba(255,255,255,.16), rgba(255,255,255,.04));
      backdrop-filter: blur(12px);
      box-shadow: 0 26px 90px rgba(0,0,0,.35);
      transform-style: preserve-3d;
      display:flex;
      align-items:center;
      justify-content:center;
      overflow:hidden;
    }}

    /* طبقة لمعة داخلية */
    .sticker::before {{
      content:"";
      position:absolute;
      inset:-40%;
      background: linear-gradient(135deg, rgba(124,58,237,.55), rgba(56,189,248,.45), rgba(34,197,94,.25));
      filter: blur(0px);
      opacity:.55;
      transform: translateZ(18px) rotate(8deg);
      animation: shine 6s ease-in-out infinite alternate;
    }}
    @keyframes shine {{
      from{{ transform: translateZ(18px) rotate(6deg) translateX(-6px); opacity:.45; }}
      to  {{ transform: translateZ(18px) rotate(-6deg) translateX(10px); opacity:.70; }}
    }}

    /* محتوى (Emoji أو PNG) */
    .sticker .icon {{
      font-size: 44px;
      transform: translateZ(26px);
      filter: drop-shadow(0 10px 18px rgba(0,0,0,.35));
    }}
    .sticker img {{
      width: 70px;
      height: 70px;
      object-fit: contain;
      transform: translateZ(26px);
      filter: drop-shadow(0 10px 18px rgba(0,0,0,.35));
    }}

    /* أماكن الملصقات على الأطراف */
    .s1 {{ right: 2.8%; top: 18%; }}
    .s2 {{ right: 5.5%; top: 62%; width:96px; height:96px; }}
    .s3 {{ left:  2.8%; top: 22%; width:112px; height:112px; }}
    .s4 {{ left:  5.5%; top: 68%; width:98px; height:98px; }}

    /* حركة طفو */
    .float1 {{ animation: floaty 7.5s ease-in-out infinite; }}
    .float2 {{ animation: floaty 9.2s ease-in-out infinite; }}
    .float3 {{ animation: floaty 8.4s ease-in-out infinite; }}
    .float4 {{ animation: floaty 10.4s ease-in-out infinite; }}

    @keyframes floaty {{
      0%  {{ transform: translateY(0) rotate(-1deg); }}
      50% {{ transform: translateY(-22px) rotate(2deg); }}
      100%{{ transform: translateY(0) rotate(-1deg); }}
    }}

    /* ====== Responsive ====== */
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .stickers {{ display:none; }}
      .brand img {{ width:86px; height:86px; }}
      .title h1 {{ font-size: 22px; }}
    }}
  </style>
</head>

<body>
  <!-- Parallax layers -->
  <div id="nebula" class="layer nebula"></div>
  <div id="stars1" class="layer stars1"></div>
  <div id="stars2" class="layer stars2"></div>
  <div id="stars3" class="layer stars3"></div>

  <!-- Meteors -->
  <div class="meteor m1"></div>
  <div class="meteor m2"></div>
  <div class="meteor m3"></div>
  <div class="meteor m4"></div>
  <div class="meteor m5"></div>
  <div class="meteor m6"></div>
  <div class="meteor m7"></div>

  <!-- Stickers (Parallax too) -->
  <div class="stickers" id="stickers">
    <div class="sticker s1 float1" data-depth="22">
      <img src="/static/stickers/robot.png" onerror="this.remove(); this.parentElement.querySelector('.icon').style.display='block';" />
      <div class="icon" style="display:none">🤖</div>
    </div>

    <div class="sticker s2 float2" data-depth="18">
      <img src="/static/stickers/search.png" onerror="this.remove(); this.parentElement.querySelector('.icon').style.display='block';" />
      <div class="icon" style="display:none">🔍</div>
    </div>

    <div class="sticker s3 float3" data-depth="20">
      <img src="/static/stickers/book.png" onerror="this.remove(); this.parentElement.querySelector('.icon').style.display='block';" />
      <div class="icon" style="display:none">📚</div>
    </div>

    <div class="sticker s4 float4" data-depth="16">
      <img src="/static/stickers/bolt.png" onerror="this.remove(); this.parentElement.querySelector('.icon').style.display='block';" />
      <div class="icon" style="display:none">⚡</div>
    </div>
  </div>

  <div class="page">
    <div class="container">

      <div class="topbar">
        <div class="brand">
          <img src="/static/logo.png" onerror="this.style.display='none'" alt="logo"/>
          <div class="title">
            <h1>
              <span class="silver">sm</span><span class="gold">AR</span><span class="silver">t</span><span class="gold">ABIC</span>
              <span class="silver"> Agent</span>
            </h1>
            <div class="subtitle">Offline Arabic RAG • E5 embeddings • FAISS retrieval</div>
          </div>
        </div>
        <div class="badge">Final Version – Graduation Submission</div>
      </div>

      <div class="ask">
        <input id="q" type="text" placeholder="اكتب سؤالك هنا… مثال: ما هي عاصمة فرنسا؟" />
        <button onclick="ask()">إرسال</button>
      </div>

      <div class="grid">
        <div class="card">
          <h3>أفضل المقاطع المسترجعة <span class="hint">Top-{TOP_K}</span></h3>
          <div id="chunks" class="box"></div>
        </div>

        <div class="card">
          <h3>الجواب <span class="hint">مع رفض الهلوسة عند ضعف الثقة</span></h3>
          <div id="answer" class="box answer">—</div>
        </div>
      </div>

      <div class="footer">© smARtABIC • يعمل محليًا بدون إنترنت بعد تنزيل النموذج لأول مرة</div>
    </div>
  </div>

<script>
  // ====== Ask API ======
  async function ask() {{
    const q = document.getElementById("q").value.trim();
    if(!q) return;

    const res = await fetch("/ask", {{
      method:"POST",
      headers:{{"Content-Type":"application/json"}},
      body: JSON.stringify({{question:q}})
    }});
    const data = await res.json();

    document.getElementById("answer").textContent = data.answer;

    const chunksDiv = document.getElementById("chunks");
    chunksDiv.innerHTML = "";
    data.chunks.forEach((ch, i) => {{
      const d = document.createElement("div");
      d.className = "chunk";
      d.innerHTML = `<span class="score">score=${{data.scores[i].toFixed(4)}} | مقطع ${{i+1}}</span>${{escapeHtml(ch)}}`;
      chunksDiv.appendChild(d);
    }});
  }}

  function escapeHtml(str){{
    return str.replace(/[&<>"']/g, function(m) {{
      return ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}})[m];
    }});
  }}

  document.getElementById("q").addEventListener("keydown", (e) => {{
    if(e.key === "Enter") ask();
  }});

  // ====== Cinematic Parallax ======
  const nebula = document.getElementById("nebula");
  const s1 = document.getElementById("stars1");
  const s2 = document.getElementById("stars2");
  const s3 = document.getElementById("stars3");
  const stickers = document.getElementById("stickers");

  function setParallax(el, x, y) {{
    el.style.setProperty("--px", x + "px");
    el.style.setProperty("--py", y + "px");
  }}

  window.addEventListener("mousemove", (e) => {{
    const cx = window.innerWidth / 2;
    const cy = window.innerHeight / 2;
    const dx = (e.clientX - cx) / cx;   // -1..1
    const dy = (e.clientY - cy) / cy;

    // الخلفية تتحرك بعمق مختلف
    setParallax(nebula, dx * 18, dy * 12);
    setParallax(s1,     dx * 10, dy * 8);
    setParallax(s2,     dx * 7,  dy * 5);
    setParallax(s3,     dx * 4,  dy * 3);

    // الملصقات: تحريك إضافي + tilt
    const els = stickers.querySelectorAll(".sticker");
    els.forEach(st => {{
      const depth = parseFloat(st.getAttribute("data-depth") || "12");
      const tx = dx * depth;
      const ty = dy * depth;

      // tilt بسيط
      const rx = (-dy * 6).toFixed(2);
      const ry = ( dx * 6).toFixed(2);

      st.style.transform = `translate3d(${{tx}}px, ${{ty}}px, 0) rotateX(${{rx}}deg) rotateY(${{ry}}deg)`;
    }});
  }});
</script>

</body>
</html>
"""


"""
=========================
ملاحظات تشغيل مهمة (قوية):
=========================

1) ضعي الشعار:
   smARtABIC-Agent/data/logo.png

2) (اختياري) ضعي صور PNG شفافة للملصقات:
   smARtABIC-Agent/data/stickers/robot.png
   smARtABIC-Agent/data/stickers/search.png
   smARtABIC-Agent/data/stickers/book.png
   smARtABIC-Agent/data/stickers/bolt.png
   * إذا لم تضعيها سيظهر Emoji تلقائيًا.

3) التشغيل:
   python -m uvicorn app.server:app --host 127.0.0.1 --port 8000

4) Offline؟
   - واجهة الموقع تعمل بدون إنترنت لأنك تفتحين 127.0.0.1 محليًا.
   - لكن نموذج SentenceTransformer قد يحتاج إنترنت "أول مرة فقط" لتنزيله.
     بعد أول تنزيل، يصبح موجود بالكاش ويعمل بدون إنترنت.
"""
