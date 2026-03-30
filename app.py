import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import streamlit.components.v1 as components
import base64
import torch
import io

from gradcam_utils import generate_gradcam
import cv2
from report_utils import detect_bone_region, calculate_fracture_area

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="BoneView AI", layout="wide")

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# LOAD DEPTH MODEL (MiDaS)
# -----------------------------
@st.cache_resource
def load_depth_model():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = transforms.small_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    return midas, transform, device

# -----------------------------
# 3D PIPELINE (trimesh only — no open3d)
# -----------------------------
def estimate_depth(pil_image, midas, transform, device):
    img_rgb = np.array(pil_image.convert("RGB"))
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth, img_rgb


def depth_to_mesh(depth_map, img_rgb, step=5):
    import trimesh
    h, w = depth_map.shape
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    rows, cols = len(ys), len(xs)

    vertices, colors = [], []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            z = float(depth_map[y, x])
            if z < 0.12:
                z = 0.12
            nx =  (x / w) * 2 - 1
            ny = -((y / h) * 2 - 1)
            nz =  z * 0.65
            vertices.append([nx, ny, nz])
            r, g, b = img_rgb[y, x]
            colors.append([r, g, b, 220])

    vertices = np.array(vertices, dtype=np.float32)
    colors   = np.array(colors,   dtype=np.uint8)

    faces = []
    for yi in range(rows - 1):
        for xi in range(cols - 1):
            tl = yi * cols + xi
            tr = tl + 1
            bl = tl + cols
            br = bl + 1
            faces.append([tl, bl, tr])
            faces.append([tr, bl, br])

    faces = np.array(faces, dtype=np.int32)
    mesh  = trimesh.Trimesh(vertices=vertices, faces=faces,
                            vertex_colors=colors, process=False)
    return mesh


def paint_fracture_zones(mesh, fractures):
    if not fractures:
        return mesh
    vertex_colors = mesh.visual.vertex_colors.copy()
    for f in fractures:
        cx     = (f["x1"] + f["x2"]) / 2 * 2 - 1
        cy     = -((f["y1"] + f["y2"]) / 2 * 2 - 1)
        radius = max(f["x2"] - f["x1"], f["y2"] - f["y1"]) * 0.8
        verts  = mesh.vertices
        dist2  = (verts[:, 0] - cx) ** 2 + (verts[:, 1] - cy) ** 2
        vertex_colors[dist2 < radius ** 2] = [226, 75, 74, 255]
    mesh.visual.vertex_colors = vertex_colors
    return mesh


def generate_3d_model(pil_image, fractures):
    midas, transform, device = load_depth_model()
    depth_map, img_rgb = estimate_depth(pil_image, midas, transform, device)
    mesh = depth_to_mesh(depth_map, img_rgb, step=5)
    mesh = paint_fracture_zones(mesh, fractures)
    buf  = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return base64.b64encode(buf.getvalue()).decode()

# -----------------------------
# LOAD GLB FILES
# -----------------------------
def load_glb(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

hero_model = load_glb("model2.glb")
m1 = load_glb("model.glb")
m2 = load_glb("model4.glb")
m3 = load_glb("model3.glb")

# -----------------------------
# GLOBAL STYLING
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0b1120, #020617 65%, #000000);
    color: white;
}
section[data-testid="stSidebar"] {display:none;}
[data-testid="collapsedControl"] {display:none;}
header, footer {visibility:hidden;}
.block-container { max-width: 1260px; padding-top: 1.3rem; padding-bottom: 2rem; }
.navbar {
    position: sticky; top: 0; z-index: 999;
    backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.03);
    padding: 14px 24px; border-radius: 18px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 34px; border: 1px solid rgba(255,255,255,0.06);
}
.nav-logo { font-weight: 700; font-size: 21px; letter-spacing: 0.5px; }
.nav-links { display: flex; gap: 22px; color: #94a3b8; font-size: 14px; }
.nav-links span:hover { color: white; cursor: pointer; }
.hero-title { font-size: 54px; font-weight: 800; line-height: 1.04; color: white; }
.hero-sub { font-size: 20px; color: #cbd5e1; margin-top: 12px; }
.hero-desc { color: #94a3b8; margin-top: 16px; line-height: 1.75; font-size: 15px; max-width: 540px; }
.stButton > button {
    background: linear-gradient(90deg, #22c55e, #4ade80);
    color: black; font-weight: 700; border-radius: 12px; height: 3em; border: none;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05); border-radius: 16px;
    padding: 16px; border: 1px solid rgba(255,255,255,0.05);
}
.section-title {
    font-size: 2.3rem; font-weight: 760; color: white;
    margin-bottom: 0.45rem; line-height: 1.08; margin-top: 0.8rem;
}
.section-sub {
    color: #9ca3af; font-size: 0.98rem; max-width: 740px;
    line-height: 1.72; margin-bottom: 1.5rem;
}
.model-title { font-size: 18px; font-weight: 650; margin-bottom: 8px; color: white; }
.big-space { height: 42px; }
.small-space { height: 18px; }
.viewer-3d-title { font-size: 1.5rem; font-weight: 700; color: white; margin-bottom: 6px; }
.viewer-3d-sub { font-size: 13px; color: #64748b; margin-bottom: 20px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# NAVBAR
# -----------------------------
st.markdown("""
<div class="navbar">
    <div class="nav-logo">🦴 BoneView</div>
    <div class="nav-links">
        <span>Product</span><span>Technology</span>
        <span>Workflow</span><span>Company</span>
    </div>
    <div style="font-weight:600;">🚀 Demo</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# HERO SECTION
# -----------------------------
left, right = st.columns([1, 1.18], gap="large")

with left:
    st.markdown("""
    <div class="hero-title">AI-Powered<br>Fracture Detection</div>
    <div class="hero-sub">Instant diagnosis using computer vision</div>
    <div class="hero-desc">
        BoneView assists doctors with real-time X-ray analysis,
        reducing diagnosis time and improving accuracy with an
        intelligent AI workflow built for radiology.
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "png", "jpeg"])

with right:
    components.html(f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <model-viewer
        src="data:model/gltf-binary;base64,{hero_model}"
        auto-rotate camera-controls exposure="1" shadow-intensity="1"
        style="width:100%;height:450px;
               filter:drop-shadow(0 0 25px rgba(34,197,94,0.35));
               background:transparent;">
    </model-viewer>
    """, height=470)

# -----------------------------
# DETECTION SECTION
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file)

    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Running AI Model..."):
            with torch.no_grad():
                results = model(np.array(image))

        result_img = results[0].plot()
        st.markdown('<div class="big-space"></div>', unsafe_allow_html=True)

        cv_image    = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gradcam_img = generate_gradcam(cv_image, results)

        st.markdown("## 📊 Visual Analysis Comparison")
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.markdown("### Original")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("### Detection")
            st.image(result_img, use_container_width=True)
        with col3:
            st.markdown("### 🔥 Heatmap")
            st.image(gradcam_img, use_container_width=True)

        st.markdown("## Analysis")

        fractures_for_3d = []
        severity       = "Mild"
        recommendation = "Minor fracture suspected. Rest and monitoring recommended."
        precautions    = "Avoid strain. Apply ice. Monitor swelling."

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            st.error("⚠️ Fracture Detected")
            try:
                conf = results[0].boxes.conf
                if hasattr(conf, "cpu"):
                    conf = conf.cpu().numpy()
                score = round(float(np.max(conf)) * 100, 2) if not isinstance(conf, (float, int, np.integer)) else round(float(conf) * 100, 2)

                if score > 80:
                    severity       = "Severe"
                    recommendation = "Immediate medical attention required. Possible surgery or casting."
                    precautions    = "Avoid movement. Immobilize immediately. Seek emergency care."
                elif score > 50:
                    severity       = "Moderate"
                    recommendation = "Consult an orthopedic specialist. Likely requires casting."
                    precautions    = "Limit movement. Use support (splint). Avoid pressure on affected area."

                boxes = results[0].boxes.xyxy
                if hasattr(boxes, "cpu"):
                    boxes = boxes.cpu().numpy()
                box    = boxes[0]
                region = detect_bone_region(box)
                area   = calculate_fracture_area(box, cv_image.shape)
                img_h, img_w = cv_image.shape[:2]

                for b in boxes:
                    fractures_for_3d.append({
                        "x1": float(b[0]) / img_w,
                        "y1": float(b[1]) / img_h,
                        "x2": float(b[2]) / img_w,
                        "y2": float(b[3]) / img_h,
                        "confidence": float(score / 100),
                        "type": "Fracture",
                        "region": region,
                        "severity": severity,
                    })

            except Exception as e:
                st.warning(f"Processing issue: {e}")
                score  = 0
                region = "Unknown"
                area   = 0

            st.metric("Confidence", f"{score}%")
            st.progress(int(score))
            st.write(f"🦴 Affected Region: {region}")
            st.write(f"📏 Fracture Area: {area}%")

        else:
            st.success("✅ No Fracture Detected")

        st.write(f"⚠️ Severity: {severity}")
        st.write(f"💊 Recommendation: {recommendation}")
        st.write(f"🛑 Precautions: {precautions}")

        # ── 3D MODEL VIEWER ───────────────────────────────────────────────────
        st.markdown('<div class="big-space"></div>', unsafe_allow_html=True)
        st.markdown('<div class="viewer-3d-title">🧊 3D Bone Model</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="viewer-3d-sub">'
            'A 3D model reconstructed from your X-ray using depth estimation. '
            'The <span style="color:#f09595">red zone</span> shows the detected fracture region. '
            'Rotate and zoom to explore. '
            '<span style="color:#475569">Educational estimate — not a clinical scan.</span>'
            '</div>',
            unsafe_allow_html=True
        )

        col_left, col_right = st.columns([1.6, 1], gap="large")

        with col_left:
            with st.spinner("Generating 3D model... (~20–40 seconds)"):
                try:
                    glb_b64 = generate_3d_model(image, fractures_for_3d)
                    components.html(f"""
                    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
                    <model-viewer
                        src="data:model/gltf-binary;base64,{glb_b64}"
                        auto-rotate camera-controls
                        exposure="1.2" shadow-intensity="0.8"
                        style="width:100%;height:420px;
                               background:rgba(8,11,16,0.95);
                               border-radius:18px;
                               border:1px solid rgba(255,255,255,0.06);">
                    </model-viewer>
                    """, height=440)
                    st.download_button(
                        label="⬇️ Download 3D model (.glb)",
                        data=base64.b64decode(glb_b64),
                        file_name="bone_fracture_model.glb",
                        mime="model/gltf-binary",
                    )
                except Exception as e:
                    st.error(f"3D generation failed: {e}")
                    st.info("Run: pip install trimesh")

        with col_right:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:18px;padding:22px;">
                <div style="font-size:13px;font-weight:600;color:#64748b;
                            letter-spacing:.06em;text-transform:uppercase;margin-bottom:14px;">
                    Model info
                </div>
                <div style="display:flex;flex-direction:column;gap:0;">
                    <div style="display:flex;justify-content:space-between;font-size:13px;
                                padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span style="color:#64748b">Method</span>
                        <span style="color:#e2e8f0;font-weight:500">MiDaS depth</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:13px;
                                padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span style="color:#64748b">Mesh</span>
                        <span style="color:#e2e8f0;font-weight:500">Grid triangulation</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:13px;
                                padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span style="color:#64748b">Fracture zone</span>
                        <span style="color:#f09595;font-weight:500">Red overlay</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:13px;
                                padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span style="color:#64748b">Format</span>
                        <span style="color:#e2e8f0;font-weight:500">.glb</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:13px;padding:8px 0;">
                        <span style="color:#64748b">Purpose</span>
                        <span style="color:#e2e8f0;font-weight:500">Patient education</span>
                    </div>
                </div>
                <div style="margin-top:20px;padding:14px;border-radius:12px;
                            background:rgba(226,75,74,0.08);border:1px solid rgba(226,75,74,0.2);">
                    <div style="font-size:11px;color:#f09595;font-weight:500;margin-bottom:6px;">
                        Clinical disclaimer
                    </div>
                    <div style="font-size:11px;color:#64748b;line-height:1.6;">
                        This 3D model is an AI estimate from a 2D image.
                        Not a substitute for a CT scan or professional diagnosis.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# BONE CONDITION VISUALS
# -----------------------------
st.markdown('<div class="big-space"></div>', unsafe_allow_html=True)
st.markdown("## Bone Condition Visuals")

def render_model(model):
    return f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <model-viewer
        src="data:model/gltf-binary;base64,{model}"
        auto-rotate camera-controls exposure="1" shadow-intensity="1"
        style="width:100%;height:270px;background:transparent;">
    </model-viewer>
    """

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.markdown('<div class="model-title">Normal</div>', unsafe_allow_html=True)
    components.html(render_model(m1), height=290)
with col2:
    st.markdown('<div class="model-title">Fracture</div>', unsafe_allow_html=True)
    components.html(render_model(m2), height=290)
with col3:
    st.markdown('<div class="model-title">Severe</div>', unsafe_allow_html=True)
    components.html(render_model(m3), height=290)

# -----------------------------
# REVEAL ANIMATION WRAPPER
# -----------------------------
def reveal_wrapper(inner_html, height=340):
    return f"""
    <html><head><style>
    html,body{{margin:0;padding:0;background:transparent;overflow:hidden;font-family:Inter,sans-serif;}}
    .reveal{{opacity:0;transform:translateY(45px);transition:all 0.9s cubic-bezier(0.22,1,0.36,1);will-change:transform,opacity;}}
    .reveal.show{{opacity:1;transform:translateY(0px);}}
    </style></head>
    <body>
        <div class="reveal" id="revealBox">{inner_html}</div>
        <script>
        const box=document.getElementById("revealBox");
        const observer=new IntersectionObserver((entries)=>{{
            entries.forEach(entry=>{{if(entry.isIntersecting)box.classList.add("show");}});
        }},{{threshold:0.15}});
        observer.observe(box);
        </script>
    </body></html>
    """

# -----------------------------
# FEATURE BLOCKS
# -----------------------------
def assist_radiologists_block():
    return """
    <style>
    body{margin:0;background:transparent;font-family:Inter,sans-serif;}
    .wrap{display:grid;grid-template-columns:1.25fr 0.95fr;gap:28px;align-items:center;min-height:300px;}
    .badge{display:inline-block;padding:8px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);color:#f4f4f5;font-size:14px;margin-bottom:22px;}
    .title{font-size:2.15rem;font-weight:760;line-height:1.08;color:white;margin-bottom:16px;}
    .desc{font-size:1rem;color:#b3b3bb;line-height:1.7;max-width:520px;margin-bottom:24px;}
    .chips{display:flex;gap:12px;flex-wrap:wrap;}
    .chip{padding:12px 16px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.03);color:#f4f4f5;font-size:13px;}
    .demo-card{height:270px;border-radius:26px;background:linear-gradient(180deg,rgba(12,12,14,0.98),rgba(8,8,10,1));border:1px solid rgba(255,255,255,0.06);position:relative;overflow:hidden;padding:26px;box-sizing:border-box;}
    .orb{width:74px;height:74px;border-radius:50%;margin:0 auto 16px auto;background:radial-gradient(circle at 35% 35%,#ff4df8,#8b2cf5 58%,#31103f 100%);box-shadow:0 0 30px rgba(217,70,239,0.45);position:relative;animation:floatOrb 3.5s ease-in-out infinite;}
    .orb::before,.orb::after{content:"";position:absolute;inset:-8px;border-radius:50%;border:3px solid rgba(217,70,239,0.18);animation:rotateRing 5s linear infinite;}
    .orb::after{inset:-14px;border-color:rgba(168,85,247,0.12);animation-direction:reverse;animation-duration:7s;}
    .demo-title{text-align:center;color:white;font-size:1.7rem;font-weight:700;margin-bottom:8px;}
    .demo-sub{text-align:center;color:#a1a1aa;font-size:13px;margin-bottom:22px;line-height:1.45;}
    .input-box{border:1px solid rgba(255,255,255,0.06);border-radius:14px;height:72px;background:rgba(0,0,0,0.28);position:relative;padding:14px 16px;box-sizing:border-box;}
    .typing{color:#d946ef;font-size:16px;white-space:nowrap;overflow:hidden;border-right:2px solid #d946ef;width:0;animation:typing 3s steps(12) infinite alternate,blink 0.8s infinite;}
    .upload-pill{margin-top:14px;display:inline-block;padding:8px 14px;border-radius:999px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);color:#a1a1aa;font-size:13px;}
    .bottom-actions{position:absolute;bottom:18px;left:0;width:100%;display:flex;justify-content:center;gap:16px;}
    .action{padding:8px 12px;border-radius:12px;border:1px solid rgba(217,70,239,0.18);color:#d946ef;font-size:13px;background:rgba(217,70,239,0.04);animation:pulseBtn 2s infinite ease-in-out;}
    @keyframes typing{from{width:0;}to{width:32px;}}
    @keyframes blink{50%{border-color:transparent;}}
    @keyframes rotateRing{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
    @keyframes floatOrb{0%,100%{transform:translateY(0px) scale(1);}50%{transform:translateY(-6px) scale(1.03);}}
    @keyframes pulseBtn{0%,100%{transform:translateY(0px);opacity:0.9;}50%{transform:translateY(-2px);opacity:1;}}
    </style>
    <div class="wrap">
        <div>
            <div class="badge">Diagnostic Assistant</div>
            <div class="title">Assist Radiologists with AI</div>
            <div class="desc">Flag potential fracture zones and generate visual overlays to assist radiologists during diagnosis.</div>
            <div class="chips"><div class="chip">Fracture Map</div><div class="chip">Visual Overlay</div><div class="chip">Annotation</div></div>
        </div>
        <div class="demo-card">
            <div class="orb"></div>
            <div class="demo-title">Try BoneScan X</div>
            <div class="demo-sub">Upload an X-ray to see how our AI highlights fractures in seconds.</div>
            <div class="input-box"><div class="typing">AI</div><div class="upload-pill">+ Add X-Ray Image</div></div>
            <div class="bottom-actions"><div class="action">Analyze</div><div class="action">Make Report</div></div>
        </div>
    </div>
    """

def remote_diagnosis_block():
    return """
    <style>
    body{margin:0;background:transparent;font-family:Inter,sans-serif;}
    .wrap{display:grid;grid-template-columns:0.95fr 1.15fr;gap:36px;align-items:center;min-height:320px;}
    .screen{width:100%;height:300px;border-radius:26px;background:linear-gradient(180deg,rgba(10,10,12,0.98),rgba(6,6,8,1));border:1px solid rgba(255,255,255,0.06);position:relative;overflow:hidden;padding:18px;box-sizing:border-box;}
    .screen-inner{position:absolute;inset:18px;border-radius:20px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);padding:14px;box-sizing:border-box;}
    .header{height:38px;border-radius:10px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);display:flex;align-items:center;padding:0 12px;color:#f4f4f5;font-size:14px;font-weight:600;margin-bottom:12px;}
    .tags{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;}
    .tag{padding:6px 10px;border-radius:999px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.05);font-size:11px;color:#d4d4d8;}
    .patient-row{display:flex;justify-content:space-between;color:#9ca3af;font-size:12px;margin-bottom:10px;}
    .scan-box{position:relative;height:130px;border-radius:16px;background:radial-gradient(circle at center,rgba(217,70,239,0.10),rgba(0,0,0,0.55));overflow:hidden;display:flex;align-items:center;justify-content:center;margin-top:12px;}
    .xray{width:120px;height:120px;opacity:0.22;border-radius:12px;background:linear-gradient(90deg,transparent 48%,rgba(255,255,255,0.16) 50%,transparent 52%),linear-gradient(0deg,transparent 48%,rgba(255,255,255,0.10) 50%,transparent 52%);background-size:24px 24px;animation:drift 4s ease-in-out infinite;}
    .detect-box{position:absolute;width:70px;height:26px;border:2px solid #d946ef;top:34px;left:50%;transform:translateX(-20%);box-shadow:0 0 16px rgba(217,70,239,0.4);animation:scanMove 2.8s ease-in-out infinite alternate;}
    .timeline{position:absolute;bottom:12px;left:18px;right:18px;display:flex;justify-content:space-between;align-items:center;}
    .line{position:absolute;left:18px;right:18px;bottom:26px;height:2px;background:rgba(217,70,239,0.18);}
    .dot{width:12px;height:12px;border-radius:50%;background:#7e22ce;box-shadow:0 0 14px rgba(217,70,239,0.35);animation:pulseDot 2s infinite ease-in-out;}
    .tl-label{font-size:11px;color:#a1a1aa;margin-top:8px;}
    .right-content .badge{display:inline-block;padding:8px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);color:#f4f4f5;font-size:14px;margin-bottom:22px;}
    .right-content .title{font-size:2.15rem;font-weight:760;line-height:1.08;color:white;margin-bottom:16px;}
    .right-content .desc{font-size:1rem;color:#b3b3bb;line-height:1.7;max-width:520px;margin-bottom:24px;}
    .chips{display:flex;gap:12px;flex-wrap:wrap;}
    .chip{padding:12px 16px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.03);color:#f4f4f5;font-size:13px;}
    @keyframes scanMove{0%{transform:translateX(-30%) translateY(0px);}100%{transform:translateX(5%) translateY(12px);}}
    @keyframes drift{0%,100%{transform:translateY(0px);}50%{transform:translateY(-4px);}}
    @keyframes pulseDot{0%,100%{transform:scale(1);opacity:0.8;}50%{transform:scale(1.18);opacity:1;}}
    </style>
    <div class="wrap">
        <div class="screen">
            <div class="screen-inner">
                <div class="header">Remote X-ray Scan Received</div>
                <div class="tags"><div class="tag">X-ray</div><div class="tag">Rural Clinic</div><div class="tag">Help Radiologist</div></div>
                <div class="patient-row"><div>Patient: 19 yrs</div><div>AI Verified</div></div>
                <div class="patient-row"><div>Region: Rural Clinic</div><div>Suspected Injury: Shoulder Fracture</div></div>
                <div class="scan-box"><div class="xray"></div><div class="detect-box"></div></div>
                <div class="line"></div>
                <div class="timeline">
                    <div style="text-align:center;"><div class="dot"></div><div class="tl-label">Uploaded</div></div>
                    <div style="text-align:center;"><div class="dot"></div><div class="tl-label">Detected</div></div>
                    <div style="text-align:center;"><div class="dot"></div><div class="tl-label">Overlay</div></div>
                </div>
            </div>
        </div>
        <div class="right-content">
            <div class="badge">Patient-Centric AI</div>
            <div class="title">Support Remote Diagnosis</div>
            <div class="desc">Enable healthcare workers in rural or underserved areas to make accurate decisions without a radiologist present.</div>
            <div class="chips"><div class="chip">Telemedicine</div><div class="chip">Offline Access</div><div class="chip">Learn More</div></div>
        </div>
    </div>
    """

st.markdown('<div class="big-space"></div>', unsafe_allow_html=True)
components.html(reveal_wrapper(assist_radiologists_block(), 340), height=340)
st.markdown('<div class="small-space"></div>', unsafe_allow_html=True)
components.html(reveal_wrapper(remote_diagnosis_block(), 360), height=360)

# -----------------------------
# WORKFLOW SECTION
# -----------------------------
st.markdown('<div class="big-space"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">How Our AI Workflow Operates</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">A premium animated workflow section designed in a modern product style — combining medical intelligence, model execution, deployment, and real-time insights.</div>', unsafe_allow_html=True)

def card_wrapper(step, title, desc, inner_html):
    return f"""
    <style>
    body{{margin:0;padding:0;background:transparent;font-family:Inter,sans-serif;overflow:hidden;}}
    .feature-card{{position:relative;height:390px;border-radius:22px;background:linear-gradient(180deg,rgba(12,12,14,0.96) 0%,rgba(8,8,10,0.98) 100%);border:1px solid rgba(255,255,255,0.08);overflow:hidden;padding:24px;box-sizing:border-box;}}
    .step-badge{{display:inline-block;padding:7px 12px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);color:#e4e4e7;font-size:0.82rem;font-weight:500;margin-bottom:16px;}}
    .card-title{{font-size:1.55rem;font-weight:700;color:white;margin-bottom:10px;line-height:1.18;}}
    .card-desc{{font-size:0.91rem;color:#b3b3bb;line-height:1.6;font-style:italic;max-width:95%;margin-bottom:18px;}}
    .visual-box{{position:absolute;left:24px;right:24px;bottom:24px;height:165px;border:1px solid rgba(255,255,255,0.06);background:rgba(0,0,0,0.32);overflow:hidden;border-radius:14px;}}
    </style>
    <div class="feature-card">
        <div class="step-badge">{step}</div>
        <div class="card-title">{title}</div>
        <div class="card-desc">{desc}</div>
        <div class="visual-box">{inner_html}</div>
    </div>
    """

def medical_data_card():
    inner = """
    <style>
    .radar-layout{display:flex;height:100%;width:100%;}
    .radar-left{width:48%;display:flex;align-items:center;justify-content:center;position:relative;}
    .radar{position:relative;width:112px;height:112px;border-radius:50%;border:1px solid rgba(255,255,255,0.05);background:radial-gradient(circle,rgba(168,85,247,0.06) 0%,transparent 70%);overflow:hidden;}
    .radar::before,.radar::after{content:"";position:absolute;border-radius:50%;border:1px solid rgba(255,255,255,0.04);}
    .radar::before{inset:18px;} .radar::after{inset:36px;}
    .center-dot{position:absolute;width:8px;height:8px;background:#a855f7;border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);box-shadow:0 0 18px rgba(168,85,247,0.9);animation:pulseDot 2s infinite ease-in-out;}
    .sweep{position:absolute;width:50%;height:50%;top:0;left:50%;transform-origin:bottom left;background:linear-gradient(135deg,rgba(168,85,247,0.42),transparent 76%);animation:rotateSweep 3.5s linear infinite;}
    .scan-text{position:absolute;bottom:12px;text-align:center;color:#e4e4e7;font-size:10px;width:100%;line-height:1.25;}
    .radar-right{width:52%;padding:10px 10px 10px 0;display:flex;flex-direction:column;gap:7px;justify-content:center;}
    .status-item{border:1px solid rgba(255,255,255,0.07);background:rgba(255,255,255,0.03);color:#f4f4f5;padding:7px 9px;border-radius:8px;font-size:10px;font-weight:500;display:flex;align-items:center;gap:8px;}
    .icon-box{width:12px;height:12px;border:1px solid rgba(255,255,255,0.15);border-radius:4px;display:inline-block;flex-shrink:0;}
    @keyframes rotateSweep{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
    @keyframes pulseDot{0%,100%{transform:translate(-50%,-50%) scale(1);}50%{transform:translate(-50%,-50%) scale(1.2);}}
    </style>
    <div class="radar-layout">
        <div class="radar-left">
            <div class="radar"><div class="sweep"></div><div class="center-dot"></div></div>
            <div class="scan-text">Scanning patient X-rays<br>for AI insights...</div>
        </div>
        <div class="radar-right">
            <div class="status-item"><span class="icon-box"></span> X-ray Imported</div>
            <div class="status-item"><span class="icon-box"></span> Case Analyzed</div>
            <div class="status-item"><span class="icon-box"></span> Detection Started</div>
            <div class="status-item"><span class="icon-box"></span> Overlay Created</div>
            <div class="status-item"><span class="icon-box"></span> Report Ready</div>
        </div>
    </div>
    """
    return card_wrapper("Step 1","Medical Data Assessment","We analyze medical and clinical data to identify diagnostic needs and define AI goals.",inner)

def ai_dev_card():
    inner = """
    <style>
    .code-window{width:100%;height:100%;background:#0b0b0d;display:flex;flex-direction:column;}
    .topbar{height:26px;background:rgba(255,255,255,0.04);display:flex;align-items:center;justify-content:space-between;padding:0 10px;color:#9ca3af;font-size:11px;}
    .code-body{flex:1;display:flex;}
    .sidebar{width:34px;border-right:1px solid rgba(255,255,255,0.05);display:flex;flex-direction:column;align-items:center;gap:12px;padding-top:12px;color:#9ca3af;font-size:13px;}
    .editor{flex:1;padding:14px 16px;color:white;font-family:monospace;font-size:10px;line-height:1.7;}
    .pink{color:#d946ef;} .gray{color:#9ca3af;} .green{color:#d4d4d8;}
    .cursor{display:inline-block;width:6px;height:12px;background:#d946ef;margin-left:4px;animation:blink 1s infinite;vertical-align:middle;}
    @keyframes blink{50%{opacity:0;}}
    </style>
    <div class="code-window">
        <div class="topbar"><span>left right</span><span style="opacity:0.35;">----------</span><span>min max close</span></div>
        <div class="code-body">
            <div class="sidebar"><div>doc</div><div>search</div><div>ext</div></div>
            <div class="editor">
                <span class="pink">def __init__</span><span class="gray">(self, backbone='YOLOv8'):</span><br>
                &nbsp;&nbsp;self.backbone = backbone<br>
                &nbsp;&nbsp;self.explainability = "GradCAM"<br><br>
                <span class="pink">def detect_fracture</span><span class="gray">(self, xray_image):</span><br>
                &nbsp;&nbsp;<span class="green"># Model logic here</span><br>
                &nbsp;&nbsp;return {"label":"Fracture","conf":0.94}<span class="cursor"></span>
            </div>
        </div>
    </div>
    """
    return card_wrapper("Step 2","AI Development","We build deep learning models for fracture detection using custom YOLOv8 and Grad-CAM.",inner)

def clinical_card():
    inner = """
    <style>
    .flow-wrap{width:100%;height:100%;position:relative;}
    .node{position:absolute;width:58px;height:58px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.03);display:flex;align-items:center;justify-content:center;color:#f4f4f5;font-size:9px;flex-direction:column;}
    .left-node{left:50px;top:52px;} .right-node{right:50px;top:52px;}
    .left-core,.right-core{width:24px;height:24px;border-radius:50%;background:radial-gradient(circle,#d946ef 0%,#7e22ce 100%);box-shadow:0 0 18px rgba(217,70,239,0.5);margin-bottom:7px;animation:pulseCore 2s infinite ease-in-out;}
    .right-core{clip-path:polygon(0 0,100% 0,55% 50%,100% 100%,0 100%);border-radius:0;}
    .line{position:absolute;height:2px;background:linear-gradient(90deg,transparent,rgba(168,85,247,0.8),transparent);background-size:200% 100%;animation:flow 2s linear infinite;}
    .l1{width:130px;left:110px;top:78px;} .l2{width:130px;left:110px;top:92px;} .l3{width:130px;left:110px;top:106px;}
    .label{position:absolute;bottom:-20px;color:#e4e4e7;font-size:9px;width:80px;text-align:center;}
    @keyframes flow{0%{background-position:200% 0;}100%{background-position:-200% 0;}}
    @keyframes pulseCore{0%,100%{transform:scale(1);opacity:0.9;}50%{transform:scale(1.12);opacity:1;}}
    </style>
    <div class="flow-wrap">
        <div class="line l1"></div><div class="line l2"></div><div class="line l3"></div>
        <div class="node left-node"><div class="left-core"></div><div class="label">BoneView</div></div>
        <div class="node right-node"><div class="right-core"></div><div class="label">Hospital AI</div></div>
    </div>
    """
    return card_wrapper("Step 3","Clinical Integration","We integrate AI into radiology workflows and ensure compatibility with hospital systems.",inner)

def insights_card():
    inner = """
    <style>
    .status-wrap{width:100%;height:100%;padding:12px;box-sizing:border-box;display:flex;flex-direction:column;gap:9px;justify-content:center;}
    .status-card{border:1px solid rgba(255,255,255,0.07);background:rgba(255,255,255,0.02);border-radius:10px;padding:10px 12px;display:flex;justify-content:space-between;align-items:center;color:white;}
    .left-side{display:flex;align-items:center;gap:10px;}
    .mini-icon{width:22px;height:22px;border-radius:8px;border:1px solid rgba(255,255,255,0.08);display:flex;align-items:center;justify-content:center;color:#d4d4d8;font-size:11px;}
    .main{font-size:11px;font-weight:600;color:#f4f4f5;} .sub{font-size:10px;color:#a1a1aa;margin-top:2px;}
    .circle-loader{width:13px;height:13px;border-radius:50%;border:3px solid rgba(217,70,239,0.15);border-top-color:#d946ef;animation:spin 1s linear infinite;}
    .up-arrow{color:#d946ef;font-size:15px;animation:floatUp 1.4s infinite ease-in-out;}
    .tick{color:#d946ef;font-size:17px;}
    @keyframes spin{to{transform:rotate(360deg);}}
    @keyframes floatUp{0%,100%{transform:translateY(0);}50%{transform:translateY(-3px);}}
    </style>
    <div class="status-wrap">
        <div class="status-card"><div class="left-side"><div class="mini-icon">chat</div><div><div class="main">Chatbot system</div><div class="sub">Efficiency will increase by 20%</div></div></div><div class="circle-loader"></div></div>
        <div class="status-card"><div class="left-side"><div class="mini-icon">cfg</div><div><div class="main">Workflow system</div><div class="sub">Update available.</div></div></div><div class="up-arrow">up</div></div>
        <div class="status-card"><div class="left-side"><div class="mini-icon">lab</div><div><div class="main">Sales system</div><div class="sub">Up to date</div></div></div><div class="tick">done</div></div>
    </div>
    """
    return card_wrapper("Step 4","Real-Time Insights & Feedback","We monitor AI in real-world use, collect feedback, and optimize for higher accuracy.",inner)

col1, col2 = st.columns(2, gap="large")
with col1:
    components.html(reveal_wrapper(medical_data_card(), 390), height=390)
with col2:
    components.html(reveal_wrapper(ai_dev_card(), 390), height=390)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="large")
with col3:
    components.html(reveal_wrapper(clinical_card(), 390), height=390)
with col4:
    components.html(reveal_wrapper(insights_card(), 390), height=390)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown('<div class="big-space"></div>', unsafe_allow_html=True)
st.markdown("---")
st.caption("Built with AI • Computer Vision • Healthcare Innovation")