import streamlit as st

from PIL import Image
from blip_inference import generate_blip_caption
import os
import tempfile

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="VisionAI — AI Image Caption Generator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "caption" not in st.session_state:
    st.session_state.caption = None
if "uploaded_image_data" not in st.session_state:
    st.session_state.uploaded_image_data = None
if "image_filename" not in st.session_state:
    st.session_state.image_filename = None
if "image_dimensions" not in st.session_state:
    st.session_state.image_dimensions = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INJECT GLOBAL CSS + ANIMATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def inject_global_styles():
    st.markdown(
        """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
  .orb { display: none !important; }
}
:root {
  --bg-primary: #07060e;
  --bg-secondary: #0d0b18;
  --bg-card: rgba(15, 13, 28, 0.65);
  --bg-glass: rgba(20, 17, 40, 0.45);
  --border-glass: rgba(120, 90, 220, 0.15);
  --border-glow: rgba(120, 90, 220, 0.3);
  --text-primary: #f0eef6;
  --text-secondary: #9892b0;
  --text-muted: #5f587a;
  --accent-purple: #8b5cf6;
  --accent-blue: #6366f1;
  --accent-violet: #a78bfa;
  --accent-indigo: #818cf8;
  --gradient-main: linear-gradient(135deg, #8b5cf6, #6366f1, #a78bfa);
  --gradient-subtle: linear-gradient(135deg, rgba(139,92,246,0.12), rgba(99,102,241,0.08));
  --shadow-glow: 0 0 30px rgba(139,92,246,0.08);
  --radius-lg: 20px;
  --radius-md: 14px;
  --radius-sm: 10px;
}
.stApp {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp > header { background: transparent !important; }
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.4); }
.block-container {
  padding-top: 0 !important;
  padding-bottom: 2rem !important;
  max-width: 1200px !important;
}
h1, h2, h3, h4, h5, h6, p, span, div, label, li {
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stFileUploader"] > label { display: none !important; }
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploader"] section {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
[data-testid="stFileUploader"] section > div { display: none !important; }
[data-testid="stFileUploader"] section > button {
  background: var(--gradient-main) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  padding: 0.55rem 1.4rem !important;
  cursor: pointer !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  box-shadow: 0 4px 15px rgba(139,92,246,0.25) !important;
}
[data-testid="stFileUploader"] section > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(139,92,246,0.35) !important;
}
[data-testid="stFileUploader"] small { color: var(--text-muted) !important; font-size: 0.7rem !important; }
[data-testid="stFileUploaderFileName"] { color: var(--text-secondary) !important; }
.stButton > button {
  background: var(--bg-glass) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  padding: 0.5rem 1.2rem !important;
  cursor: pointer !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
}
.stButton > button:hover {
  border-color: var(--accent-purple) !important;
  box-shadow: 0 0 20px rgba(139,92,246,0.15) !important;
  transform: translateY(-1px) !important;
  background: rgba(139,92,246,0.1) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.primary-btn > button {
  background: var(--gradient-main) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 15px rgba(139,92,246,0.25) !important;
}
.primary-btn > button:hover {
  box-shadow: 0 6px 25px rgba(139,92,246,0.35) !important;
  background: var(--gradient-main) !important;
}
.stSpinner > div { border-top-color: var(--accent-purple) !important; }
.stSpinner > div > span { color: var(--text-secondary) !important; }
[data-testid="stImage"] img {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border-glass) !important;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stImage"] img:hover {
  border-color: var(--border-glow) !important;
  box-shadow: 0 8px 30px rgba(139,92,246,0.12) !important;
  transform: scale(1.005) !important;
}
[data-testid="stHorizontalBlock"] { gap: 2rem !important; }
.stDownloadButton > button {
  background: var(--bg-glass) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  backdrop-filter: blur(12px) !important;
}
.stDownloadButton > button:hover {
  border-color: var(--accent-purple) !important;
  box-shadow: 0 0 20px rgba(139,92,246,0.15) !important;
  transform: translateY(-1px) !important;
}
[data-testid="stAlert"] {
  background: var(--bg-glass) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(24px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes gradientShift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 8px rgba(139,92,246,0.2); }
  50% { box-shadow: 0 0 16px rgba(139,92,246,0.4); }
}
@keyframes orbFloat {
  0%, 100% { transform: translate(0, 0) scale(1); }
  25% { transform: translate(30px, -20px) scale(1.05); }
  50% { transform: translate(-15px, -35px) scale(0.97); }
  75% { transform: translate(-25px, 10px) scale(1.03); }
}
@keyframes orbFloat2 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(-25px, 20px) scale(1.04); }
  66% { transform: translate(20px, -15px) scale(0.96); }
}
@keyframes dotPulse {
  0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
  40% { opacity: 1; transform: scale(1); }
}
@keyframes uploadPulse {
  0%, 100% { border-color: rgba(139,92,246,0.2); }
  50% { border-color: rgba(139,92,246,0.5); }
}
@keyframes slideInRight {
  from { opacity: 0; transform: translateX(20px); }
  to { opacity: 1; transform: translateX(0); }
}
</style>""",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKGROUND ORBS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_background_orbs():
    st.markdown(
        '<div style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;overflow:hidden;">'
        '<div class="orb" style="position:absolute;top:8%;left:12%;width:350px;height:350px;background:radial-gradient(circle,rgba(139,92,246,0.06) 0%,transparent 70%);border-radius:50%;filter:blur(60px);animation:orbFloat 20s ease-in-out infinite;"></div>'
        '<div class="orb" style="position:absolute;top:55%;right:8%;width:300px;height:300px;background:radial-gradient(circle,rgba(99,102,241,0.05) 0%,transparent 70%);border-radius:50%;filter:blur(55px);animation:orbFloat2 25s ease-in-out infinite;"></div>'
        '<div class="orb" style="position:absolute;bottom:10%;left:40%;width:250px;height:250px;background:radial-gradient(circle,rgba(167,139,250,0.04) 0%,transparent 70%);border-radius:50%;filter:blur(50px);animation:orbFloat 30s ease-in-out infinite reverse;"></div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NAVIGATION / HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_header():
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;padding:1.1rem 0 0.8rem 0;border-bottom:1px solid rgba(120,90,220,0.08);margin-bottom:1rem;animation:fadeIn 0.8s ease;position:relative;z-index:10;">'
        '<div style="display:flex;align-items:center;gap:12px;">'
        '<div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#8b5cf6,#6366f1);display:flex;align-items:center;justify-content:center;font-size:1rem;color:white;font-weight:700;box-shadow:0 2px 12px rgba(139,92,246,0.3);font-family:Space Grotesk,sans-serif;">V</div>'
        '<span style="font-family:Space Grotesk,sans-serif;font-size:1.15rem;font-weight:700;color:#f0eef6;letter-spacing:-0.3px;">VisionAI</span>'
        '<span style="font-size:0.6rem;font-weight:600;background:rgba(139,92,246,0.12);color:#a78bfa;padding:3px 8px;border-radius:6px;letter-spacing:0.5px;text-transform:uppercase;">Beta</span>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:16px;">'
        '<div style="display:flex;align-items:center;gap:6px;background:rgba(20,17,40,0.45);border:1px solid rgba(120,90,220,0.15);border-radius:8px;padding:5px 12px;backdrop-filter:blur(12px);">'
        '<div style="width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 8px rgba(52,211,153,0.4);animation:pulseGlow 3s ease-in-out infinite;"></div>'
        '<span style="font-size:0.72rem;color:#9892b0;font-weight:500;">BLIP Captioning</span>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HERO SECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_hero():
    # AI Badge
    st.markdown(
        '<div style="text-align:center;padding:3.5rem 0 0 0;position:relative;z-index:10;animation:fadeInUp 0.9s cubic-bezier(0.16,1,0.3,1);">'
        '<div style="display:inline-flex;align-items:center;gap:6px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.18);border-radius:20px;padding:6px 16px;margin-bottom:1.4rem;animation:pulseGlow 4s ease-in-out infinite;">'
        '<span style="font-size:0.8rem;">&#10022;</span>'
        '<span style="font-size:0.72rem;font-weight:600;color:#a78bfa;letter-spacing:0.8px;text-transform:uppercase;">Powered by Salesforce BLIP</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    # Headline
    st.markdown(
        '<div style="text-align:center;position:relative;z-index:10;animation:fadeInUp 0.9s cubic-bezier(0.16,1,0.3,1) 0.1s both;">'
        '<div style="font-family:Space Grotesk,sans-serif;font-size:clamp(2.2rem,5vw,3.4rem);font-weight:800;line-height:1.1;margin:0 0 0.8rem 0;background:linear-gradient(135deg,#f0eef6 0%,#a78bfa 50%,#818cf8 100%);background-size:200% 200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:gradientShift 6s ease infinite;letter-spacing:-1px;">'
        'Every Picture Has a Story.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    # Subheadline
    st.markdown(
        '<div style="text-align:center;position:relative;z-index:10;animation:fadeInUp 0.9s cubic-bezier(0.16,1,0.3,1) 0.15s both;">'
        '<p style="font-size:clamp(0.95rem,2vw,1.15rem);color:#9892b0;font-weight:400;max-width:520px;margin:0 auto 1.6rem auto;line-height:1.6;letter-spacing:-0.1px;">Turn visual moments into intelligent descriptions with AI.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    # Engine indicator
    st.markdown(
        '<div style="text-align:center;padding:0 0 3rem 0;position:relative;z-index:10;animation:fadeInUp 0.9s cubic-bezier(0.16,1,0.3,1) 0.2s both;">'
        '<div style="display:inline-flex;align-items:center;gap:8px;color:#5f587a;font-size:0.75rem;">'
        '<div style="width:6px;height:6px;border-radius:50%;background:#34d399;box-shadow:0 0 6px rgba(52,211,153,0.4);"></div>'
        '<span>Upload an image to get started</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UPLOAD ZONE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_upload_zone():
    st.markdown(
        '<div style="text-align:center;padding:2.8rem 2rem;border:2px dashed rgba(139,92,246,0.22);border-radius:20px;background:linear-gradient(135deg,rgba(139,92,246,0.12),rgba(99,102,241,0.08));position:relative;z-index:10;transition:all 0.4s cubic-bezier(0.4,0,0.2,1);animation:fadeInUp 1s cubic-bezier(0.16,1,0.3,1) 0.15s both,uploadPulse 4s ease-in-out infinite;margin-bottom:0.6rem;">'
        '<div style="width:56px;height:56px;margin:0 auto 1rem auto;border-radius:16px;background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.15);display:flex;align-items:center;justify-content:center;font-size:1.4rem;">'
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
        '</div>'
        '<p style="font-size:1rem;font-weight:600;color:#f0eef6;margin:0 0 0.4rem 0;">Drop your image here</p>'
        '<p style="font-size:0.8rem;color:#5f587a;margin:0;">or click below to browse — JPG, JPEG, PNG, WEBP</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION LABEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def section_label(icon, text, delay="0s"):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.9rem;animation:fadeInUp 0.7s cubic-bezier(0.16,1,0.3,1) {delay} both;">'
        f'<span style="font-size:0.85rem;">{icon}</span>'
        f'<span style="font-family:Space Grotesk,sans-serif;font-size:0.8rem;font-weight:600;color:#9892b0;letter-spacing:0.5px;text-transform:uppercase;">{text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLASS CARD (open / close wrappers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def glass_card_open(delay="0s"):
    st.markdown(
        f'<div style="background:rgba(15,13,28,0.65);border:1px solid rgba(120,90,220,0.15);border-radius:20px;padding:1.6rem;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);box-shadow:0 0 30px rgba(139,92,246,0.08);transition:all 0.4s cubic-bezier(0.4,0,0.2,1);animation:fadeInUp 0.8s cubic-bezier(0.16,1,0.3,1) {delay} both;position:relative;z-index:10;">',
        unsafe_allow_html=True,
    )

def glass_card_close():
    st.markdown('</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMPTY STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_empty_state(panel="left"):
    if panel == "left":
        st.markdown(
            '<div style="background:rgba(15,13,28,0.65);border:1px solid rgba(120,90,220,0.15);border-radius:20px;padding:1.6rem;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);box-shadow:0 0 30px rgba(139,92,246,0.08);animation:fadeInUp 0.8s cubic-bezier(0.16,1,0.3,1) 0.1s both;position:relative;z-index:10;">'
            '<div style="text-align:center;padding:3rem 1.5rem;">'
            '<div style="width:64px;height:64px;margin:0 auto 1.2rem auto;border-radius:18px;background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.1);display:flex;align-items:center;justify-content:center;">'
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#5f587a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>'
            '</div>'
            '<p style="font-size:0.9rem;font-weight:500;color:#9892b0;margin:0 0 0.3rem 0;">No image uploaded</p>'
            '<p style="font-size:0.75rem;color:#5f587a;margin:0;">Upload an image above to preview it here</p>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:rgba(15,13,28,0.65);border:1px solid rgba(120,90,220,0.15);border-radius:20px;padding:1.6rem;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);box-shadow:0 0 30px rgba(139,92,246,0.08);animation:fadeInUp 0.8s cubic-bezier(0.16,1,0.3,1) 0.2s both;position:relative;z-index:10;">'
            '<div style="text-align:center;padding:3rem 1.5rem;">'
            '<div style="width:64px;height:64px;margin:0 auto 1.2rem auto;border-radius:18px;background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.1);display:flex;align-items:center;justify-content:center;">'
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#5f587a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
            '</div>'
            '<p style="font-size:0.9rem;font-weight:500;color:#9892b0;margin:0 0 0.3rem 0;">Waiting for input</p>'
            '<p style="font-size:0.75rem;color:#5f587a;margin:0;">AI-generated caption will appear here</p>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOADING ANIMATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_loading():
    st.markdown(
        '<div style="background:rgba(15,13,28,0.65);border:1px solid rgba(120,90,220,0.15);border-radius:20px;padding:1.6rem;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);box-shadow:0 0 30px rgba(139,92,246,0.08);animation:fadeInUp 0.8s cubic-bezier(0.16,1,0.3,1) both;position:relative;z-index:10;">'
        '<div style="text-align:center;padding:2.5rem 1.5rem;">'
        '<div style="display:flex;justify-content:center;gap:6px;margin-bottom:1.2rem;">'
        '<div style="width:8px;height:8px;border-radius:50%;background:#8b5cf6;animation:dotPulse 1.4s ease-in-out infinite;"></div>'
        '<div style="width:8px;height:8px;border-radius:50%;background:#6366f1;animation:dotPulse 1.4s ease-in-out 0.2s infinite;"></div>'
        '<div style="width:8px;height:8px;border-radius:50%;background:#a78bfa;animation:dotPulse 1.4s ease-in-out 0.4s infinite;"></div>'
        '</div>'
        '<p style="font-size:0.9rem;font-weight:500;color:#9892b0;margin:0 0 0.3rem 0;">Analyzing image...</p>'
        '<p style="font-size:0.72rem;color:#5f587a;margin:0;">BLIP model is processing your image</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CAPTION RESULT CARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_caption_card(caption):
    word_count = len(caption.split())
    cap_display = caption.capitalize()
    st.markdown(
        '<div style="background:rgba(15,13,28,0.65);border:1px solid rgba(120,90,220,0.15);border-radius:20px;padding:1.6rem;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);box-shadow:0 0 30px rgba(139,92,246,0.08);animation:slideInRight 0.7s cubic-bezier(0.16,1,0.3,1) both;position:relative;z-index:10;">'
        '<div style="display:inline-flex;align-items:center;gap:5px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.15);border-radius:6px;padding:4px 10px;margin-bottom:1.1rem;">'
        '<span style="font-size:0.65rem;">&#10022;</span>'
        '<span style="font-size:0.65rem;font-weight:600;color:#a78bfa;letter-spacing:0.6px;text-transform:uppercase;">AI Generated</span>'
        '</div>'
        f'<p style="font-family:Space Grotesk,sans-serif;font-size:1.2rem;font-weight:600;color:#f0eef6;line-height:1.5;margin:0 0 1.2rem 0;letter-spacing:-0.2px;">{cap_display}</p>'
        '<div style="height:1px;background:linear-gradient(90deg,rgba(139,92,246,0.15),transparent);margin-bottom:1rem;"></div>'
        f'<div style="display:flex;align-items:center;gap:8px;"><span style="font-size:0.68rem;color:#5f587a;">Salesforce BLIP Large &middot; {word_count} words</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMAGE METADATA CARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_image_meta(filename, dimensions):
    w, h = dimensions
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;background:rgba(20,17,40,0.45);border:1px solid rgba(120,90,220,0.15);border-radius:10px;padding:0.55rem 0.9rem;margin-top:0.7rem;backdrop-filter:blur(12px);animation:fadeIn 0.6s ease 0.3s both;">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#9892b0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>'
        f'<span style="font-size:0.72rem;color:#9892b0;font-weight:500;">{filename}</span>'
        '<span style="font-size:0.65rem;color:#5f587a;">&middot;</span>'
        f'<span style="font-size:0.68rem;color:#5f587a;">{w} &times; {h}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEPARATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_separator():
    st.markdown(
        '<div style="height:1px;max-width:120px;margin:1rem auto 2.2rem auto;background:linear-gradient(90deg,transparent,rgba(139,92,246,0.2),transparent);"></div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_footer():
    st.markdown(
        '<div style="text-align:center;padding:2.5rem 0 1.5rem 0;border-top:1px solid rgba(120,90,220,0.06);margin-top:3rem;animation:fadeIn 1s ease 0.5s both;position:relative;z-index:10;">'
        '<p style="font-size:0.7rem;color:#5f587a;margin:0 0 0.4rem 0;font-weight:500;">Built with PyTorch, Hugging Face Transformers, Salesforce BLIP &amp; Streamlit</p>'
        '<p style="font-size:0.62rem;color:rgba(95,88,122,0.6);margin:0;">VisionAI &mdash; AI-Powered Image Understanding</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APPLICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    inject_global_styles()
    render_background_orbs()
    render_header()
    render_hero()

    # ── Upload Section ──
    st.markdown(
        '<div style="max-width:640px;margin:0 auto 2rem auto;position:relative;z-index:10;">',
        unsafe_allow_html=True,
    )
    render_upload_zone()
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Process upload ──
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_rgb = image.convert("RGB")
        filename = uploaded_file.name
        dimensions = image_rgb.size
        current_data = uploaded_file.getvalue()

        # A new image starts a fresh caption session.
        if st.session_state.uploaded_image_data != current_data:
            st.session_state.uploaded_image_data = current_data
            st.session_state.caption = None

        st.session_state.image_filename = filename
        st.session_state.image_dimensions = dimensions

        render_separator()

        # ── Two-Panel Layout ──
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            section_label("◐", "Image Preview", "0.1s")
            glass_card_open("0.15s")
            st.image(image_rgb, use_container_width=True)
            glass_card_close()
            render_image_meta(filename, dimensions)

        with col_right:
            section_label("◑", "AI Caption", "0.2s")
            with st.form("caption_controls"):
                detail = st.radio(
                    "Caption detail",
                    options=["Quick", "Balanced", "Detailed"],
                    index=1,
                    horizontal=True,
                    help="Choose how much detail the model can include.",
                )
                token_limits = {"Quick": 24, "Balanced": 50, "Detailed": 80}
                generate_clicked = st.form_submit_button(
                    "Generate caption" if not st.session_state.caption else "Generate another",
                    type="primary",
                    use_container_width=True,
                )

            if generate_clicked:
                with st.spinner("Generating your caption…"):
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_path = tmp.name
                        image_rgb.save(tmp_path, format="JPEG")

                    try:
                        st.session_state.caption = generate_blip_caption(
                            tmp_path,
                            max_new_tokens=token_limits[detail],
                        )
                        st.session_state.caption_detail = detail
                    except Exception as error:
                        st.session_state.caption = None
                        st.error(f"Could not generate a caption: {error}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

            # Display caption result
            if st.session_state.caption:
                caption = st.session_state.caption
                render_caption_card(caption)
                st.caption(
                    f"Detail setting: {st.session_state.get('caption_detail', 'Balanced')}"
                )

                # st.code provides a working, browser-native copy button.
                st.code(caption, language=None)
                st.markdown('<div style="height:0.8rem;"></div>', unsafe_allow_html=True)
                st.download_button(
                    label="↓  Download caption",
                    data=caption,
                    file_name="caption.txt",
                    mime="text/plain",
                    key="download_caption",
                    use_container_width=True,
                )
            else:
                st.info("Choose a detail level, then generate a caption for this image.")

    else:
        # ── Empty state ──
        st.session_state.caption = None
        st.session_state.uploaded_image_data = None

        render_separator()

        col_left, col_right = st.columns([1, 1], gap="large")
        with col_left:
            section_label("◐", "Image Preview", "0.1s")
            render_empty_state("left")
        with col_right:
            section_label("◑", "AI Caption", "0.2s")
            render_empty_state("right")

    render_footer()


if __name__ == "__main__":
    main()
