"""
VisionAI — AI Image Caption Generator
Streamlit frontend wired to the existing BLIP inference backend.
"""

import os
import tempfile
import time

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from blip_inference import generate_blip_caption

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VisionAI · Give Your Images a Voice",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
_state_defaults = {
    "caption": None,
    "caption_detail": "Balanced",
    "uploaded_image_data": None,
    "image_filename": None,
    "image_dimensions": None,
    "inference_time_s": None,
    "generate_triggered": False,
}
for _k, _v in _state_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════
# RUNTIME CONSTANTS (no fake data)
# ═══════════════════════════════════════════════════════════════
MODEL_NAME = "Salesforce/blip-image-captioning-large"
FRAMEWORK = "PyTorch (CPU)"
TOKEN_LIMITS = {"Quick": 24, "Balanced": 50, "Detailed": 80}


# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
def _css() -> str:
    return """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Motion preference ── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
  }
}

/* ── Design tokens ── */
:root {
  --obsidian:   #080710;
  --surface-1:  #0f0e1a;
  --surface-2:  #13111f;
  --surface-3:  #1a1730;
  --border:     rgba(139,92,246,0.14);
  --border-hi:  rgba(139,92,246,0.32);
  --txt-hi:     #f0edf8;
  --txt-mid:    #9892b4;
  --txt-lo:     #5a5375;
  --violet:     #8b5cf6;
  --violet-hi:  #a78bfa;
  --indigo:     #6366f1;
  --green:      #34d399;
  --red:        #f87171;
  --r-lg:       16px;
  --r-md:       12px;
  --r-sm:       8px;
}

/* ── App shell + animated background ── */
body {
    background: #080710 !important;
}

.stApp {
    background: transparent !important;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"],
section.main {
    background: transparent !important;
    position: relative !important;
    z-index: 1 !important;
}

/* Keep Streamlit content above the Topolines canvas. */
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMainBlockContainer"],
.block-container {
    position: relative !important;
    z-index: 1 !important;
}

/* Topolines is mounted as a full-screen layer by components.html below. */
.stApp::before {
  content: none !important;
}


/* Keep Streamlit's layers transparent so the animated layer remains visible. */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.stApp > header {
  background: transparent !important;
}

[data-testid="stAppViewContainer"] {
  position: relative !important;
  z-index: 1 !important;
}
.block-container {
  padding: 0 1.5rem 3rem !important;
  max-width: 1280px !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"] {
  display: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--obsidian); }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.35); }

/* ── Typography resets ── */
h1,h2,h3,h4,h5,h6,p,span,label,li {
  font-family: 'Inter', sans-serif !important;
  color: var(--txt-hi) !important;
}

/* ── File uploader: hide native UI, expose only the button ── */
[data-testid="stFileUploader"] > label { display: none !important; }
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploader"] section {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
[data-testid="stFileUploader"] section > div { display: none !important; }
[data-testid="stFileUploader"] section > button {
  background: rgba(139,92,246,0.1) !important;
  color: var(--violet-hi) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: var(--r-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.8rem !important;
  padding: 0.45rem 1.1rem !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] section > button:hover {
  background: rgba(139,92,246,0.18) !important;
  border-color: var(--violet) !important;
}
[data-testid="stFileUploader"] small { color: var(--txt-lo) !important; font-size: 0.68rem !important; }
[data-testid="stFileUploaderFileName"] { color: var(--txt-mid) !important; font-size: 0.75rem !important; }

/* ── Generic stButton ── */
.stButton > button {
  background: var(--surface-3) !important;
  color: var(--txt-hi) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  padding: 0.45rem 1rem !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  border-color: var(--violet) !important;
  color: var(--violet-hi) !important;
  background: rgba(139,92,246,0.1) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ── Primary / CTA button override ── */
.vi-cta-wrap .stButton > button {
  background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  padding: 0.75rem 1.4rem !important;
  box-shadow: 0 4px 20px rgba(139,92,246,0.28) !important;
  width: 100% !important;
}
.vi-cta-wrap .stButton > button:hover {
  box-shadow: 0 6px 28px rgba(139,92,246,0.4) !important;
  transform: translateY(-1px) !important;
  background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
  background: var(--surface-3) !important;
  color: var(--txt-hi) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.78rem !important;
  transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
  border-color: var(--violet) !important;
  background: rgba(139,92,246,0.1) !important;
}

/* ── st.image ── */
[data-testid="stImage"] img {
  border-radius: var(--r-md) !important;
  border: 1px solid var(--border) !important;
  width: 100% !important;
  object-fit: cover !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--violet) !important; }

/* ── Alert ── */
[data-testid="stAlert"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  font-size: 0.82rem !important;
}

/* ── st.code (used as copy area) ── */
[data-testid="stCode"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
}
[data-testid="stCode"] pre { color: var(--txt-hi) !important; font-size: 0.85rem !important; }

/* ── Column gap ── */
[data-testid="stHorizontalBlock"] { gap: 1.25rem !important; align-items: stretch !important; }

/* ── Form submit button: make it look like a regular button ── */
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }

/* ── Radio ── */
[data-testid="stRadio"] label { color: var(--txt-mid) !important; font-size: 0.8rem !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: var(--txt-mid) !important; font-size: 0.8rem !important; }

/* ═══ VISIONAI COMPONENT STYLES ═══ */

/* Header nav bar */
.vi-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.9rem 0;
  border-bottom: 1px solid rgba(139,92,246,0.07);
  margin-bottom: 0;
}
.vi-nav-brand {
  display: flex; align-items: center; gap: 10px;
}
.vi-nav-logo {
  width: 32px; height: 32px; border-radius: 9px;
  background: linear-gradient(135deg,#7c3aed,#6366f1);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Space Grotesk',sans-serif;
  font-size: 0.95rem; font-weight: 700; color: #fff;
  box-shadow: 0 2px 10px rgba(139,92,246,0.3);
  flex-shrink: 0;
}
.vi-nav-name {
  font-family: 'Space Grotesk',sans-serif;
  font-size: 1.05rem; font-weight: 700; color: var(--txt-hi);
  letter-spacing: -0.3px;
}
.vi-badge {
  font-size: 0.55rem; font-weight: 700;
  background: rgba(139,92,246,0.14);
  color: var(--violet-hi);
  padding: 2px 7px; border-radius: 4px;
  letter-spacing: 0.6px; text-transform: uppercase;
}
.vi-nav-right { display: flex; align-items: center; gap: 10px; }
.vi-pill {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 10px;
  font-size: 0.7rem; color: var(--txt-mid);
}
.vi-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 6px rgba(52,211,153,0.5);
  animation: dot-pulse 3s ease-in-out infinite;
  flex-shrink: 0;
}
@keyframes dot-pulse {
  0%,100% { box-shadow: 0 0 4px rgba(52,211,153,0.3); }
  50%      { box-shadow: 0 0 10px rgba(52,211,153,0.6); }
}

/* Hero */
.vi-hero {
  text-align: center;
  padding: 3rem 1rem 2.5rem;
}
.vi-hero-eyebrow {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(139,92,246,0.07);
  border: 1px solid rgba(139,92,246,0.16);
  border-radius: 20px; padding: 5px 14px;
  margin-bottom: 1.2rem;
  font-size: 0.68rem; color: var(--violet-hi);
  font-weight: 600; letter-spacing: 0.7px; text-transform: uppercase;
}
.vi-hero-h1 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: clamp(2.1rem, 5.5vw, 3.8rem) !important;
  font-weight: 800 !important;
  line-height: 1.08 !important;
  letter-spacing: -1.5px !important;
  margin: 0 0 1rem 0 !important;
  color: var(--txt-hi) !important;
}
.vi-hero-accent {
  background: linear-gradient(95deg, #a78bfa, #818cf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.vi-hero-sub {
  font-size: clamp(0.9rem,1.8vw,1.05rem) !important;
  color: var(--txt-mid) !important;
  max-width: 560px; margin: 0 auto 1.4rem !important;
  line-height: 1.65 !important;
}
.vi-model-tag {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.7rem; color: var(--txt-lo);
}

/* Upload zone */
.vi-upload-zone {
  text-align: center;
  padding: 3rem 2rem;
  border: 1.5px dashed rgba(139,92,246,0.2);
  border-radius: var(--r-lg);
  background: var(--surface-1);
  margin-bottom: 0.5rem;
  transition: border-color 0.25s ease;
}
.vi-upload-zone:hover { border-color: rgba(139,92,246,0.35); }
.vi-upload-icon {
  width: 52px; height: 52px;
  margin: 0 auto 1rem;
  border-radius: 14px;
  background: rgba(139,92,246,0.09);
  border: 1px solid rgba(139,92,246,0.14);
  display: flex; align-items: center; justify-content: center;
}
.vi-upload-title {
  font-size: 0.95rem !important; font-weight: 600 !important;
  color: var(--txt-hi) !important; margin: 0 0 0.3rem 0 !important;
}
.vi-upload-sub {
  font-size: 0.75rem !important; color: var(--txt-lo) !important; margin: 0 !important;
}

/* Panel cards */
.vi-panel {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  overflow: hidden;
}
.vi-panel-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.7rem 1rem;
  border-bottom: 1px solid var(--border);
  background: var(--surface-2);
}
.vi-panel-title {
  font-size: 0.65rem !important; font-weight: 700 !important;
  color: var(--txt-lo) !important;
  letter-spacing: 0.8px; text-transform: uppercase;
}
.vi-panel-body { padding: 0.9rem; }
.vi-panel-footer {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.55rem 1rem;
  border-top: 1px solid var(--border);
  background: var(--surface-2);
  font-size: 0.68rem; color: var(--txt-lo);
  flex-wrap: wrap; gap: 0.5rem;
}

/* Primary prediction card */
.vi-primary-card {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 1.1rem;
  margin-bottom: 0.75rem;
}
.vi-primary-label {
  display: flex; align-items: center; gap: 5px;
  font-size: 0.62rem; font-weight: 700;
  color: var(--txt-lo); letter-spacing: 0.7px;
  text-transform: uppercase; margin-bottom: 0.75rem;
}
.vi-primary-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--violet-hi);
}
.vi-caption-text {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  color: var(--txt-hi) !important;
  line-height: 1.5 !important;
  margin: 0 !important;
  font-style: italic;
}
.vi-caption-meta {
  display: flex; align-items: center; gap: 10px;
  margin-top: 0.85rem;
  padding-top: 0.75rem;
  border-top: 1px solid rgba(139,92,246,0.08);
  font-size: 0.66rem; color: var(--txt-lo);
  flex-wrap: wrap; gap: 8px;
}

/* Meta chip */
.vi-chip {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(139,92,246,0.06);
  border: 1px solid rgba(139,92,246,0.1);
  border-radius: 4px; padding: 2px 7px;
  font-size: 0.63rem; color: var(--txt-lo);
  font-weight: 500;
}

/* Pipeline row */
.vi-pipeline {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1rem 1.25rem;
  display: flex; align-items: center;
  justify-content: space-between;
  flex-wrap: wrap; gap: 0.75rem;
  margin-top: 1.25rem;
}
.vi-pipeline-left { display: flex; flex-direction: column; gap: 3px; }
.vi-pipeline-title {
  font-size: 0.78rem !important; font-weight: 600 !important;
  color: var(--txt-hi) !important;
}
.vi-pipeline-sub {
  font-size: 0.67rem !important; color: var(--txt-lo) !important;
}
.vi-pipeline-steps {
  display: flex; align-items: center; gap: 6px;
  flex-wrap: wrap;
}
.vi-step {
  font-size: 0.65rem; color: var(--txt-mid);
  background: var(--surface-3);
  border: 1px solid var(--border);
  border-radius: 4px; padding: 3px 8px;
}
.vi-arrow { color: var(--txt-lo); font-size: 0.7rem; }

/* Footer */
.vi-footer {
  text-align: center;
  padding: 2rem 0 1rem;
  border-top: 1px solid rgba(139,92,246,0.05);
  margin-top: 2.5rem;
}
.vi-footer p {
  font-size: 0.66rem !important;
  color: var(--txt-lo) !important;
  margin: 0 !important;
  line-height: 1.7 !important;
}

/* Empty state */
.vi-empty {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 3rem 1rem; text-align: center; gap: 0.7rem;
}
.vi-empty-icon {
  width: 52px; height: 52px; border-radius: 14px;
  border: 1px solid rgba(139,92,246,0.1);
  background: rgba(139,92,246,0.05);
  display: flex; align-items: center; justify-content: center;
}
.vi-empty-title { font-size: 0.85rem !important; color: var(--txt-mid) !important; margin: 0 !important; font-weight: 500 !important; }
.vi-empty-sub   { font-size: 0.72rem !important; color: var(--txt-lo) !important; margin: 0 !important; }

/* ── Responsive ── */
@media (max-width: 768px) {
  .block-container { padding: 0 0.75rem 2rem !important; }
  .vi-hero { padding: 2rem 0.5rem 1.5rem; }
  .vi-hero-h1 { letter-spacing: -0.8px !important; }
  .vi-nav-right .vi-pill:not(:last-child) { display: none; }
  .vi-upload-zone { padding: 2rem 1rem; }
  .vi-pipeline { flex-direction: column; align-items: flex-start; }
  .vi-panel-footer { flex-direction: column; gap: 4px; }
}
@media (max-width: 480px) {
  .vi-nav-name { font-size: 0.95rem; }
  .vi-caption-text { font-size: 0.92rem !important; }
  .vi-pipeline-steps { gap: 4px; }
}
</style>"""


def _topolines_embed_html() -> str:
    """Mount Topolines into the parent Streamlit document as a fixed background."""
    return r"""<!doctype html>
<html>
<head><meta charset="utf-8"></head>
<body>
<script src="https://cdn.jsdelivr.net/npm/topolines"></script>
<script>
(() => {
  try {
    const doc = window.parent.document;
    const mount = () => {
      // Streamlit reruns recreate this small iframe; reuse the page-level layer.
      let host = doc.getElementById('visionai-topolines-bg');
      if (!host) {
        host = doc.createElement('div');
        host.id = 'visionai-topolines-bg';
        host.setAttribute('aria-hidden', 'true');
        doc.body.appendChild(host);
      }

      Object.assign(host.style, {
        position: 'fixed',
        inset: '0',
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        pointerEvents: 'none',
        zIndex: '0',
        opacity: '0.52',
        background: 'transparent'
      });

      // Avoid creating multiple animation instances after Streamlit reruns.
      if (!host.dataset.topolinesReady && window.topolines?.TopoField) {
        new window.topolines.TopoField(host, {
          seed: 'visionai-topolines',
          color: '#A78BFA'
        });
        host.dataset.topolinesReady = '1';
      }
    };

    if (doc.body) mount();
    else doc.addEventListener('DOMContentLoaded', mount, { once: true });
  } catch (error) {
    console.error('VisionAI Topolines background could not mount:', error);
  }
})();
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
# SVG ICONS (inline, no external dependency)
# ═══════════════════════════════════════════════════════════════
_ICON_UPLOAD = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'stroke="#8b5cf6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="17 8 12 3 7 8"/>'
    '<line x1="12" y1="3" x2="12" y2="15"/>'
    '</svg>'
)
_ICON_IMAGE = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'stroke="#5a5375" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>'
    '<circle cx="8.5" cy="8.5" r="1.5"/>'
    '<polyline points="21 15 16 10 5 21"/>'
    '</svg>'
)
_ICON_CAPTION = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'stroke="#5a5375" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
    '</svg>'
)
_ICON_SPARKLE = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" '
    'stroke="#a78bfa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2l2.4 7.2H22l-6.2 4.5 2.4 7.3L12 17l-6.2 4 2.4-7.3L2 9.2h7.6z"/>'
    '</svg>'
)
_ICON_CPU = (
    '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" '
    'stroke="#5a5375" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="4" y="4" width="16" height="16" rx="2"/>'
    '<rect x="9" y="9" width="6" height="6"/>'
    '<line x1="9" y1="2" x2="9" y2="4"/><line x1="15" y1="2" x2="15" y2="4"/>'
    '<line x1="9" y1="20" x2="9" y2="22"/><line x1="15" y1="20" x2="15" y2="22"/>'
    '<line x1="20" y1="9" x2="22" y2="9"/><line x1="20" y1="15" x2="22" y2="15"/>'
    '<line x1="2" y1="9" x2="4" y2="9"/><line x1="2" y1="15" x2="4" y2="15"/>'
    '</svg>'
)


# ═══════════════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════
def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HEADER / NAV
# ═══════════════════════════════════════════════════════════════
def render_nav() -> None:
    _html(
        '<div class="vi-nav">'
        '<div class="vi-nav-brand">'
        '<div class="vi-nav-logo">V</div>'
        '<span class="vi-nav-name">VisionAI</span>'
        '<span class="vi-badge">Beta</span>'
        '</div>'
        '<div class="vi-nav-right">'
        '<div class="vi-pill"><div class="vi-dot"></div>BLIP Captioning</div>'
        '<div class="vi-pill">PyTorch · CPU</div>'
        '</div>'
        '</div>'
    )


# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
def render_hero() -> None:
    _html(
        '<div class="vi-hero">'
        '<div class="vi-hero-eyebrow">'
        + _ICON_SPARKLE
        + '<span>BLIP-2 · Multimodal Inference</span>'
        '</div>'
        '<div class="vi-hero-h1">Give Your Images a'
        ' <span class="vi-hero-accent">Voice.</span></div>'
        '<p class="vi-hero-sub">'
        'Transform visual moments into intelligent descriptions with'
        ' state-of-the-art vision-language AI.'
        '</p>'
        '<div class="vi-model-tag">'
        '<div class="vi-dot"></div>'
        '<span>Online</span>'
        '<span style="color:var(--txt-lo);margin:0 4px;">·</span>'
        '<span>blip-image-captioning-large</span>'
        '<span style="color:var(--txt-lo);margin:0 4px;">·</span>'
        + _ICON_CPU
        + '<span>PyTorch CPU</span>'
        '</div>'
        '</div>'
    )


# ═══════════════════════════════════════════════════════════════
# UPLOAD ZONE
# ═══════════════════════════════════════════════════════════════
def render_upload_zone() -> None:
    _html(
        '<div class="vi-upload-zone">'
        '<div class="vi-upload-icon">' + _ICON_UPLOAD + '</div>'
        '<p class="vi-upload-title">Drop your image here or browse local files</p>'
        '<p class="vi-upload-sub">Supports JPG, PNG, WEBP &nbsp;·&nbsp; Max 20 MB</p>'
        '</div>'
    )


# ═══════════════════════════════════════════════════════════════
# STUDIO VIEWPORT (left panel)
# ═══════════════════════════════════════════════════════════════
def render_viewport_empty() -> None:
    _html(
        '<div class="vi-panel" style="height:100%;">'
        '<div class="vi-panel-header">'
        '<span class="vi-panel-title">&#9654; Studio Viewport</span>'
        '<span style="font-size:0.62rem;color:var(--txt-lo);">No image</span>'
        '</div>'
        '<div class="vi-empty">'
        + '<div class="vi-empty-icon">' + _ICON_IMAGE + '</div>'
        + '<p class="vi-empty-title">Awaiting image</p>'
        '<p class="vi-empty-sub">Upload an image above to begin</p>'
        '</div>'
        '</div>'
    )


def render_viewport_image(image_rgb: Image.Image, filename: str, dimensions: tuple) -> None:
    w, h = dimensions
    _html(
        '<div class="vi-panel">'
        '<div class="vi-panel-header">'
        '<span class="vi-panel-title">&#9654; Studio Viewport</span>'
        f'<span style="font-size:0.62rem;color:var(--txt-lo);">{w}&times;{h}</span>'
        '</div>'
        '<div class="vi-panel-body" style="padding:0.6rem;">'
    )
    st.image(image_rgb, use_container_width=True)
    _html('</div>')
    _html(
        '<div class="vi-panel-footer">'
        f'<span>{filename}</span>'
        f'<span>{w} &times; {h} px</span>'
        '</div>'
        '</div>'
    )


# ═══════════════════════════════════════════════════════════════
# AI CAPTION STUDIO (right panel)
# ═══════════════════════════════════════════════════════════════
def render_caption_studio_empty() -> None:
    _html(
        '<div class="vi-panel" style="height:100%;">'
        '<div class="vi-panel-header">'
        '<span class="vi-panel-title">&#10022; AI Caption Studio</span>'
        '</div>'
        '<div class="vi-empty">'
        + '<div class="vi-empty-icon">' + _ICON_CAPTION + '</div>'
        + '<p class="vi-empty-title">Ready to caption</p>'
        '<p class="vi-empty-sub">Upload an image to generate a description</p>'
        '</div>'
        '</div>'
    )


def render_caption_studio_active(image_rgb: Image.Image) -> None:
    """Renders the controls + result for an uploaded image."""

    # Detail selector + generate button
    _html(
        '<div class="vi-panel">'
        '<div class="vi-panel-header">'
        '<span class="vi-panel-title">&#10022; AI Caption Studio</span>'
        '<span style="font-size:0.62rem;color:var(--txt-lo);">Salesforce BLIP</span>'
        '</div>'
        '<div class="vi-panel-body">'
    )

    # Detail radio
    detail = st.radio(
        "Output length",
        options=list(TOKEN_LIMITS.keys()),
        index=1,
        horizontal=True,
        help="Controls the max token budget given to the model.",
        label_visibility="collapsed",
    )

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    # CTA button
    _html('<div class="vi-cta-wrap">')
    generate_clicked = st.button(
        "⚡  Generate Caption",
        key="btn_generate",
        use_container_width=True,
    )
    _html('</div>')

    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

    # ── Run inference when button clicked ──
    if generate_clicked:
        with st.spinner("Analyzing image with BLIP…"):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                image_rgb.save(tmp_path, format="JPEG")
            t0 = time.perf_counter()
            try:
                st.session_state.caption = generate_blip_caption(
                    tmp_path,
                    max_new_tokens=TOKEN_LIMITS[detail],
                )
                st.session_state.caption_detail = detail
                st.session_state.inference_time_s = round(time.perf_counter() - t0, 1)
            except Exception as exc:
                st.session_state.caption = None
                st.error(f"Caption generation failed: {exc}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # ── Display result ──
    if st.session_state.caption:
        caption = st.session_state.caption
        cap_display = caption.capitalize()
        word_count = len(caption.split())
        token_count = len(caption.split())          # approx token ≈ word for display
        inf_t = st.session_state.inference_time_s
        detail_used = st.session_state.get("caption_detail", "Balanced")

        # Primary prediction card
        _html(
            '<div class="vi-primary-card">'
            '<div class="vi-primary-label">'
            '<div class="vi-primary-dot"></div>'
            'Primary Prediction'
            '</div>'
            f'<p class="vi-caption-text">"{cap_display}"</p>'
            '<div class="vi-caption-meta">'
            f'<span class="vi-chip">Words: {word_count}</span>'
            f'<span class="vi-chip">Detail: {detail_used}</span>'
            + (f'<span class="vi-chip">Time: {inf_t}s</span>' if inf_t else '')
            + '</div>'
            '</div>'
        )

        # Copy area (st.code gives a native copy button for free)
        st.code(cap_display, language=None)

        # Action row
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            st.download_button(
                label="↓ Download .txt",
                data=caption,
                file_name="visionai_caption.txt",
                mime="text/plain",
                key="dl_caption",
                use_container_width=True,
            )
        with btn_c2:
            if st.button("↺ Regenerate", key="btn_regen", use_container_width=True):
                st.session_state.caption = None
                st.session_state.inference_time_s = None
                st.rerun()

    else:
        _html(
            '<div style="text-align:center;padding:1.5rem 0.5rem;">'
            '<p style="font-size:0.78rem;color:var(--txt-lo);">Choose a detail level and press <strong style=\'color:var(--txt-mid)\'>Generate Caption</strong>.</p>'
            '</div>'
        )

    _html('</div></div>')   # close vi-panel-body + vi-panel


# ═══════════════════════════════════════════════════════════════
# PIPELINE SECTION (only factual info)
# ═══════════════════════════════════════════════════════════════
def render_pipeline() -> None:
    _html(
        '<div class="vi-pipeline">'
        '<div class="vi-pipeline-left">'
        '<span class="vi-pipeline-title">End-to-End Multimodal Pipeline</span>'
        '<span class="vi-pipeline-sub">Image → BLIP Encoder → Language Model → Caption text</span>'
        '</div>'
        '<div class="vi-pipeline-steps">'
        '<span class="vi-step">PIL Image</span>'
        '<span class="vi-arrow">→</span>'
        '<span class="vi-step">BlipProcessor</span>'
        '<span class="vi-arrow">→</span>'
        '<span class="vi-step">BlipForConditionalGeneration</span>'
        '<span class="vi-arrow">→</span>'
        '<span class="vi-step">Decoded Text</span>'
        '</div>'
        '</div>'
    )


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
def render_footer() -> None:
    _html(
        '<div class="vi-footer">'
        '<p>Built with PyTorch · Hugging Face Transformers · Salesforce BLIP · Streamlit</p>'
        '<p style="margin-top:3px;">VisionAI &mdash; AI-Powered Image Understanding</p>'
        '</div>'
    )


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    # 1. Inject styles
    _html(_css())

    # 2. Mount full-screen Topolines background
    components.html(_topolines_embed_html(), height=0, width=0)

    # 3. Header
    render_nav()

    # 3. Hero
    render_hero()

    # 4. Upload row (centred, max-width capped)
    _html('<div style="max-width:680px;margin:0 auto 1.5rem;">')
    render_upload_zone()
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader",
        label_visibility="collapsed",
    )
    _html('</div>')

    # 5. Main workspace
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_rgb = image.convert("RGB")
        filename = uploaded_file.name
        dimensions = image_rgb.size
        current_bytes = uploaded_file.getvalue()

        # New image → clear stale caption
        if st.session_state.uploaded_image_data != current_bytes:
            st.session_state.uploaded_image_data = current_bytes
            st.session_state.caption = None
            st.session_state.inference_time_s = None

        st.session_state.image_filename = filename
        st.session_state.image_dimensions = dimensions

        # Thin separator
        _html(
            '<div style="height:1px;max-width:80px;margin:0.5rem auto 1.5rem;'
            'background:linear-gradient(90deg,transparent,rgba(139,92,246,0.18),transparent);"></div>'
        )

        # Two-column desktop layout
        col_left, col_right = st.columns([10, 9], gap="medium")

        with col_left:
            render_viewport_image(image_rgb, filename, dimensions)

        with col_right:
            render_caption_studio_active(image_rgb)

        # Pipeline section (below both panels)
        render_pipeline()

    else:
        # Clear stale state when no image
        if st.session_state.uploaded_image_data is not None:
            st.session_state.caption = None
            st.session_state.uploaded_image_data = None
            st.session_state.inference_time_s = None

        _html(
            '<div style="height:1px;max-width:80px;margin:0.5rem auto 1.5rem;'
            'background:linear-gradient(90deg,transparent,rgba(139,92,246,0.1),transparent);"></div>'
        )
        col_left, col_right = st.columns([10, 9], gap="medium")
        with col_left:
            render_viewport_empty()
        with col_right:
            render_caption_studio_empty()

        render_pipeline()

    # 6. Footer
    render_footer()


if __name__ == "__main__":
    main()
