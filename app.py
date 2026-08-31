import io
import re
import cv2
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import pytesseract
import streamlit as st
import zxingcpp

# Page Setup
st.set_page_config(
    page_title="PDF QR & App Number Scanner",
    page_icon="📑",
    layout="centered"
)

# Custom Styling & Footer CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1rem;
        text-align: center;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F3F4F6;
        color: #1F2937;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        font-weight: 600;
        border-top: 1px solid #E5E7EB;
        z-index: 100;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 Document Data & QR Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">High-Accuracy Document OCR & Multi-Angle QR Extraction</div>', unsafe_allow_html=True)

# ----------------- Core Extraction Engine -----------------

def try_zxing(img):
    """Zebra Crossing scan across all rotations (0, 90, 180, 270) with auto-downscale & inversion."""
    try:
        results = zxingcpp.read_barcodes(
            img,
            formats=zxingcpp.BarcodeFormat.QRCode,
            try_rotate=True,
            try_downscale=True,
            try_invert=True
        )
        for r in results:
            if r.text and r.text.strip():
                return r.text.strip()
    except Exception:
        pass
    return None

def scan_qr_code_extreme(cv_img):
    """Ultra-resilient multi-pass QR decoder."""
    # 1. Direct Full Page Scan
    res = try_zxing(cv_img)
    if res:
        return res

    # 2. Focus on Top-Left Area (Expanded to 55% H x 60% W)
    h, w = cv_img.shape[:2]
    top_left = cv_img[0:int(h * 0.55), 0:int(w * 0.60)]
    res = try_zxing(top_left)
    if res:
        return res

    # 3. Grayscale + CLAHE (Local Contrast Enhancement for faded/shadowed scans)
    gray = cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    res = try_zxing(enhanced)
    if res:
        return res

    # 4. Otsu Adaptive Binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = try_zxing(thresh)
    if res:
        return res

    # 5. Adaptive Thresholding
    adapt_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    res = try_zxing(adapt_thresh)
    if res:
        return res

    # 6. Morphological Closing (Connects broken QR module dots)
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    res = try_zxing(closed)
    if res:
        return res

    return "Missing / No QR Code"

def extract_app_number(cv_img):
    """Target top-right 40% area to read App. No."""
    h, w = cv_img.shape[:2]
    top_right = cv_img[0:int(h * 0.35), int(w * 0.45):w]
    
    gray = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    
    # Priority Match: Typical Kingdom Valley formats (e.g. KP1026144, LM0052577, ZA8028190)
    match = re.search(r'\b([A-Z]{2,3}\d{6,8})\b', text)
    if match:
        return match.group(1)
        
    fallback_match = re.search(r'[A-Z0-9]{7,10}', text)
    if fallback_match:
        return fallback_match.group(0)
        
    return "Not Detected"

# ----------------- UI Workflow -----------------

uploaded_file = st.file_uploader("Upload Multi-Page PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("🚀 Analyze All Pages"):
        with st.spinner("Processing pages with multi-pass recognition..."):
            pdf_bytes = uploaded_file.read()
            pdf = pdfium.PdfDocument(pdf_bytes)
            total_pages = len(pdf)

            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for idx in range(total_pages):
                status_text.text(f"Analyzing Page {idx + 1} of {total_pages}...")
                page = pdf[idx]
                
                # Render at high resolution (scale 4.0 = ~300 DPI for precision)
                bitmap = page.render(scale=4.0)
                pil_img = bitmap.to_pil()
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                app_no = extract_app_number(cv_img)
                qr_data = scan_qr_code_extreme(cv_img)

                results.append({
                    "Page No": idx + 1,
                    "Application Number": app_no,
                    "QR Code Data": qr_data
                })

                progress_bar.progress((idx + 1) / total_pages)

            status_text.empty()
            df = pd.DataFrame(results)

            st.success("Analysis Complete!")
            st.dataframe(df, use_container_width=True)

            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)

            st.download_button(
                label="📥 Download Excel File (.xlsx)",
                data=excel_buffer,
                file_name="Extracted_Property_Records.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Developer Footer
st.markdown("""
    <div class="footer">
        🛠️ Developed by <b>Tayyab Khan</b> | 📞 Phone: <b>03088622779</b>
    </div>
""", unsafe_allow_html=True)
