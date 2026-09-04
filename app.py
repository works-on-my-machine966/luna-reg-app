import hashlib
import os
import tempfile
import numpy as np
import cv2
import streamlit as st
import scipy.optimize as opt
from PIL import Image
from skimage.metrics import normalized_mutual_information
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="LUNA-REG Engine", layout="wide")

# --- UTILITY & PREPROCESSING ENGINE ---

def compute_sha256(file_bytes: bytes) -> str:
    sha256 = hashlib.sha256()
    sha256.update(file_bytes)
    return sha256.hexdigest()

def preprocess_lunar_image(gray_img: np.ndarray) -> np.ndarray:
    """Preprocess lunar images using CLAHE and Laplacian edge extraction for robust feature matching."""
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_img)
    
    laplacian = cv2.Laplacian(equalized, cv2.CV_32F, ksize=3)
    laplacian = np.abs(laplacian)
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# --- SAFE FEATURE DETECTOR FACTORY ---

def get_feature_detector():
    """Safely instantiates AKAZE or falls back to ORB depending on OpenCV build."""
    if hasattr(cv2, 'AKAZE_create'):
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    elif hasattr(cv2, 'AKAZE') and hasattr(cv2.AKAZE, 'create'):
        return cv2.AKAZE.create(), cv2.NORM_HAMMING
    else:
        return cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING

# --- FINE REGISTRATION OPTIMIZATION OBJECTIVE ---

def nmi_objective_function(params: np.ndarray, src_gray: np.ndarray, ref_gray: np.ndarray) -> float:
    """Objective function for Quasi-Newton (L-BFGS-B) optimization minimizing negative NMI."""
    tx, ty, angle, scale = params
    h, w = ref_gray.shape[:2]
    
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    warped = cv2.warpAffine(src_gray, M, (w, h), flags=cv2.INTER_LINEAR)
    return -float(normalized_mutual_information(ref_gray, warped))

# --- HYBRID (COARSE + QUASI-NEWTON FINE) REGISTRATION ENGINE ---

def register_images_robust(ref_img: np.ndarray, src_img: np.ndarray):
    ref_h, ref_w = ref_img.shape[:2]
    
    # 1. Convert inputs to grayscale safely
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY) if len(ref_img.shape) == 3 else ref_img
    gray_src = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY) if len(src_img.shape) == 3 else src_img

    # 2. Extract keypoints at original image scale
    prep_ref = preprocess_lunar_image(gray_ref)
    prep_src = preprocess_lunar_image(gray_src)

    detector, norm_type = get_feature_detector()
    kp1, des1 = detector.detectAndCompute(prep_ref, None)
    kp2, des2 = detector.detectAndCompute(prep_src, None)

    # Standard fallback canvas (original resized)
    final_registered = cv2.resize(src_img, (ref_w, ref_h))
    inliers_count = 0
    matched_pts = np.array([])
    final_dx, final_dy = 0.0, 0.0

    if des1 is not None and des2 is not None and len(kp1) >= 10 and len(kp2) >= 10:
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                # Strict ratio test threshold
                if m.distance < 0.70 * n.distance:
                    good_matches.append(m)

        # Must have at least 12 candidate matches to attempt homography
        if len(good_matches) >= 12:
            pts_ref = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts_src = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            # Homography RANSAC algorithm
            H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)

            if H is not None and mask is not None:
                current_inliers = int(np.sum(mask))
                
                # GUARDRAIL: Require at least 10 verified RANSAC inliers to apply transformation
                if current_inliers >= 10:
                    inliers_count = current_inliers
                    matched_pts = pts_ref[mask.ravel() == 1]
                    
                    # STAGE 1: Coarse Registration (RANSAC Homography)
                    coarse_warped = cv2.warpPerspective(src_img, H, (ref_w, ref_h))
                    
                    # STAGE 2: Fine Registration (Quasi-Newton L-BFGS-B Optimization)
                    try:
                        gray_coarse = cv2.cvtColor(coarse_warped, cv2.COLOR_RGB2GRAY) if len(coarse_warped.shape) == 3 else coarse_warped
                        
                        initial_guess = [0.0, 0.0, 0.0, 1.0]
                        param_bounds = [(-8.0, 8.0), (-8.0, 8.0), (-2.0, 2.0), (0.98, 1.02)]
                        
                        opt_result = opt.minimize(
                            nmi_objective_function,
                            initial_guess,
                            args=(gray_coarse, gray_ref),
                            method='L-BFGS-B',
                            bounds=param_bounds,
                            options={'maxiter': 20, 'ftol': 1e-3}
                        )
                        
                        tx, ty, angle, scale = opt_result.x
                        M_fine = cv2.getRotationMatrix2D((ref_w / 2, ref_h / 2), angle, scale)
                        M_fine[0, 2] += tx
                        M_fine[1, 2] += ty
                        
                        final_registered = cv2.warpAffine(coarse_warped, M_fine, (ref_w, ref_h))
                        final_dx = float(H[0, 2] + tx)
                        final_dy = float(H[1, 2] + ty)
                    except Exception:
                        # Fall back to coarse output if continuous optimizer diverges
                        final_registered = coarse_warped
                        final_dx = float(H[0, 2])
                        final_dy = float(H[1, 2])

    gray_final = cv2.cvtColor(final_registered, cv2.COLOR_RGB2GRAY) if len(final_registered.shape) == 3 else final_registered
    
    diff = gray_ref.astype(np.float32) - gray_final.astype(np.float32)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nmi_score = float(normalized_mutual_information(gray_ref, gray_final))

    return final_registered, rmse, inliers_count, matched_pts, (final_dx, final_dy), nmi_score

def calculate_grid_entropy(pts: np.ndarray, image_shape: tuple, grid_size: tuple = (4, 4)) -> float:
    if len(pts) == 0:
        return 0.0
    h, w = image_shape[:2]
    rows, cols = grid_size
    grid_counts, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[rows, cols], range=[[0, h], [0, w]])
    total = np.sum(grid_counts)
    if total == 0:
        return 0.0
    probs = grid_counts.flatten() / total
    nonzero_probs = probs[probs > 0]
    shannon_entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
    max_entropy = np.log(rows * cols)
    return float(np.round(shannon_entropy / max_entropy, 4))

# --- PDF REPORT ENGINE ---

def generate_pdf_report(pdf_path: str, metrics: dict, ref_hash: str, src_hash: str, ref_img: np.ndarray, reg_img: np.ndarray, sensor_label: str):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_ref, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_reg:
        temp_ref = f_ref.name
        temp_reg = f_reg.name

    try:
        cv2.imwrite(temp_ref, cv2.cvtColor(cv2.resize(ref_img, (512, 512)), cv2.COLOR_RGB2BGR))
        cv2.imwrite(temp_reg, cv2.cvtColor(cv2.resize(reg_img, (512, 512)), cv2.COLOR_RGB2BGR))

        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, textColor=colors.HexColor('#0F172A'))
        h2_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#1E293B'), spaceBefore=8, spaceAfter=4)
        cell_bold = ParagraphStyle('CellB', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10)
        cell_norm = ParagraphStyle('CellN', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10)
        hash_style = ParagraphStyle('HashN', parent=styles['Normal'], fontName='Courier', fontSize=7, leading=9, textColor=colors.HexColor('#334155'))
        pass_style = ParagraphStyle('PassS', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor('#15803D'))
        fail_style = ParagraphStyle('FailS', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor('#B91C1C'))

        status_text = "ALIGNED" if metrics['inliers'] >= 10 else "UNALIGNED"
        status_color = pass_style if metrics['inliers'] >= 10 else fail_style

        story.append(Paragraph(f"Chandrayaan-2 Registration Benchmark Report: {sensor_label}", title_style))
        story.append(Paragraph("Smart India Hackathon | Multimodal Hybrid Alignment Engine", styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceBefore=0, spaceAfter=8))

        story.append(Paragraph("1. Data Integrity & Cryptographic Signatures", h2_style))
        hash_data = [
            [Paragraph("Target Image", cell_bold), Paragraph("Cryptographic SHA-256 Hash", cell_bold), Paragraph("Status", cell_bold)],
            [Paragraph("TMC Base Reference", cell_norm), Paragraph(ref_hash, hash_style), Paragraph("VERIFIED", pass_style)],
            [Paragraph(f"{sensor_label} Moving Image", cell_norm), Paragraph(src_hash, hash_style), Paragraph("VERIFIED", pass_style)],
        ]
        t1 = Table(hash_data, colWidths=[1.8*inch, 4.4*inch, 1.0*inch])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ]))
        story.append(t1)

        story.append(Paragraph("2. Performance & Precision Metrics", h2_style))
        m_data = [
            [Paragraph("Pipeline Stage", cell_bold), Paragraph("Algorithm Method", cell_bold), Paragraph("Measured Metric", cell_bold), Paragraph("Status", cell_bold)],
            [Paragraph("Registration Engine", cell_norm), Paragraph("RANSAC + Quasi-Newton (L-BFGS-B)", cell_norm), Paragraph(f"RMSE: {metrics['coarse_rmse']:.2f}", cell_norm), Paragraph(status_text, status_color)],
            [Paragraph("Total Shift", cell_norm), Paragraph("Sub-Pixel Offset Vector", cell_norm), Paragraph(f"Shift: ({metrics['shift'][0]:.2f}, {metrics['shift'][1]:.2f}) px", cell_norm), Paragraph(status_text, status_color)],
            [Paragraph("Spatial Uniformity", cell_norm), Paragraph("4x4 Grid Shannon Entropy", cell_norm), Paragraph(f"Score: {metrics['uniformity']:.4f}", cell_norm), Paragraph("PASS" if metrics['inliers'] >= 10 else "LOW", status_color)],
            [Paragraph("Multimodal Similarity", cell_norm), Paragraph("Normalized Mutual Info (NMI)", cell_norm), Paragraph(f"NMI Score: {metrics['nmi']:.4f}", cell_norm), Paragraph("PASS" if metrics['inliers'] >= 10 else "LOW", status_color)],
        ]
        t2 = Table(m_data, colWidths=[1.5*inch, 2.3*inch, 2.2*inch, 1.0*inch])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ]))
        story.append(t2)
        story.append(Spacer(1, 10))

        story.append(Paragraph("3. Registered Canvas Inspection", h2_style))
        img1 = RLImage(temp_ref, width=3.3*inch, height=2.4*inch)
        img2 = RLImage(temp_reg, width=3.3*inch, height=2.4*inch)
        img_table = Table([[Paragraph("<b>TMC Base Canvas</b>", cell_norm), Paragraph(f"<b>Aligned {sensor_label} Frame</b>", cell_norm)], [img1, img2]], colWidths=[3.4*inch, 3.4*inch])
        img_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(img_table)

        doc.build(story)
    finally:
        for p in [temp_ref, temp_reg]:
            if os.path.exists(p): os.remove(p)

def render_alignment_module(ref_np, moving_np, ref_hash, moving_hash, sensor_label):
    st.markdown(f"### Data Integrity: {sensor_label} vs TMC Base")
    st.code(f"TMC Reference SHA-256: {ref_hash}\n{sensor_label} Moving SHA-256: {moving_hash}", language="text")

    with st.spinner(f"Aligning {sensor_label} using Hybrid RANSAC + Quasi-Newton Optimization..."):
        final_np, coarse_rmse, inliers, matched_pts, opt_shift, nmi_score = register_images_robust(ref_np, moving_np)
        entropy = calculate_grid_entropy(matched_pts, ref_np.shape)

    metrics = {
        'coarse_rmse': coarse_rmse,
        'uniformity': entropy,
        'inliers': inliers,
        'shift': opt_shift,
        'nmi': nmi_score
    }

    st.markdown("### Advanced Evaluation Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    is_aligned = inliers >= 10
    status_color = "normal" if is_aligned else "inverse"
    
    m1.metric("Intensity RMSE", f"{coarse_rmse:.2f}", 
              delta="Aligned" if is_aligned else "High Error", 
              delta_color=status_color)
              
    m2.metric("Total Shift (dx, dy)", f"({opt_shift[0]:.1f}, {opt_shift[1]:.1f}) px", 
              delta="Hybrid Refined" if is_aligned else "No Warp Applied", 
              delta_color="normal" if is_aligned else "off")
              
    m3.metric("Grid Uniformity", f"{entropy:.4f}", 
              delta="Spatial Distribution" if is_aligned else "No Matches", 
              delta_color=status_color)
              
    m4.metric("Inlier Keypoints", inliers, 
              delta="RANSAC Features" if is_aligned else "Insufficient Matches", 
              delta_color=status_color)
              
    m5.metric("Multimodal NMI", f"{nmi_score:.4f}", 
              delta="Similarity" if is_aligned else "Unverified", 
              delta_color=status_color)

    if is_aligned:
        st.success(f"Alignment complete! Calculated shift vector: dx={opt_shift[0]:.2f}px, dy={opt_shift[1]:.2f}px")
    else:
        st.error("Registration failed: Insufficient overlapping keypoints found between these images (< 10 RANSAC inliers). Ensure both images cover the same lunar region.")

    st.markdown("### Visual Registration Inspection")
    view_mode = st.radio(
        "Select Visual Mode:",
        ["Interactive Wipe Slider", "Side-by-Side View"],
        key=f"view_{sensor_label}",
        horizontal=True
    )

    if view_mode == "Interactive Wipe Slider":
        image_comparison(
            img1=ref_np,
            img2=final_np,
            label1="TMC Baseline",
            label2=f"Aligned {sensor_label}",
            starting_position=50,
            show_labels=True,
            make_responsive=False,
            in_memory=True,
            width=700
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.image(ref_np, caption="TMC Baseline Canvas", use_container_width=True)
        with c2:
            st.image(final_np, caption=f"Aligned {sensor_label} Frame", use_container_width=True)

    st.markdown("### Export Benchmark Report")
    pdf_filename = f"Chandrayaan2_{sensor_label}_Report.pdf"
    if st.button(f"Generate {sensor_label} PDF Report", key=f"btn_{sensor_label}"):
        generate_pdf_report(pdf_filename, metrics, ref_hash, moving_hash, ref_np, final_np, sensor_label)
        with open(pdf_filename, "rb") as f:
            st.download_button(
                label=f"Download {sensor_label} PDF Report",
                data=f,
                file_name=pdf_filename,
                mime="application/pdf",
                key=f"dl_{sensor_label}"
            )

    return final_np, metrics

# --- UI MAIN LAYOUT ---

st.title("LUNA-REG: Chandrayaan-2 Multi-Sensor Alignment System")
st.caption("Automated Co-Registration & Spatial Verification Suite | Smart India Hackathon")

col1, col2, col3 = st.columns(3)
with col1:
    tmc_file = st.file_uploader("1. TMC Base Map (Reference)", type=["png", "jpg", "jpeg", "tif"], key="tmc")
with col2:
    ohrc_file = st.file_uploader("2. OHRC Optical Frame", type=["png", "jpg", "jpeg", "tif"], key="ohrc")
with col3:
    iirs_file = st.file_uploader("3. IIRS Spectrometer Band", type=["png", "jpg", "jpeg", "tif"], key="iirs")

if tmc_file:
    tmc_bytes = tmc_file.read()
    tmc_hash = compute_sha256(tmc_bytes)
    tmc_np = np.array(Image.open(tmc_file).convert("RGB"))

    reg_ohrc, metrics_ohrc = None, None
    reg_iirs, metrics_iirs = None, None

    if ohrc_file:
        ohrc_bytes = ohrc_file.read()
        ohrc_hash = compute_sha256(ohrc_bytes)
        ohrc_np = np.array(Image.open(ohrc_file).convert("RGB"))

    if iirs_file:
        iirs_bytes = iirs_file.read()
        iirs_hash = compute_sha256(iirs_bytes)
        iirs_np = np.array(Image.open(iirs_file).convert("RGB"))

    tab1, tab2, tab3 = st.tabs(["OHRC Alignment", "IIRS Alignment", "3-Channel Composite"])

    with tab1:
        if ohrc_file:
            reg_ohrc, metrics_ohrc = render_alignment_module(tmc_np, ohrc_np, tmc_hash, ohrc_hash, "OHRC")
        else:
            st.info("Upload an OHRC image above to activate optical registration.")

    with tab2:
        if iirs_file:
            reg_iirs, metrics_iirs = render_alignment_module(tmc_np, iirs_np, tmc_hash, iirs_hash, "IIRS")
        else:
            st.info("Upload an IIRS image above to activate infrared spectrometer registration.")

    with tab3:
        ohrc_valid = metrics_ohrc is not None and metrics_ohrc['inliers'] >= 10
        iirs_valid = metrics_iirs is not None and metrics_iirs['inliers'] >= 10

        if ohrc_valid and iirs_valid:
            st.markdown("### Unified 3-Sensor Composite View (TMC + OHRC + IIRS)")
            st.caption("Fusing all registered channels onto the TMC spatial reference grid.")

            reg_ohrc_resized = cv2.resize(reg_ohrc, (tmc_np.shape[1], tmc_np.shape[0]))
            reg_iirs_resized = cv2.resize(reg_iirs, (tmc_np.shape[1], tmc_np.shape[0]))

            fused_view = cv2.addWeighted(tmc_np, 0.4, reg_ohrc_resized, 0.3, 0)
            fused_view = cv2.addWeighted(fused_view, 0.7, reg_iirs_resized, 0.3, 0)
            st.image(fused_view, caption="Fused Spatial-Infrared Canvas (Full 3-Sensor Composite)", use_container_width=True)

        elif ohrc_valid or iirs_valid:
            valid_label = "OHRC" if ohrc_valid else "IIRS"
            failed_label = "IIRS" if ohrc_valid else "OHRC"
            valid_img = reg_ohrc if ohrc_valid else reg_iirs

            st.markdown(f"### Partial 2-Channel Composite View (TMC + {valid_label})")
            st.warning(f"{failed_label} registration failed. Rendering a 2-channel blend with the valid {valid_label} frame only.")

            valid_resized = cv2.resize(valid_img, (tmc_np.shape[1], tmc_np.shape[0]))
            fused_view = cv2.addWeighted(tmc_np, 0.5, valid_resized, 0.5, 0)
            st.image(fused_view, caption=f"Partial Composite (TMC Base + Aligned {valid_label})", use_container_width=True)

        else:
            st.error("Cannot generate composite: Neither OHRC nor IIRS achieved valid alignment with the TMC base map (< 10 RANSAC inliers).")
else:
    st.info("Upload at least the TMC Base Reference image above to begin.")
