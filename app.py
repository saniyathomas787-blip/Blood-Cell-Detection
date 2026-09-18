import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Blood Cell Detection",
    page_icon="🩸",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f6f8fb;
    }

    .main-title {
        text-align: center;
        color: #b42318;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #667085;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .info-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e4e7ec;
        margin-bottom: 25px;
    }

    .result-title {
        font-size: 25px;
        font-weight: 650;
        color: #344054;
        margin-top: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e4e7ec;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 17px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 13px;
        margin-top: 45px;
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


st.markdown(
    '<div class="main-title">🩸 Blood Cell Detection & Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Based RBC, WBC and Platelet Detection System</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
<b>About the System</b><br><br>
Upload a microscopic blood smear image. The trained deep learning model
detects and classifies Red Blood Cells (RBC), White Blood Cells (WBC),
and Platelets and provides the detected cell count.
</div>
""", unsafe_allow_html=True)


st.subheader("Upload Blood Smear Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        '<div class="result-title">Uploaded Image</div>',
        unsafe_allow_html=True
    )

    st.image(image, width=650)

    if st.button("🔬 Detect Blood Cells", type="primary"):

        with st.spinner("Analyzing blood smear image..."):

            results = model.predict(
                source=np.array(image),
                conf=0.25
            )

            result = results[0]

            counts = {
                "RBC": 0,
                "WBC": 0,
                "Platelets": 0
            }

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name in counts:
                    counts[class_name] += 1

            annotated_image = result.plot()
            annotated_image = annotated_image[:, :, ::-1]

        st.markdown(
            '<div class="result-title">Detection Result</div>',
            unsafe_allow_html=True
        )

        st.image(annotated_image, width=850)

        st.markdown(
            '<div class="result-title">Detected Blood Cell Count</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="🔴 Red Blood Cells (RBC)",
                value=counts["RBC"]
            )

        with col2:
            st.metric(
                label="⚪ White Blood Cells (WBC)",
                value=counts["WBC"]
            )

        with col3:
            st.metric(
                label="🟣 Platelets",
                value=counts["Platelets"]
            )

        total_cells = (
            counts["RBC"]
            + counts["WBC"]
            + counts["Platelets"]
        )

        st.info(f"Total detected blood cells: {total_cells}")

        st.success("Blood cell detection completed successfully.")


st.markdown("""
<div class="footer">
Blood Cell Detection & Classification using Deep Learning<br>
Educational and Research Purpose Only
</div>
""", unsafe_allow_html=True)