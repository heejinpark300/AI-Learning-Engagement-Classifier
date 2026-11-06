
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path

# 프로젝트 경로
BASE = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE / "models" / "engagement_mobilenetv2.keras"
)

CLASS_NAMES = ["Disengagement", "Engagement"]

IMG_SIZE = (224, 224)

# 웹앱 설정
st.set_page_config(
    page_title="AI Learning Engagement Classifier",
    page_icon="📚",
    layout="centered"
)

st.title("AI Learning Engagement Classifier")

st.caption(
    "MobileNetV2-Based Learning-Scene Image Classification"
)

st.info(
    "This application predicts Engagement and Disengagement "
    "labels assigned to learning-scene images. "
    "It does not provide a validated assessment of "
    "students' actual learning engagement."
)

uploaded_file = st.file_uploader(
    "Upload a learning-scene image.",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="업로드한 이미지",
        use_container_width=True
    )

    # 이미지 크기 조정
    resized = image.resize(IMG_SIZE)

    # 모델 입력 형태: (1, 224, 224, 3)
    image_array = np.array(
        resized,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # 딥러닝 예측
    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    # Engagement가 클래스 1
    predicted_class = (
        1 if probability >= 0.5 else 0
    )

    predicted_label = CLASS_NAMES[predicted_class]

    confidence = (
        probability
        if predicted_class == 1
        else 1 - probability
    )

    st.divider()

    st.subheader("AI 예측 결과")

    st.metric(
        "예측된 이미지 클래스",
        predicted_label
    )

    st.metric(
        "모델 예측 확률",
        f"{confidence * 100:.2f}%"
    )

    st.write("클래스별 예측 확률")

    st.progress(
        probability,
        text=f"Engagement: {probability * 100:.1f}%"
    )

    st.progress(
        1 - probability,
        text=f"Disengagement: {(1 - probability) * 100:.1f}%"
    )

    st.caption(
        "표시된 확률은 모델의 예측 점수이며, "
        "실제 학습 몰입도나 신뢰도가 검증된 확률은 아닙니다."
    )
