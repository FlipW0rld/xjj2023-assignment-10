# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
from utils import load_model, load_data, normalize_features, get_pca_transform, encode_text, encode_image, compute_similarity, get_topk_images

# 配置路径（请根据实际情况修改）
EMBEDDING_PATH = "image_embeddings.pickle"  # 已有的 embeddings 文件
IMAGE_FOLDER = "coco_images_resized"        # 解压后的图片文件夹

# 全局样式
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        font-size: 3em;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-size: 16px;
    }
    .stSlider {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 应用标题
st.markdown("<h1>CS506: Image Search Application</h1>", unsafe_allow_html=True)

# 加载 CLIP 模型和数据
@st.cache_resource
def initialize():
    model, preprocess_val = load_model()
    df, embeddings, file_names = load_data(EMBEDDING_PATH, IMAGE_FOLDER)
    embeddings = normalize_features(embeddings)

    # 准备 PCA 特征
    pca_k = 100
    pca, reduced_embeddings = get_pca_transform(embeddings, k=pca_k)
    reduced_embeddings = normalize_features(reduced_embeddings)
    return model, preprocess_val, df, embeddings, file_names, pca, reduced_embeddings

model, preprocess_val, df, embeddings, file_names, pca, reduced_embeddings = initialize()

# 分为两列布局
col1, col2 = st.columns([1, 1])

with col1:
    # 文本输入
    text_query = st.text_input("🔍 Enter a text query", "")

with col2:
    # 图像上传
    uploaded_file = st.file_uploader(
        "📤 Upload an image query",
        type=["jpg", "jpeg", "png"]
    )

# 中间分隔线
st.markdown("<hr style='border:2px solid #4CAF50;'>", unsafe_allow_html=True)

# 滑块和选择框
lambda_val = st.slider("⚖️ Adjust text vs image weight (λ)", 0.0, 1.0, 0.5, key="slider")
feature_type = st.selectbox("🛠️ Choose feature type", ["CLIP", "PCA"])

# 搜索按钮
if st.button("Search 🚀"):
    text_query_vec = None
    image_query_vec = None

    # 文本查询
    if text_query.strip():
        text_query_vec = encode_text(model, [text_query])

    # 图像查询
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        image_query_vec = encode_image(model, preprocess_val, img)

    # 合成最终查询向量
    if text_query_vec is not None and image_query_vec is not None:
        combined = lambda_val * text_query_vec + (1 - lambda_val) * image_query_vec
        query_vector = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    elif text_query_vec is not None:
        query_vector = text_query_vec
    elif image_query_vec is not None:
        query_vector = image_query_vec
    else:
        st.error("⚠️ Please provide at least one query (text or image).")
        st.stop()

    # 根据特征类型处理查询
    if feature_type == "PCA":
        query_vector_pca = pca.transform(query_vector)
        query_vector_pca = query_vector_pca / np.linalg.norm(query_vector_pca, axis=1, keepdims=True)
        scores = compute_similarity(query_vector_pca, reduced_embeddings)
    else:
        scores = compute_similarity(query_vector, embeddings)

    # 获取并显示前 k 个结果
    results = get_topk_images(scores, file_names, k=5)

    st.subheader("Search Results:")
    for fname, score in results:
        st.write(f"Image: {fname}, Score: {score:.4f}")
        img_path = os.path.join(IMAGE_FOLDER, fname)
        st.image(img_path, width=200)
