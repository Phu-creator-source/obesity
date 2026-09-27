import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Phân loại Tình trạng Béo phì",
    page_icon="⚖️",
    layout="wide"
)

# Thêm tiêu đề và mô tả
st.title("⚖️ Dẫn đoán & Phân loại Tình trạng Béo phì")
st.write("Ứng dụng dự đoán phân loại thể trạng dựa trên thuật toán Logistic Regression.")

# 1. Huấn luyện mô hình (sử dụng cache để tối ưu hiệu năng)
@st.cache_resource
def train_model():
    # Đọc dữ liệu
    df = pd.read_csv("Obesity Classification.csv")
    
    # Ánh xạ nhãn Label
    label_map = {
        'Underweight': 0,
        'Normal Weight': 1,
        'Overweight': 2,
        'Obese': 3
    }
    df['Label'] = df['Label'].map(label_map)
    
    # Mã hóa biến định tính Gender (nam/nữ)
    category_col = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=category_col, drop_first=True)
    
    # Phân chia thuộc tính X và nhãn y
    X = df.drop(columns=["Label"])
    y = df["Label"]
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Huấn luyện mô hình Logistic Regression
    model = LogisticRegression(max_iter=100)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns

# Load mô hình và scaler
model, scaler, feature_columns = train_model()

# Ánh xạ kết quả ngược lại tên phân loại
inv_label_map = {
    0: "Thiếu cân (Underweight) 🦴",
    1: "Cân nặng bình thường (Normal Weight) ✅",
    2: "Thừa cân (Overweight) ⚠️",
    3: "Béo phì (Obese) 🚨"
}

# 2. Giao diện nhập thông tin từ người dùng
st.sidebar.header("📋 Nhập thông tin chỉ số")

user_id = st.sidebar.number_input("ID", min_value=1, value=1, step=1)
age = st.sidebar.number_input("Tuổi (Age)", min_value=1, max_value=120, value=25)
gender = st.sidebar.selectbox("Giới tính (Gender)", ["Male", "Female"])
height = st.sidebar.number_input("Chiều cao (cm)", min_value=50, max_value=250, value=170)
weight = st.sidebar.number_input("Cân nặng (kg)", min_value=10, max_value=250, value=65)

# Tự động tính chỉ số BMI
height_m = height / 100
bmi = round(weight / (height_m ** 2), 1)

# Hiển thị các thông số cơ bản ở màn hình chính
col1, col2, col3, col4 = st.columns(4)
col1.metric("Chiều cao", f"{height} cm")
col2.metric("Cân nặng", f"{weight} kg")
col3.metric("Tuổi", f"{age}")
col4.metric("BMI tính toán", f"{bmi}")

st.markdown("---")

# 3. Xử lý dự đoán
if st.button("🔍 Dự đoán tình trạng thể trạng", type="primary"):
    # Chuẩn bị dữ liệu đầu vào phù hợp với các cột dữ liệu gốc
    gender_male = True if gender == "Male" else False
    
    input_data = pd.DataFrame([{
        "ID": user_id,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "BMI": bmi,
        "Gender_Male": gender_male
    }])
    
    # Đảm bảo thứ tự các cột khớp với lúc huấn luyện
    input_data = input_data[feature_columns]
    
    # Chuẩn hóa dữ liệu đầu vào
    input_scaled = scaler.transform(input_data)
    
    # Thực hiện dự đoán
    prediction = model.predict(input_scaled)[0]
    result_text = inv_label_map[prediction]
    
    # Hiển thị kết quả
    st.subheader("📌 Kết quả dự đoán:")
    if prediction == 1:
        st.success(f"**{result_text}**")
    elif prediction == 0:
        st.info(f"**{result_text}**")
    elif prediction == 2:
        st.warning(f"**{result_text}**")
    else:
        st.error(f"**{result_text}**")