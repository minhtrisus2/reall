import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io # Thêm thư viện io để xử lý dữ liệu ảnh trong bộ nhớ

# Cấu hình các định dạng ảnh cho phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Hàm tải mô hình nhận dạng hình ảnh.
# Sử dụng @st.cache_resource để chỉ tải mô hình một lần duy nhất khi ứng dụng khởi động,
# giúp cải thiện hiệu suất vì Streamlit sẽ chạy lại toàn bộ script trên mỗi tương tác.
@st.cache_resource
def load_recognition_model():
    """Tải và trả về mô hình MobileNetV2 đã được huấn luyện sẵn."""
    try:
        model_instance = tf.keras.applications.MobileNetV2(weights='imagenet')
        return model_instance
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop() # Dừng ứng dụng nếu không thể tải mô hình

# Tải mô hình
model = load_recognition_model()

# Hàm xử lý và dự đoán hình ảnh
def predict_image(image_file):
    """
    Dự đoán nội dung của hình ảnh sử dụng mô hình đã tải.

    Args:
        image_file: Đối tượng file ảnh được tải lên từ st.file_uploader.

    Returns:
        Danh sách các dự đoán (nhãn, độ tin cậy) hoặc None nếu có lỗi.
    """
    try:
        # Mở hình ảnh từ đối tượng file được tải lên và thay đổi kích thước
        img = Image.open(io.BytesIO(image_file.read())).resize((224, 224))
        
        # Chuyển ảnh sang định dạng RGB nếu cần (quan trọng cho ảnh GIF hoặc PNG có kênh alpha)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Chuyển đổi ảnh sang mảng numpy và tiền xử lý cho mô hình
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Thêm chiều batch
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Thực hiện dự đoán
        predictions = model.predict(img_array)
        
        # Giải mã các dự đoán và lấy 3 kết quả hàng đầu
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

        return decoded_predictions
    except Exception as e:
        st.error(f"Lỗi xử lý ảnh hoặc dự đoán: {e}")
        return None

# --- Giao diện người dùng Streamlit ---

st.set_page_config(layout="centered") # Cấu hình bố cục trang

st.title("Ứng dụng Nhận diện Hình ảnh")
st.markdown("---")

st.write("Tải lên một hình ảnh và tôi sẽ cố gắng nhận diện nó!")

# Bộ tải tệp cho phép người dùng chọn hình ảnh
uploaded_file = st.file_uploader(
    "Chọn một hình ảnh...", 
    type=list(ALLOWED_EXTENSIONS), # Chuyển set sang list cho tham số type
    help="Chỉ cho phép các định dạng: PNG, JPG, JPEG, GIF"
)

# Xử lý khi có file được tải lên
if uploaded_file is not None:
    # Hiển thị hình ảnh đã tải lên
    # ĐÃ THAY ĐỔI: use_column_width=True thành use_container_width=True
    st.image(uploaded_file, caption="Hình ảnh đã tải lên.", use_container_width=True)
    st.markdown("---")

    # Hiển thị spinner trong khi dự đoán
    with st.spinner('Đang phân tích hình ảnh...'):
        # Gọi hàm dự đoán
        predictions = predict_image(uploaded_file)

    if predictions:
        st.subheader("Kết quả dự đoán:")
        result_list = [
            (label.replace('_', ' ').title(), f"{prob*100:.2f}%") 
            for (_, label, prob) in predictions
        ]
        
        # Hiển thị các dự đoán
        for item, prob in result_list:
            st.markdown(f"- **{item}**: `{prob}`")
        st.success("Phân tích hoàn tất!")
    else:
        st.warning("Không thể dự đoán nội dung của hình ảnh này. Vui lòng thử một hình ảnh khác.")
else:
    st.info("Vui lòng tải lên một hình ảnh để bắt đầu.")

st.markdown("---")
st.caption("Ứng dụng được xây dựng bằng Streamlit và TensorFlow (MobileNetV2).")
