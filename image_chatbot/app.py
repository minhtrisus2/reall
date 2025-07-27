import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình nhận dạng hình ảnh đã được huấn luyện sẵn
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Cấu hình thư mục để tải ảnh lên
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Hàm xử lý và dự đoán hình ảnh
def predict_image(image_path):
    try:
        img = Image.open(image_path).resize((224, 224))
        # Chuyển ảnh sang định dạng RGB nếu là ảnh GIF hoặc có kênh alpha
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

        return decoded_predictions
    except Exception as e:
        print(f"Lỗi xử lý ảnh: {e}")
        return None

# Trang chủ và xử lý việc tải ảnh lên
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predictions = predict_image(filepath)
            if predictions:
                result = [(label.replace('_', ' ').title(), f"{prob*100:.2f}%") for (_, label, prob) in predictions]
                return render_template('index.html', filename=filename, predictions=result)
            else:
                error_message = "Không thể xử lý tệp ảnh này. Vui lòng thử một ảnh khác."
                return render_template('index.html', error=error_message)
        else:
            error_message = "Tệp không hợp lệ! Vui lòng chỉ tải lên tệp ảnh (.png, .jpg, .jpeg, .gif)."
            return render_template('index.html', error=error_message)

    return render_template('index.html')

# Route để hiển thị ảnh đã tải lên
@app.route('/uploads/<filename>')
def send_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    # Tạo các thư mục cần thiết nếu chưa tồn tại
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

    import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Sử dụng cache để không phải tải lại mô hình mỗi lần chạy
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

model = load_model()

# Hàm dự đoán (gần như không đổi)
def predict_image(image):
    # Chuyển ảnh sang định dạng RGB nếu cần
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Xử lý ảnh để phù hợp với đầu vào của mô hình
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Thực hiện dự đoán
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    
    return decoded_predictions

# --- Bắt đầu xây dựng giao diện ---

st.set_page_config(page_title="Bot Nhận Dạng Ảnh", page_icon="🤖")

st.title("🤖 Chatbot Nhận Dạng Hình Ảnh")
st.write("Tải lên một bức ảnh và tôi sẽ cho bạn biết tôi thấy gì trong đó!")

# Tạo thành phần để người dùng tải file lên
uploaded_file = st.file_uploader(
    "Chọn một tệp ảnh...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Đọc và hiển thị ảnh người dùng tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn đã tải lên.", use_column_width=True)

    st.write("") # Thêm một khoảng trống

    # Hiển thị trạng thái đang xử lý và thực hiện dự đoán
    with st.spinner("Bot đang phân tích... 🕵️"):
        predictions = predict_image(image)

    st.success("Đây là những gì tôi dự đoán!")

    # Hiển thị kết quả
    for i, (_, label, prob) in enumerate(predictions):
        st.write(f"**Dự đoán {i+1}:** `{label.replace('_', ' ').title()}` - Độ chính xác: `{prob*100:.2f}%`")