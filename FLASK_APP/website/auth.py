from flask import Blueprint, render_template, request
from predict import predict
import os

auth = Blueprint('auth', __name__)


def saveimg(img):
    # đường dẫn đến thư mục uploads
    directory = "./website/static/uploads/"

    # nếu thư mục không tồn tại, tạo mới thư mục
    if not os.path.exists(directory):
        os.makedirs(directory)

    # cấp quyền truy cập cho thư mục
    os.chmod(directory, 0o777)
    img_path = directory + img.filename

    img.save(img_path)

    if os.path.exists(img_path):
        print("Hình ảnh đã được lưu trữ thành công!")
        print("Đường dẫn: ", img_path)
    else:
        print("Có lỗi xảy ra khi lưu trữ hình ảnh!")
    return img_path

@auth.route('/', methods=['POST'])
def result():
    img = request.files['inputFile']
    img_path = saveimg(img)
    pred = predict(img_path)
    img_src = 'uploads/' + img.filename
    return render_template("index.html", prediction = pred, image_url = img_src)