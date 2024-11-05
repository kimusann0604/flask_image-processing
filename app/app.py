from flask import Flask,render_template
from flask import Flask, request, send_file, jsonify
import boto3
import cv2
import numpy as np
import os
from io import BytesIO


app = Flask(__name__, static_folder='./img')  # static_folderをappディレクトリに設定

UPLOAD_FOLDER = 'app/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
    
@app.route('/')
def hello():
    return ('hello word')

class FaceLandmarkProcessor:
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name='ap-northeast-1'):
        self.rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
    
    def detect_faces_landmark(self, image_bytes):
        response = self.rekognition_client.detect_faces(
            Image={'Bytes': image_bytes}, 
            Attributes=['ALL']
        )
        return response

    def draw_double_eye_ellipse(self, img, eye_center, size, angle, start_angle, end_angle, color=(68, 92, 135), thickness=2):
        cv2.ellipse(
            img,
            eye_center,
            size,
            angle,
            start_angle,
            end_angle,
            color,
            thickness
        )

    def left_eye_point(self, landmarks):
        left_eye_left = landmarks['leftEyeLeft']
        left_eye_right = landmarks['leftEyeRight']
        left_eye_up = landmarks['leftEyeUp']
        left_eye_down = landmarks['leftEyeDown']
        return left_eye_left, left_eye_right, left_eye_up, left_eye_down
    
    def right_eye_point(self, landmarks):
        right_eye_left = landmarks['rightEyeLeft']
        right_eye_right = landmarks['rightEyeRight']
        right_eye_up = landmarks['rightEyeUp']
        right_eye_down = landmarks['rightEyeDown']
        return right_eye_left, right_eye_right, right_eye_up, right_eye_down
    
    def calculate_Eye_Position_Draw(self, image, landmarks, roll_angle, height_factor=0.99):
        left_eye_left, left_eye_right, left_eye_up, left_eye_down = self.left_eye_point(landmarks)
        right_eye_left, right_eye_right, right_eye_up, right_eye_down = self.right_eye_point(landmarks)
        
        left_eye_center = ((left_eye_left[0] + left_eye_right[0]) // 2, 
                           (left_eye_up[1] + left_eye_down[1]) // 2)
        
        right_eye_center = ((right_eye_left[0] + right_eye_right[0]) // 2, 
                            (right_eye_up[1] + right_eye_down[1]) // 2)
        
        left_eye_width = left_eye_right[0] - left_eye_left[0]
        left_eye_height = left_eye_down[1] - left_eye_up[1]
        
        right_eye_width = right_eye_right[0] - right_eye_left[0]
        right_eye_height = right_eye_down[1] - right_eye_up[1]

        right_eyelid_height = int(right_eye_up[1] * height_factor)
        left_eyelid_height = int(left_eye_up[1] * height_factor)
        
        right_double_eyelid_center = (right_eye_center[0] + 8, right_eyelid_height + 12)
        left_double_eyelid_center = (left_eye_center[0] - 4, left_eyelid_height + 13)
        
        double_eye_width = 16
        double_eye_height = 9
        start_angle = 190
        end_angle = 360
        
        self.draw_double_eye_ellipse(image, left_double_eyelid_center, 
                                    (left_eye_width - double_eye_width, left_eye_height + double_eye_height),
                                    roll_angle, start_angle, end_angle, color=(68, 92, 135), thickness=2)
        
        self.draw_double_eye_ellipse(image, right_double_eyelid_center, 
                                    (right_eye_width - double_eye_width, right_eye_height + double_eye_height),
                                    roll_angle, start_angle, end_angle, color=(68, 92, 135), thickness=2)
        
    def draw_landmarks(self, image, face_details, height, width):
        for face_detail in face_details:
            landmarks = {landmark['Type']: (int(landmark['X'] * width), int(landmark['Y'] * height)) 
                        for landmark in face_detail['Landmarks']}
        
        pose = face_detail['Pose']
        roll_angle = pose['Roll']
        
        self.calculate_Eye_Position_Draw(image, landmarks, roll_angle)


@app.route('/process-image', methods=['GET', 'POST'])
def process_image():
    if request.method == 'GET':
        return render_template('file_upload.html')
    
    elif request.method == 'POST':
        try:
            if 'example' not in request.files:
                return jsonify({"error": "ファイルがアップロードされていません"}), 400
                
            file = request.files['example']
            if file.filename == '':
                return jsonify({"error": "ファイルが選択されていません"}), 400

            # ファイル名の安全性確保
            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            
            # オリジナル画像の保存
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{filename}')
            file.save(original_path)
            
            # 画像を読み込み
            with open(original_path, 'rb') as f:
                image_bytes = f.read()

            # AWS認証情報の確認
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            if not aws_access_key_id or not aws_secret_access_key:
                return jsonify({"error": "AWS認証情報が設定されていません"}), 500

            # 顔認識処理
            face_processor = FaceLandmarkProcessor(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )

            # 画像処理
            image_np = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({"error": "画像の読み込みに失敗しました"}), 400

            height, width = image.shape[:2]
            response = face_processor.detect_faces_landmark(image_bytes)
            
            if 'FaceDetails' in response and len(response['FaceDetails']) > 0:
                face_processor.draw_landmarks(image, response['FaceDetails'], height, width)
    
                # 処理済み画像の保存
                processed_filename = f'processed_{filename}'
                processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_path, image)
                
                # URLを生成（imgディレクトリからの相対パス）
                original_url = f'img/original_{filename}'
                processed_url = f'img/processed_{filename}'
                
                return render_template('results.html', 
                                     original_filename=original_url,
                                     processed_filename=processed_url,
                                     result="顔認識処理が完了しました")
            else:
                return render_template('results.html',
                                     original_filename=f'img/original_{filename}',
                                     result="顔を検出できませんでした")

        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            return jsonify({"error": f"画像処理中にエラーが発生しました: {str(e)}"}), 500
        
@app.route('/results_2.html')
def index():
    return render_template('results_2.html')

if __name__ == "__main__":
    app.run(debug=True, port=5003)