from flask import Flask,render_template
from flask import Flask, request, send_file, jsonify
import boto3
import cv2
import numpy as np
import os
from io import BytesIO
import tempfile
from PIL import Image
import io
import base64


app = Flask(__name__)  
    
@app.route('/')
def hello():
    return render_template('index.html')

class FaceLandmarkProcessor:
    ''''二重を作成するclass'''
    
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name='ap-northeast-1'):
        #awsのアクセスキーをosから読み込む
        
        self.rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
    
    def detect_faces_landmark(self, image_bytes):
        #顔認証をするための関数
        
        response = self.rekognition_client.detect_faces(
            Image={'Bytes': image_bytes}, 
            Attributes=['ALL']
        )
        return response

    def draw_double_eye_ellipse(self, img, eye_center, size, angle, start_angle, end_angle, color=(68, 92, 135), thickness=2):
        #opencvの描画機能で二重の線を描く
        
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
        #左目、４点の座標を取ってくる
        
        left_eye_left = landmarks['leftEyeLeft']
        left_eye_right = landmarks['leftEyeRight']
        left_eye_up = landmarks['leftEyeUp']
        left_eye_down = landmarks['leftEyeDown']
        return left_eye_left, left_eye_right, left_eye_up, left_eye_down
    
    def right_eye_point(self, landmarks):
        #右目、４点の座標を取ってくる
        
        right_eye_left = landmarks['rightEyeLeft']
        right_eye_right = landmarks['rightEyeRight']
        right_eye_up = landmarks['rightEyeUp']
        right_eye_down = landmarks['rightEyeDown']
        return right_eye_left, right_eye_right, right_eye_up, right_eye_down
    
    def calculate_Eye_Position_Draw(self, image, landmarks, roll_angle, height_factor=0.99):
        #二重線の位置を調整するための関数
        
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
        #ランドマークの情報を読み込み、首の傾きから線を傾かせる
        for face_detail in face_details:
            landmarks = {landmark['Type']: (int(landmark['X'] * width), int(landmark['Y'] * height)) 
                        for landmark in face_detail['Landmarks']}
        
        pose = face_detail['Pose']
        roll_angle = pose['Roll']
        
        self.calculate_Eye_Position_Draw(image, landmarks, roll_angle)


@app.route('/process-image', methods=['GET', 'POST'])
#/process-imageにアクセスし、ボタンを押すことでPOSTを実行。その後画像処理

def process_image():
    if request.method == 'GET':
        return render_template('results.html')
    
    elif request.method == 'POST':
        try:
            if 'example' not in request.files:
                return render_template('results.html', img_data=None, result="ファイルがアップロードされていない")
                
            file = request.files['example']
            if file.filename == '':
                return render_template('results.html', img_data=None, result="ファイルが選択されていない")

            # ファイルをバイナリデータに変換
            image_bytes = file.read()
            
            # AWS認証情報の確認
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            if not aws_access_key_id or not aws_secret_access_key:
                return render_template('results.html', img_data=None, result="AWS認証情報が設定されていません")

            # Rekognitionによる顔認識
            face_processor = FaceLandmarkProcessor(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            response = face_processor.detect_faces_landmark(image_bytes)
            
            # PILで画像を読み込み、OpenCV形式に変換
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            if 'FaceDetails' in response and len(response['FaceDetails']) > 0:
                height, width = image_np.shape[:2]
                face_processor.draw_landmarks(image_np, response['FaceDetails'], height, width)

                # 処理済み画像をbase64にエンコードして埋め込み表示
                _, buffer = cv2.imencode('.png', image_np)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                img_data_uri = f"data:image/png;base64,{img_base64}"

                return render_template('results.html', img_data=img_data_uri, result="顔認識処理が完了しました")
        
        except Exception as e:
            return render_template('results.html', img_data=None, result=f"エラーが発生しました: {str(e)}")
        


class PtosisCorrection:
    '''眼瞼下垂を行うためのclass'''
    
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name='ap-northeast-1'):
        self.rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
    
    def detect_eye_landmarks(self, image_bytes):
        #画像バイトデータから顔のランドマークを検出
        
        try:
            faces = self.rekognition_client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            
            if not faces['FaceDetails']:
                raise ValueError("顔が検出されませんでした")
            
            # 顔のランドマークを表示
            landmarks = faces['FaceDetails'][0]['Landmarks']
            eye_points = ['leftEyeLeft', 'leftEyeRight', 'leftEyeUp', 'leftEyeDown',
                        'rightEyeLeft', 'rightEyeRight', 'rightEyeUp', 'rightEyeDown']
            
            
            # 画像データに適したものにするためにnumpy配列にしている
            binary_image_data = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(binary_image_data, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            
            return {
                'landmarks': {
                    landmark['Type']: {'X': int(landmark['X'] * w), 'Y': int(landmark['Y'] * h)}
                    for landmark in landmarks if landmark['Type'] in eye_points
                },
                'image': img
            }
                
        except Exception as e:
            raise Exception(f"顔認識処理でエラーが発生しました: {str(e)}")

    def mosaic_area(self, src, x, y, width, height, mosaic):
        #指定された領域にモザイク処理を適用
        
        dst = src.copy()
        if x < 0 or y < 0 or x + width > src.shape[1] or y + height > src.shape[0]:
            return dst
        
        for _ in range(mosaic):
            dst[y:y + height, x:x + width] = cv2.GaussianBlur(
                dst[y:y + height, x:x + width], (3, 3), 3
            )
        return dst

    def process_image(self, image_data, eye_magnification=1.5, mosaic=3):
        #eye_magnificationの数値を変えることで標準の目の大きさを変更
        #mosaicの数値を変更することでモザイクの範囲を設定
        
        try:
            result = self.detect_eye_landmarks(image_data)
            im = result['image']
            eye_points = result['landmarks']
            
            # 目の周辺領域の調整用パラメータ
            padding_x, padding_y = 20, 5
            
            # 左目の処理
            left_coords = {
                'top': min(eye_points[key]['Y'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft']),
                'bottom': max(eye_points[key]['Y'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft']),
                'right': max(eye_points[key]['X'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft']),
                'left': min(eye_points[key]['X'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft'])
            }
            
            # 右目の処理
            right_coords = {
                'top': min(eye_points[key]['Y'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft']),
                'bottom': max(eye_points[key]['Y'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft']),
                'right': max(eye_points[key]['X'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft']),
                'left': min(eye_points[key]['X'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft'])
            }
            
            # 左目の拡大処理
            left_eye = im[left_coords['top']:left_coords['bottom']+padding_y,
                         left_coords['left']-padding_x:left_coords['right']+padding_x]
            left_eye = cv2.resize(left_eye, 
                                (left_eye.shape[1], 
                                 int(left_eye.shape[0]* eye_magnification)))
            
            # 右目の拡大処理
            right_eye = im[right_coords['top']:right_coords['bottom']+padding_y,
                          right_coords['left']-padding_x:right_coords['right']+padding_x]
            right_eye = cv2.resize(right_eye,
                                 (right_eye.shape[1],
                                  int(right_eye.shape[0]* eye_magnification)))
            
            # 拡大した目を元の画像に配置
            im[left_coords['top']:left_coords['top']+left_eye.shape[0],
               left_coords['left']-padding_x:left_coords['left']+left_eye.shape[1]-padding_x] = left_eye
            
            im[right_coords['top']:right_coords['top']+right_eye.shape[0],
               right_coords['left']-padding_x:right_coords['left']+right_eye.shape[1]-padding_x] = right_eye
            
            # ぼかし処理を適用する領域の定義
            blur_areas = [
                # 左目周辺
                (left_coords['left']-padding_x-int(padding_x/2),
                 left_coords['top'],
                 padding_x,
                 left_eye.shape[0]+padding_y),
                
                (left_coords['right']+int(padding_x/2),
                 left_coords['top'],
                 padding_x,
                 left_eye.shape[0]+padding_y),
                
                (left_coords['left']-padding_x,
                 left_coords['top']+left_eye.shape[0]-int(padding_y/2),
                 left_eye.shape[1],
                 padding_y),
                
                # 右目周辺
                (right_coords['left']-padding_x-int(padding_x/2),
                 right_coords['top'],
                 padding_x,
                 right_eye.shape[0]+padding_y),
                
                (right_coords['right']+int(padding_x/2),
                 right_coords['top'],
                 padding_x,
                 right_eye.shape[0]+padding_y),
                
                (right_coords['left']-padding_x,
                 right_coords['top']+right_eye.shape[0]-int(padding_y/2),
                 right_eye.shape[1],
                 padding_y)
            ]
            
            # 各領域にぼかし処理を適用
            for area in blur_areas:
                im = self.mosaic_area(im, *area, mosaic)
            
            return im
            
        except Exception as e:
            raise Exception(f"画像処理でエラーが発生しました: {str(e)}")


@app.route('/eye-process', methods=['GET', 'POST'])
#/eye-processにアクセスし、画像をアップロードすることでPOSTを実行その後顔認証

def process_image():
    if request.method == 'GET':
        return render_template('results.html',
                             img_data=None,
                             result="ファイルをアップロードしてください")
    
    try:
        if 'example' not in request.files:
            return render_template('results.html',
                                 img_data=None,
                                 result="ファイルがアップロードされていません")
        # print(request.files)
        #出力：ImmutableMultiDict([('example', <FileStorage: 'image.jpg' ('image/jpeg')>)])
        # request.filesの中に画像のデータが入っている
        file = request.files['example']
        print(file)
        # 出力：<FileStorage: 'bbc6c9066fa41d8de797b46e34d91a39.jpg' ('image/jpeg')>
        if file.filename == '':
            # もしファイルの中にファイルが選択されていなかったら
            return render_template('results.html',
                                 img_data=None,
                                 result="ファイルが選択されていません")
        
        # ファイルをバイナリデータとして読み込み
        image_bytes = file.read()
        
        # AWS認証情報の取得
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not aws_access_key_id or not aws_secret_access_key:
            return render_template('results.html',
                                 img_data=None,
                                 result="AWS認証情報が設定されていません")
        
        # 画像処理の実行
        processor = PtosisCorrection(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        processed_image = processor.process_image(image_bytes)
        
        # 画像をbase64エンコードし、HTMLに読み込む
        _, buffer = cv2.imencode('.png', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_uri = f"data:image/png;base64,{img_base64}"
        
        return render_template('results.html',
                             img_data=img_data_uri,
                             result="画像処理が完了しました")
    
    except Exception as e:
        return render_template('results.html',
                             img_data=None,
                             result=f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5003)
