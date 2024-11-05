from flask import Flask, render_template, request
import numpy as np
import cv2
from PIL import Image
import base64
import boto3
from io import BytesIO
import os

app = Flask(__name__)

class FaceLandmarkProcessor2:
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name='ap-northeast-1'):
        self.rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
    
    def detect_faces_landmark(self, image_bytes):
        """
        画像バイトデータから顔のランドマークを検出
        """
        try:
            faces = self.rekognition_client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            
            if not faces['FaceDetails']:
                raise ValueError("顔が検出されませんでした")
                
            landmarks = faces['FaceDetails'][0]['Landmarks']
            eye_types = ['leftEyeLeft', 'leftEyeRight', 'leftEyeUp', 'leftEyeDown',
                        'rightEyeLeft', 'rightEyeRight', 'rightEyeUp', 'rightEyeDown']
            
            # OpenCV画像に変換
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            
            return {
                'landmarks': {
                    landmark['Type']: {'X': int(landmark['X'] * w), 'Y': int(landmark['Y'] * h)}
                    for landmark in landmarks if landmark['Type'] in eye_types
                },
                'image': img
            }
        except Exception as e:
            raise Exception(f"顔認識処理でエラーが発生しました: {str(e)}")

    def mosaic_area(self, src, x, y, width, height, blur_num):
        """
        指定された領域にモザイク処理を適用
        """
        dst = src.copy()
        if x < 0 or y < 0 or x + width > src.shape[1] or y + height > src.shape[0]:
            return dst
        
        for _ in range(blur_num):
            dst[y:y + height, x:x + width] = cv2.GaussianBlur(
                dst[y:y + height, x:x + width], (3, 3), 3
            )
        return dst

    def process_image(self, image_data, magnification=1.5, blur_num=3):
        """
        画像処理のメイン関数
        """
        try:
            result = self.detect_faces_landmark(image_data)
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
                                 int(left_eye.shape[0]*magnification)))
            
            # 右目の拡大処理
            right_eye = im[right_coords['top']:right_coords['bottom']+padding_y,
                          right_coords['left']-padding_x:right_coords['right']+padding_x]
            right_eye = cv2.resize(right_eye,
                                 (right_eye.shape[1],
                                  int(right_eye.shape[0]*magnification)))
            
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
                im = self.mosaic_area(im, *area, blur_num)
            
            return im
            
        except Exception as e:
            raise Exception(f"画像処理でエラーが発生しました: {str(e)}")

@app.route('/')
def hello():
    return render_template('results.html')

@app.route('/process-image', methods=['GET', 'POST'])
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
        
        file = request.files['example']
        if file.filename == '':
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
        processor = FaceLandmarkProcessor2(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        processed_image = processor.process_image(image_bytes)
        
        # 処理済み画像をbase64エンコード
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