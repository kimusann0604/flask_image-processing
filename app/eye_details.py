from flask import Flask,render_template
from flask import Flask, request, send_file, jsonify

import numpy as np
import cv2
import math
from PIL import Image
import pandas as pd
import base64
import boto3
from botocore.config import Config
from io import BytesIO  
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/image_second'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
    
@app.route('/')
def hello():
    return ('hello world')

class Eye_big:
    
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name= 'ap-northeast-1'):
        self.rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        
    def rekog_eye(self, im):
        
        _, buf = cv2.imencode('.jpg', im)
        faces = self.rekognition_client.detect_faces(
            Image={'Bytes':buf.tobytes()}, 
            Attributes=['ALL'])
        
        landmarks = faces['FaceDetails'][0]['Landmarks']
        eye_types = ['leftEyeLeft', 'leftEyeRight', 'leftEyeUp', 'leftEyeDown', 
                    'rightEyeLeft', 'rightEyeRight', 'rightEyeUp', 'rightEyeDown']
        
        h, w, _ = im.shape
        return {landmark['Type']: {'X': int(landmark['X'] * w), 'Y': int(landmark['Y'] * h)}
                for landmark in landmarks if landmark['Type'] in eye_types}

    def mosaic_area(self,src, x, y, width, height, blur_num):
        dst = src.copy()
        for _ in range(blur_num):
            dst[y:y + height, x:x + width] = cv2.GaussianBlur(dst[y:y + height, x:x + width], (3,3), 3)
        return dst

    def process_image(self,im, magnification, blur_num):
        EyePoints = self.rekog_eye(im)
        bityouseix, bityouseiy = 20, 5

        leftTop = min(EyePoints[key]['Y'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft'])
        leftBottom = max(EyePoints[key]['Y'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft'])
        leftRight = max(EyePoints[key]['X'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft'])
        leftLeft = min(EyePoints[key]['X'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft'])
        
        rightTop = min(EyePoints[key]['Y'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft'])
        rightBottom = max(EyePoints[key]['Y'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft'])
        rightRight = max(EyePoints[key]['X'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft'])
        rightLeft = min(EyePoints[key]['X'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft'])

        leftEye = im[leftTop:leftBottom+bityouseiy, leftLeft-bityouseix:leftRight+bityouseix]
        leftEye = cv2.resize(leftEye, (leftEye.shape[1], int(leftEye.shape[0]*magnification)))
        rightEye = im[rightTop:rightBottom+bityouseiy, rightLeft-bityouseix:rightRight+bityouseix]
        rightEye = cv2.resize(rightEye, (rightEye.shape[1], int(rightEye.shape[0]*magnification)))

        im[leftTop:leftTop+leftEye.shape[0], leftLeft-bityouseix:leftLeft+leftEye.shape[1]-bityouseix] = leftEye
        im[rightTop:rightTop+rightEye.shape[0], rightLeft-bityouseix:rightLeft+rightEye.shape[1]-bityouseix] = rightEye

        blur_areas = [
            (leftLeft-bityouseix-int(bityouseix/2), leftTop, bityouseix, leftEye.shape[0]+bityouseiy),
            (leftRight+int(bityouseix/2), leftTop, bityouseix, leftEye.shape[0]+bityouseiy),
            (leftLeft-bityouseix, leftTop+leftEye.shape[0]-int(bityouseiy/2), leftEye.shape[1], bityouseiy),
            (rightLeft-bityouseix-int(bityouseix/2), rightTop, bityouseix, rightEye.shape[0]+bityouseiy),
            (rightRight+int(bityouseix/2), rightTop, bityouseix, rightEye.shape[0]+bityouseiy),
            (rightLeft-bityouseix, rightTop+rightEye.shape[0]-int(bityouseiy/2), rightEye.shape[1], bityouseiy)
        ]

        for area in blur_areas:
            im = self.mosaic_area(im, *area, blur_num)

        return im
    

@app.route('/process-image_second' methods=['GET','POST'])
def process_image_second():
        if request.method == 'GET':
            return render_template('file_upload.html')
    
        elif request.method == 'POST':
            try:
                if 'example' not in request.files:
                    return jsonify({"error": "ファイルがアップロードされていません"}), 400
                    
                file = request.files['example']
                if file.filename == '':
                    return jsonify({"error": "ファイルが選択されていません"}), 400
                
                print(file.filename)

                # ファイル名の安全性確保(このファイル読み込んで大丈夫かなど)
                from werkzeug.utils import secure_filename
                filename = secure_filename(file.filename)
                
                
                # ファイルを保存
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                

                # 画像を読み込み
                with open(image_path, 'rb') as f:
                    im = f.read()
                    

                # AWS認証情報の確認
                aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
                aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
                if not aws_access_key_id or not aws_secret_access_key:
                    return jsonify({"error": "AWS認証情報が設定されていません。環境変数を確認してください。"}), 500
                
                Eye_big = Eye_big(
                    aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
                )
                
        

