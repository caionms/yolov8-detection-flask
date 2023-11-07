import base64
from flask import Flask, render_template, request, redirect, url_for
import requests
import moviepy.editor as mp
import os
from PIL import Image
from io import BytesIO
import cv2
from ultralytics import YOLO

# Instanciando a classe na variavel app
app = Flask(__name__)

# Definindo rotas da api (nesse caso home)
@app.route('/')
def home(): # Funcao executada quando a rota eh acessada
    return render_template('home.html')

# Outra rota/endpoint(ponto de acesso)
@app.route('/clear', methods=['POST'])
def clear():
    return redirect(url_for('home')) # Nome do método

# Outra rota/endpoint(ponto de acesso)
@app.route('/predict_image/', methods=['GET', 'POST'])
def predict_image():
    result = None
    if request.method == 'POST':
        image_url = request.form['image_url']
        result = process_image(image_url)
    return render_template('predict_image.html', result=result)

def process_image(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            
            # Load a model
            model = YOLO('yolov8n-pose.pt')
            results = model(image)
            im_array = results[0].plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            
            buffered = BytesIO()
            im.save(buffered, format="JPEG")
            result_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return result_image
        else:
            return "Erro ao acessar a imagem."
    except Exception as e:
        return str(e)

# Outra rota/endpoint(ponto de acesso)
@app.route('/predict_video/', methods=['GET', 'POST'])
def predict_video():
    result_video = None
    if request.method == 'POST':
        video_url = request.form['video_url']
        result_video = process_video(video_url)
    return render_template('predict_video.html', result_video=result_video)

def process_video(video_url):
    try:
        response = requests.get(video_url)
        if response.status_code == 200:
            video_path = 'temp_video.mp4'
            with open(video_path, 'wb') as f:
                f.write(response.content)

            # Faça o processamento do vídeo aqui (por exemplo, cortar um trecho)
            processed_video = process_video_logic(video_path)

            # Remova o arquivo de vídeo temporário
            os.remove(video_path)

            return processed_video
        else:
            return "Erro ao acessar o vídeo."
    except Exception as e:
        return str(e)
    
def process_video_logic(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Obtem o framerate para contabilizar o tempo
    fps = None
    if cap is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        fps = 30
        
    # Define VideoWriter para salvar video de saida
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    filename = 'processed_video.mp4'
    out = cv2.VideoWriter(filename, fourcc, (30.0 if fps is None else fps), (int(cap.get(3)), int(cap.get(4))))
    
    # Load a model
    model = YOLO('yolov8n-pose.pt')
    
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

             # Escreve o frame no video de saida
            frame = cv2.resize(annotated_frame, (int(cap.get(3)), int(cap.get(4))))
            out.write(frame)
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()

    return out

# Executando aplicacao
if __name__ == '__main__':
    app.run(debug=True)