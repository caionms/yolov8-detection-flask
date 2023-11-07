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
   return redirect(url_for('predict_image'))

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
    
# Executando aplicacao
if __name__ == '__main__':
    app.run(debug=True)