import os
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from flask import Flask, render_template, request
from sklearn.svm import SVC
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = joblib.load('svm_pet.joblib')

pre_trained = models.resnet18(pretrained=True)
feat_extractor = nn.Sequential(*list(pre_trained.children())[:-1])
feat_extractor.to(device)
feat_extractor.eval()

transform_image = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224,224)),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def transform_img(img_path):
    img = Image.open(img_path)
    img_t = transform_image(img)
    arr = feat_extractor(img_t.unsqueeze(0).to(device))
    arr = torch.reshape(arr, (1, 512))
    pred = classifier.predict(arr.detach().cpu().numpy())
    return pred


UPLOAD_FOLDER = r'D:\Waldir\Cursos\Facens\ML\Deploy\Aula\upload'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # slashes should be handeled properly
        file.save(file_path)
        print(filename)
        product = transform_img(file_path)
        product = 'Cat' if product[0] == 0 else 'Dog'
        print(product)

    return render_template('predict.html', product=product,
                           user_image=file_path)  # file_path can or may used at the place of filename


if __name__ == "__main__":
    app.run()
