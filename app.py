import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load classifier
classifier = joblib.load('svm_pet.joblib')

# Load pretrained ResNet18
pre_trained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feat_extractor = nn.Sequential(*list(pre_trained.children())[:-1])
feat_extractor.to(device)
feat_extractor.eval()

# Image transformations
transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def transform_img(uploaded_img):
    img = Image.open(uploaded_img).convert("RGB")
    img_t = transform_image(img)
    arr = feat_extractor(img_t.unsqueeze(0).to(device))
    arr = torch.reshape(arr, (1, 512))
    pred = classifier.predict(arr.detach().cpu().numpy())
    return pred[0]

# -------------------------
# STREAMLIT APP
# -------------------------

st.title("🐶🐱 PetCheck - Classificação de Imagens")

st.write("Envie uma imagem de um **gato** ou **cachorro** para classificação.")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagem enviada", width=300)

    if st.button("Classificar"):
        pred = transform_img(uploaded_file)
        #result = "🐱 Gato" if pred == 0 else "🐶 Cachorro"

        if pred == 0:
            result = "🐶 Cachorro"
        elif pred == 1:
            result = "🐱 Gato"
        else:
             result = "Não é 🐶 Cachorro ou 🐱 Gato"

        st.subheader("Resultado:")
        st.success(result)
