import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import os

# Função para carregar o modelo
@st.cache_resource
def load_model():
    # Use um caminho relativo para o modelo no repositório do Streamlit Cloud
    return tf.keras.models.load_model('inception_transfer.keras')

labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Função de predição
def predict(model, img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Normalização após redimensionamento
    img = np.array(img)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalização dividindo por 255
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img)
    
    return pred[0]

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  # Garantir 3 canais RGB
    img = img.resize((150, 150))
    img = np.array(img) / 255.0  # Normalizar a imagem
    return img

# Carregar os dados de teste
@st.cache_data
def load_test_data(base_path):
    X_test = []
    y_test = []
    for i, label in enumerate(labels):
        label_path = os.path.join(base_path, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            try:
                img = load_and_preprocess_image(img_path)
                X_test.append(img)
                y_test.append(i)
            except Exception as e:
                st.warning(f"Não foi possível carregar a imagem {img_path}: {str(e)}")
    return np.array(X_test), np.array(y_test)

# Avaliar o modelo
@st.cache_data
def evaluate_model(model, X_test, y_test):
    # Converter os rótulos para o formato one-hot encoded
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=len(labels))
    
    # Avaliar o modelo com X_test e rótulos one-hot
    results = model.evaluate(X_test, y_test_one_hot, verbose=0)
    loss, accuracy = results[0], results[1]
    
    # Prever as classes do conjunto de teste
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calcular métricas adicionais
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    # Curva ROC
    y_test_bin = tf.keras.utils.to_categorical(y_test, num_classes=len(labels))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return loss, accuracy, precision, recall, f1, conf_matrix, fpr, tpr, roc_auc

# Sidebar for navigation
st.sidebar.title("Navegação")
page = st.sidebar.selectbox("Selecione a página", ["Página Inicial", "Previsão de Tumor", "Estatísticas", "Artigo"])

# Página Inicial
if page == "Página Inicial":
    st.title("Projeto de Detecção de Tumor Cerebral")
    st.write("""
        Este projeto utiliza o modelo InceptionV3 treinado para a detecção de diferentes tipos de tumores cerebrais 
        em imagens de ressonância magnética (MRI). O objetivo é auxiliar no diagnóstico médico, fornecendo uma predição
        sobre a presença e o tipo de tumor, com uma confiança associada.
        
        Na página de previsão, você poderá carregar uma imagem de ressonância magnética e o modelo irá classificar o tipo de tumor 
        entre as quatro classes: **Glioma Tumor**, **No Tumor**, **Meningioma Tumor** e **Pituitary Tumor**.
    """)
    
    # Add a section about the importance of the project
    st.header("Importância do Projeto")
    st.write("""
        A detecção precoce de tumores cerebrais é crucial para o tratamento eficaz e a melhoria dos resultados dos pacientes. 
        Este projeto visa:
        
        1. Auxiliar médicos na triagem rápida de imagens de MRI.
        2. Reduzir o tempo de diagnóstico.
        3. Aumentar a precisão na identificação de diferentes tipos de tumores.
        4. Potencialmente salvar vidas através da detecção precoce.
    """)
    
    # Add a section about how to use the app
    st.header("Como Usar o Aplicativo")
    st.write("""
        1. Navegue até a página "Previsão de Tumor" usando o menu lateral.
        2. Carregue uma imagem de MRI cerebral.
        3. Aguarde a análise do modelo.
        4. Veja os resultados da predição e a confiança para cada classe.
        5. Explore as estatísticas gerais na página "Estatísticas".
        6. Leia o artigo completo na página "Artigo" para mais detalhes sobre a metodologia.
    """)

# Página de Previsão de Tumor
elif page == "Previsão de Tumor":
    st.title("Detecção de Tumor Cerebral com InceptionV3")
    st.write("Carregue uma imagem de ressonância magnética (MRI) para detectar o tipo de tumor cerebral.")
    
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem Carregada.', use_column_width=True)
        st.write("")
        st.write("Classificando...")
        
        model = load_model()  # Carregar o modelo do cache
        predictions = predict(model, image)
        
        # Criar gráfico de barras para exibir as predições
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=predictions,
            text=[f'{p:.2%}' for p in predictions],
            textposition='auto',
        )])
        fig.update_layout(title='Probabilidades de Classificação',
                          xaxis_title='Classe',
                          yaxis_title='Probabilidade')
        st.plotly_chart(fig)
        
        result = labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        st.write(f"Predição: **{result}**")
        st.write(f"Confiança: **{confidence:.2f}%**")
        
# Página de Estatísticas
elif page == "Estatísticas":
    st.header("Estatísticas do Modelo")

    # Caminho base
    base_path = "Dataset/Testing"

    # Carregar os dados de teste
    X_test, y_test = load_test_data(base_path)

    # Carregar o modelo
    model = load_model()

    # Avaliar o modelo
    loss, accuracy, precision, recall, f1, conf_matrix, fpr, tpr, roc_auc = evaluate_model(model, X_test, y_test)

    # Exibir as métricas
    st.write(f"Acurácia: {accuracy:.2%}")
    st.write(f"Precisão: {precision:.2%}")
    st.write(f"Recall: {recall:.2%}")
    st.write(f"F1-Score: {f1:.2%}")

    # Exibir matriz de confusão
    st.subheader("Matriz de Confusão")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    st.pyplot(fig)

    # Exibir Curva ROC
    st.subheader("Curva ROC")
    fig, ax = plt.subplots()
    for i, label in enumerate(labels):
        ax.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC por Classe')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Página do Artigo
elif page == "Artigo":
    st.title("Artigo Científico")
    st.write("Leia o artigo completo sobre o projeto de detecção de tumor cerebral.")
    
    pdf_path = "Classificação de Tumores Cerebrais.pdf"
    
    def show_pdf(file_path):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    show_pdf(pdf_path)
    
    with open(pdf_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    
    st.download_button(label="Baixar Artigo em PDF", 
                       data=PDFbyte, 
                       file_name="artigo.pdf", 
                       mime='application/octet-stream')
