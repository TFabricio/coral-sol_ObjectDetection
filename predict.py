import streamlit as st
from ultralytics import YOLO
import os
import cv2
import yaml
import tempfile
import numpy as np
import yt_dlp as youtube_dl 
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model_path = "coral_sol.pt"
data_yaml_path = "data.yaml"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} não encontrado!")

if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"{data_yaml_path} não encontrado!")

model = YOLO(model_path)

with open(data_yaml_path, "r") as yaml_file:
    data_config = yaml.safe_load(yaml_file)

st.title("Object Detection Dashboard")
st.markdown("object prediction using YOLOv11")

st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        .sidebar .sidebar-content {background-color:rgb(12, 12, 12); color:rgb(250, 250, 250);}
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Ajuda e Guias")

st.sidebar.markdown("""
    ### Guia Rápido de Uso:
    1. **Escolha o modo de detecção**: Selecione se você deseja usar imagens, vídeos, YouTube ou câmera.
    2. **Ajuste os parâmetros de detecção**:
        - **Confidence Threshold**: Ajuste o nível de confiança mínimo para considerar uma detecção válida.
        - **Detection Line Width**: Controle a espessura das linhas de detecção.
        - **Mostrar Rótulos**: Ative para exibir os nomes dos objetos detectados.
    3. **Clique no botão "Iniciar Detecção"** para processar as imagens ou vídeos carregados.

    ### O que é cada parâmetro:
    - **Confidence Threshold**: Controla o nível mínimo de confiança que a detecção precisa ter para ser exibida. Valores mais altos podem eliminar detecções erradas, mas também podem perder objetos reais.
    - **Line Width**: Controla a espessura das caixas de detecção. Útil para visibilidade.
    - **Mostrar Rótulos**: Ative para exibir os nomes dos objetos detectados diretamente nas imagens.

    ### Documentação Oficial do YOLO:
    Caso você esteja utilizando o YOLO para detecção, você pode se beneficiar das documentações oficiais de versões amplamente utilizadas do YOLO:

    - **[Documentação YOLOv11](https://docs.ultralytics.com/pt/models/yolo11/)**: Guia completo sobre como usar, treinar e ajustar o YOLOv11, incluindo tutoriais, dicas e melhores práticas. (Substitua o link com o seu)
    - **[Repositório YOLOv11 no GitHub](https://github.com/ultralytics/ultralytics)**: Código-fonte, issues e contribuições para o YOLOv11. Este repositório inclui scripts para treinar e testar modelos de detecção. (Substitua o link com o seu)
    
    **Nota**: YOLO (You Only Look Once) é uma arquitetura de rede neural profunda especializada na detecção em tempo real de múltiplos objetos em imagens e vídeos. A versão YOLOv11 pode ter características próprias e a documentação pode ser interna ou personalizada.

    Caso tenha dificuldades ao usar a dashboard ou qualquer outra parte do sistema, consulte as seções de FAQ no repositório do seu modelo YOLOv11 ou entre em contato com a comunidade de desenvolvedores.
""")


def display_help_section():
    st.sidebar.markdown("""
        ## Como Usar:
        1. Carregue suas **imagens** ou **vídeos**.
        2. Ajuste os **parâmetros de detecção** como a confiança e a largura da linha de detecção.
        3. Selecione o modo de detecção desejado:
            - **Imagem**: Carregar imagens para detectar objetos nelas.
            - **Vídeo**: Carregar vídeos para detectar objetos em cada quadro.
            - **YouTube**: Fornecer um link de vídeo do YouTube para detecção.
            - **Câmera**: Usar a câmera ao vivo para detectar objetos em tempo real.
        4. Clique em **Iniciar Detecção**.
    """)
    
display_help_section()

st.sidebar.title("Controle de Detecção")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
line_width = st.sidebar.slider("Detection Line Width", 1, 10, 2)
show_labels = st.sidebar.checkbox("Show Labels", value=True)
save_txt = st.sidebar.checkbox("Save Results as Text", value=True)
save_crop = st.sidebar.checkbox("Save Cropped Images", value=False)
top_n = st.sidebar.slider("Melhores Detecções", 1, 500, 5, 1)
confidence_filter = st.sidebar.slider('Filtrar por Confiança Mínima', 0.0, 1.0, 0.4)
detection_mode = st.sidebar.radio("Escolha o modo de detecção", ["Imagem", "Vídeo", "YouTube", "Câmera"])

def run_prediction(frame, conf, line_width, show_labels, classes):
    results = model.predict(frame, conf=conf, line_width=line_width, show_labels=show_labels, classes=classes)
    return results

def display_detections_details(detections):
    if detections:
        data = []
        for idx, (confidence, detection, filename, _) in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            data.append({
                'Arquivo': filename,
                'Detecção': f'{idx+1}',
                'Confiança': f'{confidence:.2f}',
                'Caixa': f'({x1}, {y1}) - ({x2}, {y2})'
            })
        df = pd.DataFrame(data)
        st.dataframe(df)

def plot_confidence_distribution(confidences):
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribuição de Confiança das Detecções')
    plt.xlabel('Confiança')
    plt.ylabel('Número de Detecções')
    st.pyplot(plt)

def display_detection_alerts(detections, labels_of_interest=["person", "car"]):
    for detection in detections:
        label = detection[1].cls[0]  
        if label in labels_of_interest:
            object_name = data_config['names'][label]
            st.success(f"Detecção de {object_name}!")
            break 

def process_images(uploaded_files):
    all_detections = []
    all_confidences = []  
    seen_files = set()  
    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            st.warning(f"Erro ao carregar a imagem {uploaded_file.name}.")
            continue

        image_resized = cv2.resize(image, (640, 640))  
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        results = run_prediction(image_rgb, conf=conf_threshold, line_width=line_width, show_labels=show_labels, classes=[0])

        detections = []
        for detection in results[0].boxes:
            confidence = detection.conf.item()
            if confidence >= confidence_filter:
                detections.append((confidence, detection, uploaded_file.name, image))  
                all_confidences.append(confidence)  

        display_detection_alerts(detections)

        if detections:
            best_detection = max(detections, key=lambda x: x[0])

            if best_detection[2] not in seen_files:  
                all_detections.append(best_detection)
                seen_files.add(best_detection[2])  

    all_detections.sort(key=lambda x: x[0], reverse=True)
    
    top_detections = all_detections[:top_n]

    st.write(f"Top {top_n} Detecções em todas as imagens:")
    display_detections_details(top_detections)
    plot_confidence_distribution(all_confidences)

    for i, (confidence, detection, filename, img) in enumerate(top_detections):
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        st.write(f"{i+1}. Arquivo: {filename} | Confiança: {confidence:.2f} | Caixa: ({x1}, {y1}) - ({x2}, {y2})")
        
        img_with_detections = img.copy()

        cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_with_detections, f"Conf: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        img_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"Detecção {i+1} em {filename}", use_container_width=True)

        if save_crop:
            crop_img = img[y1:y2, x1:x2]
            crop_output_path = f"crop_{filename}"
            cv2.imwrite(crop_output_path, crop_img)
            st.write(f"Cropped image saved to {crop_output_path}")

    if all_confidences:
        avg_confidence = np.mean(all_confidences)
        st.write(f"Média de Confiança Geral do Modelo: {avg_confidence:.2f}")
    else:
        st.write("Nenhuma detecção válida encontrada.")

def process_and_show_video_with_detections(uploaded_video):
    video_name = os.path.basename(uploaded_video) if isinstance(uploaded_video, str) else uploaded_video.name
    st.write(f"Processando o vídeo: {video_name}")

    if isinstance(uploaded_video, str):
        video_path = uploaded_video
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

    if not os.path.exists(video_path):
        st.error("O arquivo de vídeo não foi encontrado no caminho temporário.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Falha ao abrir o arquivo de vídeo: {video_path}. O OpenCV não consegue capturar o vídeo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_confidences = []
    detection_counts = []

    frame_slider = st.slider('Selecione o Frame:', min_value=1, max_value=total_frames, value=1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider - 1)
    ret, frame = cap.read()

    if ret:
        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        results = run_prediction(frame_rgb, conf=0.4, line_width=2, show_labels=True, classes=[0])

        detections = []
        for detection in results[0].boxes:
            confidence = detection.conf.item()
            if confidence >= 0.5:  
                detections.append((confidence, detection))

        detection_counts.append(len(detections))
        all_confidences.extend([conf for conf, _ in detections])

        img_with_detections = frame.copy()
        for confidence, detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_with_detections, f"Conf: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, channels="RGB", use_container_width=True)

    cap.release()

    if all_confidences:
        detections_df = pd.DataFrame({
            "Frame": range(1, len(detection_counts) + 1),
            "Detecções": detection_counts,
            "Confiança Média": [np.mean(all_confidences[i:i+len(detection_counts[i])]) for i in range(len(detection_counts))]
        })

        st.write("Resumo das Detecções por Frame")
        st.dataframe(detections_df)

        st.write("Distribuição das Confianças")
        plt.hist(all_confidences, bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribuição de Confiança das Detecções')
        plt.xlabel('Confiança')
        plt.ylabel('Número de Detecções')
        st.pyplot(plt)

        st.write("Contagem de Detecções por Frame")
        plt.plot(range(1, len(detection_counts) + 1), detection_counts, marker='o', color='orange')
        plt.title('Contagem de Detecções por Frame')
        plt.xlabel('Frame')
        plt.ylabel('Número de Detecções')
        st.pyplot(plt)
    else:
        st.write("Nenhuma detecção válida encontrada no vídeo.")

def process_video_in_real_time(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)

    all_confidences = []
    detection_counts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        results = run_prediction(frame_rgb, conf=conf_threshold, line_width=2, show_labels=True, classes=[0])

        detections = []
        for detection in results[0].boxes:
            confidence = detection.conf.item()
            if confidence >= confidence_filter:
                detections.append((confidence, detection))

        detection_counts.append(len(detections))
        all_confidences.extend([conf for conf, _ in detections])

        img_with_detections = frame.copy()
        for confidence, detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_with_detections, f"Conf: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, channels="RGB", use_container_width=True)

        st.progress(len(all_confidences) / 100)

        if len(all_confidences) % 10 == 0:
            st.write("Distribuição de Confiança em Tempo Real")
            plt.hist(all_confidences, bins=20, color='skyblue', edgecolor='black')
            plt.title('Distribuição de Confiança em Tempo Real')
            st.pyplot(plt)

            st.write("Contagem de Detecções em Tempo Real")
            plt.plot(range(1, len(detection_counts) + 1), detection_counts, marker='o', color='orange')
            plt.title('Contagem de Detecções em Tempo Real')
            st.pyplot(plt)

    cap.release()

if detection_mode == "Imagem":
    uploaded_images = st.sidebar.file_uploader("Escolha imagens para detecção", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_images:
        if st.sidebar.button("Iniciar Detecção"):
            process_images(uploaded_images)

def download_youtube_video(url):
    ydl_opts = {
        'format': 'best',  
        'outtmpl': tempfile.mktemp(suffix='.mp4')  
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_file = ydl.prepare_filename(info_dict) 
        return video_file

if detection_mode == "Vídeo":
    uploaded_videos = st.sidebar.file_uploader("Escolha vídeos para detecção", type=["mp4", "avi", "mov"], accept_multiple_files=True)
    if uploaded_videos:
        if st.sidebar.button("Iniciar Detecção"):
            for uploaded_video in uploaded_videos:
                video_file = uploaded_video
                process_and_show_video_with_detections(video_file)

if detection_mode == "YouTube":
    youtube_url = st.sidebar.text_input("URL do vídeo do YouTube")
    if youtube_url:
        if st.sidebar.button("Iniciar Detecção"):
            video_file = download_youtube_video(youtube_url)
            if video_file:
                process_and_show_video_with_detections(video_file)

if detection_mode == "Câmera":
    if st.sidebar.button("Iniciar Detecção em Tempo Real"):
        process_video_in_real_time(video_path=None)