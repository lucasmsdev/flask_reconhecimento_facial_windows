# app.py - VERSÃO COM ÁUDIO DO S3

from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import face_recognition
import boto3
from botocore.exceptions import ClientError # ### NOVO ### Para tratar erros do S3

# --- CONFIGURAÇÕES ---
# ⚠️ Lembre-se de substituir com suas credenciais válidas e temporárias do AWS Learner Lab!
AWS_ACCESS_KEY_ID = 'ASIA6ODU7GZ6I3C5MZHW'
AWS_SECRET_ACCESS_KEY = 'tsDjxUzr8iGzXZM67Y59Iku+62Hiv0JBQRcDVBA9'
AWS_SESSION_TOKEN = 'IQoJb3JpZ2luX2VjED8aCXVzLXdlc3QtMiJHMEUCIQCFdoLoVyVc1+CMTQqh7ujnix/k71jJGbazQGyhWLAJ6QIgV7BU8YkTih2Qf6H/ejJM/y1KVix4isCDKjqieeVCFgIqrwIIeBAAGgw5OTIzODI3NjA1NzIiDBvXvAzs+j/WKsGzFiqMApgfvlK7P0AqXrLadRo3+5fY0fT8csSt4hc/CbQ34vfKDobbCqHYubrpWBMUWw4GXtIsaIe9Thwj7tyIco9sDSEDLN+1EqobH7w5ky29ZgNV/TSJB5Sfs0pglGisCwIp1cUPXBNghSOgm6dE2E55LxB+RxqNX2Upe56aUGOjIIPXQ+3S2Xb6pQOsehg0FpNb+f9CW2oQAfCwS2g//T/arfTY70p+AWYRtrb8CEXTjGFM1x9hZTQ+GAXFdcjI+A5w2VqOW1eo/5sm3IHMGyT6NGdoLrYmS2M9/Vp048YfzP3wIKjczaG77RZH18sy1dLWmHAFnBRR0aQt9qjgF0jAxbj4QulpX4n+S0TCZS0wrczNxAY6nQExjf9l5U5K09Q5YQ+vHEvBrw34x/aI+VSeaZT2C6GfABgfWMEMCXUYQTKeYK1QO1bTFkwYPeTOsHdgsXk4jn7Q8+ZazhKPr0Urfp6d+dh6AmMa4a8LdqhM17XrpT6HjhgoEhwhvWzlnlv38kL7yKKP7YQdQq7NIl5PkmgiTSv0C2TBXLGDQMMHrj33SR10zydIGR630QF/rzEAAyil'
AWS_REGION = 'us-east-1'
S3_BUCKET_NAME = 'visaocomputacional-senai'
MIN_FACE_AREA_THRESHOLD = 30000 

# --- VARIÁVEIS GLOBAIS E CLIENTE S3 ---
known_face_encodings = []
known_face_names = []
recognized_person_set = set()

# ### NOVO ### Instancia o cliente S3 fora das funções para reutilização
s3_client = boto3.client('s3', 
                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                         aws_session_token=AWS_SESSION_TOKEN,
                         region_name=AWS_REGION)

# --- FUNÇÕES DE LÓGICA ---

def load_known_faces():
    """Carrega as assinaturas faciais do S3 para a memória."""
    global known_face_encodings, known_face_names
    print("➡️  Carregando rostos conhecidos do S3...")
    try:
        main_folder_prefix = 'known_faces/'
        paginator = s3_client.get_paginator('list_objects_v2')
        person_folders = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=main_folder_prefix, Delimiter='/')
        
        for page in person_folders:
            for prefix in page.get('CommonPrefixes', []):
                person_folder_prefix = prefix.get('Prefix')
                person_name = person_folder_prefix.replace(main_folder_prefix, '').strip('/')
                person_encodings = []
                image_files = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=person_folder_prefix)
                for image_obj_summary in image_files.get('Contents', []):
                    s3_key = image_obj_summary['Key']
                    if s3_key.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                        image_content = image_obj['Body'].read()
                        image = face_recognition.load_image_file(BytesIO(image_content))
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            person_encodings.append(encodings[0])
                if person_encodings:
                    average_encoding = np.mean(person_encodings, axis=0)
                    known_face_encodings.append(average_encoding)
                    known_face_names.append(person_name)
        
        if not known_face_names:
            print("⚠️ AVISO: Nenhuma face encontrada no S3. O reconhecimento não funcionará.")
        else:
            print(f"✅ {len(known_face_names)} pessoas carregadas: {', '.join(known_face_names)}")
            
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao carregar faces do S3: {e}")
        print("   Verifique suas credenciais, nome do bucket e permissões.")

# ### VERSÃO 3: FUNÇÃO AINDA MAIS ROBUSTA CONTRA O TypeError ###
def process_frame(image_data_url):
    """Processa um frame para reconhecer o rosto mais próximo com alta precisão."""
    header, encoded = image_data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    
    try:
        pil_image = Image.open(BytesIO(image_data))
        # Converte a imagem para o formato que a biblioteca face_recognition espera (RGB)
        frame = np.array(pil_image.convert('RGB'))
    except Exception as e:
        print(f"Erro ao converter a imagem: {e}")
        return "Erro de Imagem"

    # 1. Detecta todos os rostos na imagem
    # Usando o modelo 'cnn' pode ser mais lento, mas é mais preciso. Se ficar lento, volte para "hog".
    face_locations = face_recognition.face_locations(frame, model="hog")
    if not face_locations:
        return "Desconhecido"

    # 2. Gera as assinaturas faciais para todos os rostos encontrados.
    # Esta é a chamada que está causando o erro, vamos mantê-la simples.
    # Se esta linha continuar a falhar, o problema está no ambiente/dependências.
    try:
        face_encodings = face_recognition.face_encodings(frame, face_locations)
    except TypeError as e:
        # Se o erro acontecer aqui, saberemos que é um problema fundamental de chamada.
        print(f"!!! ERRO CRÍTICO no face_encodings: {e}")
        print("!!! O problema provavelmente está nas versões das bibliotecas dlib/face_recognition.")
        return "Erro Interno"
        
    if not face_encodings:
        # Não foi possível gerar uma assinatura para os rostos encontrados
        return "Desconhecido"

    # 3. Calcula a área de cada rosto para encontrar o maior (mais próximo)
    face_areas = [(bottom - top) * (right - left) for top, right, bottom, left in face_locations]
    largest_face_index = np.argmax(face_areas)
    max_area = face_areas[largest_face_index]
    
    # 4. Verifica se o rosto está perto o suficiente
    if max_area < MIN_FACE_AREA_THRESHOLD:
        return "Aproxime-se"

    # 5. Seleciona a assinatura facial que corresponde ao maior rosto
    face_encoding_to_check = face_encodings[largest_face_index]
    
    name = "Desconhecido"
    
    # 6. Compara a assinatura do maior rosto com as conhecidas
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.4)
    
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
    
    return name

# ### NOVO ### Função para gerar URL pré-assinada para o áudio no S3
def generate_presigned_audio_url(person_name):
    """Gera uma URL segura e temporária para o arquivo audio.mp3 de uma pessoa."""
    object_key = f"known_faces/{person_name}/audio.mp3"
    try:
        url = s3_client.generate_presigned_url('get_object',
                                               Params={'Bucket': S3_BUCKET_NAME, 'Key': object_key},
                                               ExpiresIn=300) # URL válida por 5 minutos
        print(f"🔑 URL de áudio gerada para {person_name}")
        return url
    except ClientError as e:
        # Ocorre se o arquivo 'audio.mp3' não for encontrado para essa pessoa
        print(f"⚠️ AVISO: Não foi possível gerar URL para '{object_key}'. O arquivo existe? Erro: {e}")
        return None

# --- APLICAÇÃO WEB (FLASK) ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'um-segredo-muito-secreto!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list')
def list_page():
    current_names = sorted(list(recognized_person_set))
    return render_template('list.html', names=current_names)

@socketio.on('image')
def handle_image(image_data_url):
    recognized_name = process_frame(image_data_url)
    # Envia o nome para ser exibido na tela em tempo real
    socketio.emit('response', {'name': recognized_name})
    
    # ### MODIFICADO ### Lógica de reconhecimento e áudio
    # Verifica se a pessoa é conhecida e se é a PRIMEIRA vez que é reconhecida nesta sessão
    if recognized_name not in ["Desconhecido", "Aproxime-se"] and recognized_name not in recognized_person_set:
        print(f"✔️ Nova pessoa adicionada à lista: {recognized_name}")
        
        # 1. Adiciona à lista de presença
        recognized_person_set.add(recognized_name)
        sorted_list = sorted(list(recognized_person_set))
        socketio.emit('update_list', {'names': sorted_list})

        # 2. Tenta gerar e enviar a URL do áudio para o cliente
        audio_url = generate_presigned_audio_url(recognized_name)
        if audio_url:
            # Emite um evento específico para tocar o áudio no frontend
            socketio.emit('play_audio', {'url': audio_url})

# ### REMOVIDO ### A rota /get-speech não é mais necessária.
# @app.route('/get-speech', methods=['POST'])
# def get_speech():
#     ...

if __name__ == '__main__':
    load_known_faces()
    print("🚀 Servidor pronto!")
    print("   - Página da Webcam: http://127.0.0.1:5000")
    print("   - Página da Lista:  http://127.0.0.1:5000/list")
    socketio.run(app, host='0.0.0.0', port=5000)

