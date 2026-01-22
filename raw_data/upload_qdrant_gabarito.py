import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Configurações do Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# Nome diferente para não misturar com as perguntas
COLLECTION_NAME = "gabarito_8_ano_12_12_2025" 

# Inicializar o cliente Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Usamos o mesmo modelo para gerar vetores compatíveis, 
# embora para o gabarito o vetor seja menos importante que o ID.
print("Carregando modelo de embeddings...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_collection_if_not_exists(dim):
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    if COLLECTION_NAME not in collection_names:
        print(f"Criando coleção '{COLLECTION_NAME}' com dimensão {dim}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        print(f"Coleção '{COLLECTION_NAME}' já existe.")

def process_and_upload():
    # 1. Carregar dados do gabarito
    try:
        gabarito = load_data('gabarito_8_ano_formatado.json')
    except FileNotFoundError:
        print("Erro: Arquivo 'gabarito_8_ano_formatado.json' não encontrado.")
        return

    # 2. Preparar pontos
    points = []
    print(f"Processando {len(gabarito)} respostas...")
    
    for item in gabarito:
        # ID do item (deve corresponder ao ID da questão)
        q_id = item['id']
        resposta = item['resposta_correta']
        
        # Criar um texto descritivo para o vetor (opcional, mas necessário para o Qdrant)
        texto_vetorizavel = f"Gabarito da questão {q_id}: Alternativa {resposta}"
        
        # Gerar vetor
        embedding = model.encode(texto_vetorizavel).tolist()
        
        # Criar estrutura do ponto
        point = PointStruct(
            id=q_id,  # Importante: Mesmo ID da questão para facilitar o cruzamento
            vector=embedding,
            payload={
                "id": q_id,
                "resposta_correta": resposta,
                "tipo": "gabarito"
            }
        )
        points.append(point)

    # 3. Criar coleção se necessário
    if points:
        embedding_dim = len(points[0].vector)
        create_collection_if_not_exists(embedding_dim)

        # 4. Fazer upload
        print(f"Enviando {len(points)} gabaritos para o Qdrant...")
        operation_info = client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        
        print(f"Upload do gabarito concluído! Status: {operation_info.status}")
    else:
        print("Nenhum dado para enviar.")

if __name__ == "__main__":
    process_and_upload()