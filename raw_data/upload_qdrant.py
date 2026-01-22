import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Configurações do Qdrant
# Se estiver rodando localmente (Docker), use "localhost" e porta 6333
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "questoes_8_ano_12_12_2025"

# Inicializar o cliente Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Inicializar modelo de Embeddings (multilingue recomendado para PT-BR)
# 'paraphrase-multilingual-MiniLM-L12-v2' é excelente para português
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
    # 1. Carregar dados
    questoes = load_data('questoes_8_ano_formatado.json')
    
    # 2. Gerar embeddings e preparar pontos
    points = []
    print(f"Processando {len(questoes)} questões...")
    
    # Usamos enumerate para garantir IDs de 1 a 49 sequenciais gerados pelo script
    for i, questao in enumerate(questoes, start=1):
        # Criar o texto que será vetorizado (Texto Base + Enunciado)
        texto_para_vetorizar = f"{questao['texto_base']} {questao['enunciado']}"
        
        # Gerar vetor
        embedding = model.encode(texto_para_vetorizar).tolist()
        
        # Criar estrutura do ponto para o Qdrant
        # O gabarito NÃO está incluído no payload
        point = PointStruct(
            id=i,  # ID sequencial do Qdrant (1 a 49)
            vector=embedding,
            payload={
                "id": i,  # ID salvo também dentro dos dados para fácil visualização
                "texto_base": questao['texto_base'],
                "enunciado": questao['enunciado'],
                "alternativas": questao['alternativas']
            }
        )
        points.append(point)

    # 3. Criar coleção (a dimensão depende do modelo, geralmente 384 para MiniLM)
    embedding_dim = len(points[0].vector)
    create_collection_if_not_exists(embedding_dim)

    # 4. Fazer upload (Upsert)
    print("Enviando dados para o Qdrant...")
    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points
    )
    
    print(f"Upload concluído! Status: {operation_info.status}")

if __name__ == "__main__":
    # Certifique-se de que o arquivo 'questoes_saeb_2009.json' esteja na mesma pasta
    try:
        process_and_upload()
    except FileNotFoundError:
        print("Erro: Arquivo 'questoes_saeb_2009.json' não encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")