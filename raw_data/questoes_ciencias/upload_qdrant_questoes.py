import os
import json
import uuid
import time
from typing import List, Dict, Any

# Bibliotecas de terceiros
import dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer  # Nova dependência

# Carrega variáveis de ambiente (.env) - Útil para configurações do Qdrant se houver
dotenv.load_dotenv()

# --- CONFIGURAÇÕES GERAIS ---
# Modelo de Embedding (Local / HuggingFace)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ATENÇÃO: A dimensão desse modelo específico é 384, não 3072.
VECTOR_SIZE = 384  
BATCH_SIZE = 50     # Quantidade de itens para vetorizar por vez

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "saeb_questoes_ciencias_local" # Mudei o nome para evitar conflito com a coleção antiga de tamanho diferente
JSON_FILE_PATH = "questoes_saeb_ciencias.json" 

class EmbeddingClient:
    """Wrapper para geração de embeddings locais usando SentenceTransformers."""
    
    def __init__(self):
        print(f"📥 Carregando modelo local: {EMBEDDING_MODEL_NAME}...")
        try:
            # Carrega o modelo na memória (CPU ou GPU automaticamente)
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("✅ Modelo carregado com sucesso!")
        except Exception as e:
            raise RuntimeError(f"🚨 Erro ao carregar o modelo SentenceTransformer: {e}")

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para uma lista de textos de uma só vez localmente."""
        # Limpeza básica (opcional, pois o SentenceTransformers lida bem com quebras)
        clean_texts = [text.replace("\n", " ") for text in texts]
        
        try:
            # O encode já retorna uma lista de numpy arrays ou tensores
            embeddings = self.model.encode(clean_texts, convert_to_numpy=True)
            
            # Converte para lista Python pura para serialização JSON/Qdrant
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ Erro na geração de Embedding Local: {e}")
            return []

class JsonPipeline:
    def __init__(self):
        self.embedder = EmbeddingClient()
        
        # Cliente Qdrant
        self.qdrant = QdrantClient(
            host=QDRANT_HOST, 
            port=QDRANT_PORT,
            timeout=120,
        )
        
        self._init_qdrant_collection()

    def _init_qdrant_collection(self):
        """Garante que a coleção exista com a dimensão correta (384)."""
        try:
            collections_response = self.qdrant.get_collections()
            existing_names = [c.name for c in collections_response.collections]
            
            if COLLECTION_NAME not in existing_names:
                print(f"🛠️ Criando coleção '{COLLECTION_NAME}' (Size: {VECTOR_SIZE}, Distance: Cosine)...")
                self.qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                )
            else:
                # Verificação de segurança de dimensão (Opcional, mas recomendado)
                info = self.qdrant.get_collection(COLLECTION_NAME)
                if info.config.params.vectors.size != VECTOR_SIZE:
                    print(f"🚨 AVISO CRÍTICO: A coleção '{COLLECTION_NAME}' existe mas tem tamanho {info.config.params.vectors.size}, e o modelo usa {VECTOR_SIZE}.")
                    print("➡️ Sugestão: Delete a coleção antiga ou mude o nome da COLLECTION_NAME.")
                    raise ValueError("Conflito de dimensão de vetores.")
                
                print(f"✅ Coleção '{COLLECTION_NAME}' já existe e está pronta.")
        except Exception as e:
            print(f"❌ Falha ao conectar ou configurar o Qdrant: {e}")
            raise e

    def _format_alternatives_text(self, alternativas: Any) -> str:
        """Converte o dicionário de alternativas em texto corrido APENAS para vetorização."""
        if not alternativas:
            return "Questão discursiva/aberta."
        if isinstance(alternativas, dict):
            return " ".join([f"{k}) {v}" for k, v in alternativas.items()])
        return str(alternativas)

    def process_dataset(self, file_path: str):
        """Lê o JSON, gera vetores em lote e faz upload."""
        print(f"\n📂 Iniciando processamento do arquivo: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Erro ao abrir JSON: {e}")
            return

        total_items = len(data)
        print(f"📊 Encontradas {total_items} questões para indexar.")
        
        points_to_upsert = []
        batch_texts = []
        batch_payloads = []
        batch_ids = []
        
        start_time = time.time()

        # Iterar sobre os dados e criar lotes
        for i, item in enumerate(data):
            # 1. Preparar Texto Vetorial e Payload
            q_id = item.get("id")
            texto_base = item.get("texto_base") or ""
            enunciado = item.get("enunciado") or ""
            alternativas = item.get("alternativas")
            
            alternativas_txt = self._format_alternatives_text(alternativas)
            conteudo_vetorial = f"Contexto: {texto_base}\nPergunta: {enunciado}\nOpções: {alternativas_txt}"
            
            # Payload estruturado para RAG
            payload = {
                "id": q_id,
                "texto_base": texto_base,
                "enunciado": enunciado,
                "alternativas": alternativas
            }
            
            # ID determinístico
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"saeb_q_{q_id}"))

            # Adiciona às listas temporárias do lote
            batch_texts.append(conteudo_vetorial)
            batch_payloads.append(payload)
            batch_ids.append(point_uuid)

            # Se o lote encheu ou é o último item, processa
            if len(batch_texts) >= BATCH_SIZE or i == total_items - 1:
                print(f"  ⚡ Processando lote {i - len(batch_texts) + 2} a {i + 1} de {total_items}...", end="\r")
                
                # Chamada Local (Sentence Transformers)
                vectors = self.embedder.get_embeddings_batch(batch_texts)
                
                if len(vectors) != len(batch_texts):
                    print(f"\n  ⚠️ Erro de tamanho no lote. Esperado: {len(batch_texts)}, Recebido: {len(vectors)}")
                    batch_texts, batch_payloads, batch_ids = [], [], []
                    continue

                # Cria os PointStructs
                for pid, vector, pay in zip(batch_ids, vectors, batch_payloads):
                    points_to_upsert.append(PointStruct(
                        id=pid,
                        vector=vector,
                        payload=pay
                    ))
                
                # Limpa listas temporárias
                batch_texts, batch_payloads, batch_ids = [], [], []

        # 5. Upload Final para Qdrant
        if points_to_upsert:
            try:
                print(f"\n🚀 Enviando {len(points_to_upsert)} vetores para o Qdrant...")
                
                # Qdrant lida bem com upserts grandes, mas se tiver MILHARES, 
                # pode ser ideal dividir o upsert em chunks também.
                # Aqui faremos direto conforme script original.
                self.qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_to_upsert
                )
                elapsed = time.time() - start_time
                print(f"✨ Sucesso! {len(points_to_upsert)} itens processados em {elapsed:.2f}s.")
            except Exception as e:
                print(f"❌ Erro durante o upload para o Qdrant: {e}")
        else:
            print("⚠️ Nenhum dado válido foi gerado para upload.")

# --- PONTO DE ENTRADA ---
if __name__ == "__main__":
    if os.path.exists(JSON_FILE_PATH):
        pipeline = JsonPipeline()
        pipeline.process_dataset(JSON_FILE_PATH)
    else:
        print(f"❌ Arquivo '{JSON_FILE_PATH}' não encontrado.")
        print("Certifique-se de que o arquivo JSON das questões está na mesma pasta.")