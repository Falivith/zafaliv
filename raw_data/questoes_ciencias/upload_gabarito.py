import os
import json
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any

# Bibliotecas de terceiros
import dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer # Nova dependência

# Carrega variáveis de ambiente (.env)
dotenv.load_dotenv()

# --- CONFIGURAÇÕES GERAIS ---
# Modelo de Embedding (Local / HuggingFace)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ATENÇÃO: A dimensão desse modelo específico é 384, não 3072.
VECTOR_SIZE = 384

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Nome da coleção específica para o Gabarito (versão local)
COLLECTION_NAME = "saeb_gabarito_ciencias_local"
JSON_FILE_PATH = "gabarito.json" 

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

    def get_embedding(self, text: str) -> List[float]:
        """Gera embedding para um único texto."""
        # Limpeza básica
        text = text.replace("\n", " ")
        try:
            # Gera o embedding (convertendo para lista Python)
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            return embedding
        except Exception as e:
            print(f"❌ Erro na geração de Embedding Local: {e}")
            return []

class GabaritoPipeline:
    def __init__(self):
        self.embedder = EmbeddingClient()
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
                print(f"🛠️ Criando coleção '{COLLECTION_NAME}' (Size: {VECTOR_SIZE})...")
                self.qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                )
            else:
                # Verificação de segurança de dimensão
                info = self.qdrant.get_collection(COLLECTION_NAME)
                if info.config.params.vectors.size != VECTOR_SIZE:
                    print(f"🚨 AVISO CRÍTICO: A coleção '{COLLECTION_NAME}' tem tamanho {info.config.params.vectors.size}, mas o modelo usa {VECTOR_SIZE}.")
                    print("➡️ Sugestão: Delete a coleção antiga ou mude o nome da COLLECTION_NAME.")
                    raise ValueError("Conflito de dimensão de vetores.")
                
                print(f"✅ Coleção '{COLLECTION_NAME}' já existe e está compatível.")
        except Exception as e:
            print(f"❌ Falha crítica ao conectar no Qdrant: {e}")
            raise e

    def process_gabarito(self, file_path: str):
        print(f"\n📂 Iniciando processamento do arquivo: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Erro ao abrir JSON: {e}")
            return

        print(f"📊 Encontrados {len(data)} itens de gabarito.")
        
        points_to_upsert = []
        start_time = time.time()

        for item in data:
            # 1. Extração dos dados brutos
            q_id = item.get("id")
            resposta = item.get("resposta_correta")
            comentario = item.get("comentario_resposta", "Sem comentário disponível.") # Fallback se vazio

            # Validação básica
            if q_id is None or resposta is None:
                print(f"⚠️ Item ignorado por falta de ID ou Resposta: {item}")
                continue

            # 2. Criação do texto rico para busca semântica
            conteudo_vetorial = f"Gabarito Questão {q_id}. Resposta: {resposta}. Explicação: {comentario}"
            
            print(f"  ⚡ Vetorizando Gabarito ID {q_id}...", end="\r")
            
            # Chamada local
            vector = self.embedder.get_embedding(conteudo_vetorial)
            
            if not vector:
                continue

            # 3. Payload Estruturado
            payload = {
                "id": q_id,
                "resposta_correta": resposta,
                "tipo": "gabarito",
                "comentario_resposta": comentario
            }

            # 4. ID do Ponto no Qdrant (UUID)
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"saeb_gab_{q_id}"))

            points_to_upsert.append(PointStruct(
                id=point_uuid,
                vector=vector,
                payload=payload
            ))

        # 5. Envio em Batch (Lote)
        if points_to_upsert:
            try:
                print(f"\n\n🚀 Enviando {len(points_to_upsert)} vetores para '{COLLECTION_NAME}'...")
                self.qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_to_upsert
                )
                elapsed = time.time() - start_time
                print(f"✨ Sucesso! Processamento concluído em {elapsed:.2f}s.")
                
                # Verificação pós-upload
                print("\n🔍 Testando recuperação do primeiro item inserido...")
                test_point = self.qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[points_to_upsert[0].id],
                    with_payload=True
                )
                if test_point:
                    print(f"   Payload recuperado: {test_point[0].payload}")
                
            except Exception as e:
                print(f"❌ Erro durante o upload: {e}")
        else:
            print("⚠️ Nenhum dado válido gerado para upload.")

if __name__ == "__main__":
    if os.path.exists(JSON_FILE_PATH):
        pipeline = GabaritoPipeline()
        pipeline.process_gabarito(JSON_FILE_PATH)
    else:
        print(f"❌ Arquivo '{JSON_FILE_PATH}' não encontrado.")
        print("Certifique-se de que o arquivo 'gabarito.json' está na mesma pasta.")