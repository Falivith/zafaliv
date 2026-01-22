import os
import re
import json
import uuid
import time
import base64
import hashlib
import unicodedata
import difflib
import importlib.metadata
from io import BytesIO
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional

import pdfplumber
import dotenv
from PIL import Image
import requests

# =========================
# 📦 IMPORTAÇÃO SEGURA
# =========================
try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
    
    # FIX: Maneira segura de pegar a versão
    try:
        qdrant_version = importlib.metadata.version("qdrant-client")
    except Exception:
        qdrant_version = "Desconhecida"
        
    print(f"📦 Versão Qdrant Client: {qdrant_version}")

except ImportError:
    print("❌ Erro: Biblioteca 'qdrant-client' não instalada.")
    exit(1)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# ⚙️ CONFIGURAÇÕES OTIMIZADAS
# =========================
dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# MUDANÇA 1: Modelo melhor para Português (Multilingue L12 vs L6)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
# Se usar 'multilingual-e5-small', lembre de adicionar prefixos "query:" e "passage:"
# Mas o paraphrase-multilingual funciona bem direto.

VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))
ENRICH_FOR_EMBEDDING = True 
TEXT_MAX_CHARS_FOR_EMBEDDING = 1000 

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_ciencias_v3") # Nova versão da coleção

# MUDANÇA 2: Chunks maiores para pegar contexto completo da explicação
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800")) 
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200")) 

METADATA_FILE = os.getenv("METADATA_FILE", "busca_semantica.json")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs_proedu_ciencias")


# =========================
# 🧪 HIGIENE DE TEXTO AVANÇADA
# =========================
class TextProcessor:
    _CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text: return ""
        
        # 1. Normalização Unicode
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Remover URLs (Causa do seu erro da Fotossíntese)
        text = TextProcessor._URL_PATTERN.sub('', text)
        
        # 3. Corrigir hifenização de quebra de linha (ex: "fo-\ntossíntese" -> "fotossíntese")
        # Isso é crucial para PDFs acadêmicos
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # 4. Remover caracteres de controle
        text = TextProcessor._CTRL_CHARS.sub("", text)
        
        # 5. Fixes científicos
        replacements = [
            (r"(\d+)\s*°\s*C", r"\1°C"),
            (r"CO\s*2", "CO2"),
            (r"H\s*2\s*O", "H2O"),
            (r"\s+", " "), # Remove múltiplos espaços
        ]
        for p, r in replacements:
            text = re.sub(p, r, text, flags=re.IGNORECASE)
            
        # 6. Remover linhas de "Fonte:" ou "Figura:" soltas que não explicam nada
        lines = []
        for line in text.split('\n'):
            l = line.strip()
            # Se a linha começa com Fonte/Figura e tem menos de 60 chars, provavelmente é lixo
            if (l.lower().startswith('fonte:') or l.lower().startswith('figura')) and len(l) < 60:
                continue
            lines.append(l)
            
        text = "\n".join(lines)
        return text.strip()

    @staticmethod
    def normalize_key(s: str) -> str:
        if not s: return ""
        s = unicodedata.normalize("NFKC", s).lower().strip()
        s = re.sub(r"\.pdf$", "", s)
        s = re.sub(r"[^a-z0-9]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s


# =========================
# 👁️ VISÃO (GEMINI)
# =========================
class GeminiVisionClient:
    def __init__(self):
        self.enabled = False
        if not GEMINI_API_KEY:
            print("ℹ️  GEMINI_API_KEY não configurada. Modo texto-apenas.")
            return
        self.api_key = GEMINI_API_KEY
        self.model_name = "gemini-2.5-flash-preview"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.enabled = True

    def analyze_page_image(self, pil_image: Image.Image) -> str:
        if not self.enabled: return ""
        try:
            pil_image.thumbnail((1024, 1024)) 
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            prompt = "Transcreva texto, tabelas e descreva gráficos científicos desta imagem. Ignore cabeçalhos e rodapés."
            payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}]}]}
            
            resp = requests.post(f"{self.base_url}?key={self.api_key}", json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        except Exception:
            pass
        return ""


# =========================
# 🧠 EMBEDDINGS & RERANKER
# =========================
class EmbeddingClient:
    def __init__(self):
        print(f"📥 Carregando Embedder: {EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()
        
        # MUDANÇA 3: Cross-Encoder para Reranking
        # Cross-encoders são mais lentos mas muito mais precisos. Eles leem a par (Query, Documento)
        print("📥 Carregando Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2...")
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except:
            print("⚠️ Falha ao carregar Reranker, usando fallback.")
            self.reranker = None

    def prepare_text(self, text: str, meta: Dict[str, Any], is_query: bool = False) -> str:
        text = TextProcessor.clean_text(text)
        if is_query: return text
        
        # Melhor formatação do cabeçalho
        header = f"Assunto: {meta.get('discipline','Ciências')} - {meta.get('title','')}"
        content = f"{header}\nConteúdo: {text}" if ENRICH_FOR_EMBEDDING else text
        return content[:TEXT_MAX_CHARS_FOR_EMBEDDING]

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        return [v.tolist() for v in self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)]

    def rerank_results(self, query: str, results: List[Any], top_k: int = 5) -> List[Any]:
        """Reordena os resultados do Qdrant usando um modelo mais preciso."""
        if not self.reranker or not results:
            return results[:top_k]
            
        # Prepara pares [Query, Documento]
        pairs = [[query, r.payload.get('text_content', '')] for r in results]
        
        # Calcula scores
        scores = self.reranker.predict(pairs)
        
        # Atribui novos scores e ordena
        for r, s in zip(results, scores):
            r.score = float(s) # Substitui score de cosseno pelo score do CrossEncoder
            
        # Ordena descrescente
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# =========================
# 📂 METADADOS
# =========================
class MetadataManager:
    def __init__(self, json_path: str):
        self.lookup = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else data.get("data", [])
                    for item in items:
                        if item.get("filename"): self.lookup[TextProcessor.normalize_key(item["filename"])] = item
            except Exception as e:
                print(f"⚠️ Erro ao ler metadados: {e}")

    def get(self, filename: str) -> Dict[str, Any]:
        key = TextProcessor.normalize_key(filename)
        meta = self.lookup.get(key)
        if not meta:
            matches = difflib.get_close_matches(key, self.lookup.keys(), n=1, cutoff=0.9)
            if matches: meta = self.lookup[matches[0]]
            
        return {
            "title": meta.get("title", filename.replace("_", " ").replace(".pdf", "").title()) if meta else filename,
            "discipline": meta.get("disciplina", meta.get("discipline", "Ciências")) if meta else "Ciências",
            "grade": meta.get("ano", "Geral") if meta else "Geral",
        }


# =========================
# 🚀 PIPELINE PRINCIPAL
# =========================
class ScienceRAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingClient()
        self.meta_mgr = MetadataManager(METADATA_FILE)
        self.vision = GeminiVisionClient()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        print(f"🔌 Conectando ao Qdrant ({QDRANT_HOST}:{QDRANT_PORT})...")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60, check_compatibility=False)
        self._setup_collection()

    def _setup_collection(self):
        try:
            collections = self.qdrant.get_collections().collections
            exists = any(c.name == COLLECTION_NAME for c in collections)
            
            if exists:
                info = self.qdrant.get_collection(COLLECTION_NAME)
                # Verifica se dimensão bate, se não recria
                if info.config.params.vectors.size != self.embedder.dim:
                    print("♻️ Dimensão nova detectada. Recriando coleção...")
                    self.qdrant.delete_collection(COLLECTION_NAME)
                    exists = False
            
            if not exists:
                self.qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
                )
                print(f"✅ Coleção '{COLLECTION_NAME}' criada.")
        except Exception as e:
            print(f"⚠️ Erro Qdrant: {e}")

    def process_pdf(self, file_path: str):
        filename = os.path.basename(file_path)
        print(f"🔬 Processando: {filename}")
        meta = self.meta_mgr.get(filename)
        
        full_text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                
                # Se pouco texto, tenta OCR/Vision se tiver chave
                if len(txt) < 100 and self.vision.enabled:
                    print(f"   👁️ Usando Gemini Vision na página {page.page_number}...")
                    vision_text = self.vision.analyze_page_image(page.to_image(resolution=200).original)
                    if vision_text: txt += "\n" + vision_text
                    
                full_text.append(TextProcessor.clean_text(txt))
        
        # Junta tudo e depois divide
        raw_text = "\n\n".join(full_text)
        chunks = self.splitter.split_text(raw_text)
        
        if not chunks: 
            print("   ⚠️ Arquivo vazio ou ilegível.")
            return

        points = []
        vecs = self.embedder.encode([self.embedder.prepare_text(c, meta) for c in chunks])
        
        for i, (txt, vec) in enumerate(zip(chunks, vecs)):
            payload = {
                "text_content": txt, 
                "source_id": filename, 
                "metadata": meta,
                "chunk_index": i
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

        if points:
            # Upsert em batches para não estourar memória
            batch_size = 64
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.qdrant.upsert(COLLECTION_NAME, points=batch)
            print(f"   ✅ Indexado: {len(points)} chunks.")

    def search_safe(self, query: str, limit: int = 5):
        """Busca Híbrida: Vector Search (recuperação) + Reranking (precisão)"""
        q_vec = self.embedder.encode([self.embedder.prepare_text(query, {}, is_query=True)])[0]
        
        # 1. Recuperação (Retrieval) - Trazemos mais resultados (ex: 20) para filtrar depois
        initial_limit = 20 
        try:
            if hasattr(self.qdrant, 'search'):
                hits = self.qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=initial_limit)
            else:
                 hits = self.qdrant.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=initial_limit).points
        except Exception as e:
            print(f"❌ Erro busca Qdrant: {e}")
            return []

        # 2. Reordenamento (Reranking) - O segredo da precisão
        reranked_hits = self.embedder.rerank_results(query, hits, top_k=limit)
        return reranked_hits

if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR): os.makedirs(DOCS_DIR)
    pipeline = ScienceRAGPipeline()
    import sys
    
    # Modo Busca
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n🔎 Buscando: '{query}'")
        start = time.time()
        res = pipeline.search_safe(query)
        total_time = time.time() - start
        
        print(f"⏱️ Tempo total: {total_time:.3f}s")
        for i, r in enumerate(res):
            score_icon = "🟢" if r.score > 0 else "🔴" # Scores do reranker podem variar
            snippet = r.payload.get('text_content','').replace('\n', ' ')[:150]
            print(f"{i+1}. {score_icon} [{r.score:.3f}] {r.payload['source_id']}: \"{snippet}...\"")
            
    # Modo Indexação
    else:
        print("\n📂 Iniciando Indexação de PDFs...")
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        for f in files: 
            pipeline.process_pdf(os.path.join(DOCS_DIR, f))