import os
import json
import time
import uuid
import glob
import dotenv
import torch
import warnings
from typing import List, Dict, Any

# =========================
# 📦 IMPORTAÇÃO SEGURA & DEPENDÊNCIAS
# =========================
missing_deps = []

# Qdrant & Sentence Transformers
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
    from sentence_transformers import SentenceTransformer
except ImportError:
    missing_deps.append("qdrant-client sentence-transformers torch")

# PDF Plumber
try:
    import pdfplumber
except ImportError:
    missing_deps.append("pdfplumber")

# Google GenAI (Novo SDK)
try:
    from google import genai
    from google.genai import types
except ImportError:
    missing_deps.append("google-genai")

if missing_deps:
    print(f"❌ Erro: Faltam dependências. Instale rodando:")
    print(f"pip install {' '.join(missing_deps)} python-dotenv")
    exit(1)

# =========================
# ⚙️ CONFIGURAÇÕES
# =========================
dotenv.load_dotenv()

# Silenciar avisos
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
warnings.filterwarnings("ignore", category=UserWarning)

# Configurações do Ambiente
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_QUESTOES", "questoes_ciencias_v1")
DOCS_DIR = os.getenv("DOCS_DIR", "./provas_pdfs")

# Chaves de API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.0-flash" 
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
VECTOR_SIZE = 1024

# =========================
# 🧠 EXTRAÇÃO INTELIGENTE (PDF -> JSON)
# =========================
class PDFProcessor:
    def __init__(self):
        if not GEMINI_API_KEY:
            print("⚠️ AVISO: GEMINI_API_KEY não encontrada no .env. A estruturação falhará.")
            self.client = None
        else:
            try:
                # Inicialização do novo SDK
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"✨ Gemini ({GEMINI_MODEL_NAME}) ativado (via google-genai).")
            except Exception as e:
                print(f"❌ Erro ao inicializar Gemini: {e}")
                self.client = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrai texto bruto do PDF."""
        full_text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
            return "\n".join(full_text)
        except Exception as e:
            print(f"❌ Erro ao ler PDF {pdf_path}: {e}")
            return ""

    def parse_questions_with_gemini(self, raw_text: str, source_filename: str) -> List[Dict[str, Any]]:
        """Usa o novo SDK do Gemini para estruturar o JSON."""
        if not self.client or not raw_text.strip():
            return []

        print(f"   🤖 Interpretando questões de '{source_filename}'...")
        
        prompt = """
        Você é um especialista em estruturar dados de provas escolares.
        Analise o texto abaixo e extraia as questões.
        
        Retorne estritamente um JSON Array onde cada objeto segue esta estrutura:
        {
            "id": (número ou identificador),
            "texto_base": (texto de apoio, se houver),
            "enunciado": (a pergunta),
            "alternativas": { "A": "...", "B": "..." },
            "full_text": (concatenação de tudo para busca)
        }
        
        Texto da Prova:
        """ + raw_text[:30000]

        try:
            # Chamada atualizada para google-genai
            response = self.client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            # Parsing da resposta
            if response.text:
                data = json.loads(response.text)
                if isinstance(data, list):
                    for q in data:
                        q['source_file'] = source_filename
                        # Garante full_text para o embedding
                        if 'full_text' not in q or not q['full_text']:
                            alts = " ".join([f"{k}) {v}" for k,v in q.get('alternativas', {}).items()])
                            q['full_text'] = f"{q.get('texto_base','')} {q.get('enunciado','')} {alts}".strip()
                    return data
            return []
            
        except Exception as e:
            print(f"❌ Erro na estruturação com IA: {e}")
            return []

    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        all_questions = []
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ Nenhum PDF encontrado em '{folder_path}'")
            return []

        print(f"📂 Encontrados {len(pdf_files)} arquivos PDF.")
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            print(f"📄 Processando: {filename}...")
            raw_text = self.extract_text_from_pdf(pdf_file)
            
            if len(raw_text) > 50:
                questions = self.parse_questions_with_gemini(raw_text, filename)
                if questions:
                    all_questions.extend(questions)
                    print(f"   ✅ {len(questions)} questões extraídas.")
                else:
                    print("   ⚠️ Nenhuma questão estruturada encontrada.")
        return all_questions

# =========================
# 🔍 INDEXADOR (RAG)
# =========================
class QuestionIndexer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Hardware de Inferência: {self.device.upper()}")
        
        print(f"📥 Carregando Embedder: {EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        
        print(f"🔌 Conectando ao Qdrant ({QDRANT_HOST}:{QDRANT_PORT})...")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._setup_collection()

    def _setup_collection(self):
        try:
            collections = self.qdrant.get_collections().collections
            exists = any(c.name == COLLECTION_NAME for c in collections)
            if not exists:
                print(f"🆕 Criando coleção '{COLLECTION_NAME}'...")
                self.qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                )
        except Exception as e:
            print(f"⚠️ Erro Qdrant Setup: {e}")

    def generate_vector(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

    def index_questions(self, questions_data: List[Dict[str, Any]]):
        if not questions_data: return
        points = []
        print(f"\n🚀 Indexando {len(questions_data)} questões no Qdrant...")
        
        for q in questions_data:
            try:
                full_text = q.get("full_text", "")
                if not full_text: continue

                vector = self.generate_vector(full_text)
                
                payload = {
                    "texto_base": q.get("texto_base", ""),
                    "enunciado": q.get("enunciado", ""),
                    "alternativas": q.get("alternativas", {}),
                    "full_text": full_text,
                    "source_file": q.get("source_file", "manual"),
                    "id": q.get("id")
                }
                
                points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

            except Exception as e:
                print(f"❌ Erro processando questão ID {q.get('id')}: {e}")

        if points:
            try:
                self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f"✅ Sucesso! {len(points)} questões enviadas.")
            except Exception as e:
                print(f"❌ Erro no envio para Qdrant: {e}")

    def search_similar(self, query: str, limit: int = 3):
        print(f"\n🔎 Buscando por: '{query}'")
        vector = self.generate_vector(query)
        
        hits = []
        try:
            # Tenta usar o método moderno query_points (compatível com v1.10+)
            if hasattr(self.qdrant, 'query_points'):
                results = self.qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    limit=limit
                )
                hits = results.points
            # Fallback para o método search (versões v1.0 - v1.9)
            elif hasattr(self.qdrant, 'search'):
                hits = self.qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=vector,
                    limit=limit
                )
            else:
                print("❌ Erro: Cliente Qdrant não possui método 'search' nem 'query_points'. Verifique a versão.")
                return

        except Exception as e:
            print(f"❌ Erro durante a busca: {e}")
            return

        for hit in hits:
            print(f"\n🎯 Score: {hit.score:.4f} | Fonte: {hit.payload.get('source_file')}")
            print(f"   Enunciado: {hit.payload.get('enunciado')[:150]}...")

# =========================
# 🧪 EXECUÇÃO
# =========================
if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"📁 Pasta '{DOCS_DIR}' criada.")

    # 1. Processar PDFs
    processor = PDFProcessor()
    extracted_questions = processor.process_folder(DOCS_DIR)
    
    # 2. Indexar Resultados
    if extracted_questions:
        indexer = QuestionIndexer()
        indexer.index_questions(extracted_questions)
        
        # 3. Teste Rápido
        indexer.search_similar("controle da célula núcleo")
    else:
        print("\nℹ️ Nenhuma nova questão para processar.")