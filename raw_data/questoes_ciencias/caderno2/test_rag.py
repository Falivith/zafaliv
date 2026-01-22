import time
import statistics
from rag_science_pipeline import ScienceRAGPipeline

# =========================
# 🧪 SCRIPT DE VALIDAÇÃO
# =========================
TEST_QUESTIONS = [
    {"q": "1º princípio do Direito Ambiental?", "cat": "Biologia"},
    {"q": "Como ocorre a fotossíntese?", "cat": "Biologia"},
    {"q": "Qual a fórmula da velocidade média?", "cat": "Física"},
    {"q": "O que a norma ISO 14001?", "cat": "Química"},
    {"q": "O que é Reversibilidade?", "cat": "Física"}
]

def run_validation():
    print("🚀 Iniciando Teste Rápido...")
    
    try:
        pipeline = ScienceRAGPipeline()
    except Exception as e:
        print(f"❌ Falha ao iniciar pipeline: {e}")
        return

    scores = []
    
    for t in TEST_QUESTIONS:
        query = t["q"]
        print(f"\n❓ Pergunta: '{query}' ({t['cat']})")
        
        start = time.time()
        # 🛡️ FIX: Usa o método search_safe da classe pipeline, não acessa .qdrant direto
        hits = pipeline.search_safe(query, limit=3)
        duration = time.time() - start
        
        if hits:
            top = hits[0]
            scores.append(top.score)
            print(f"   ✅ Top Score: {top.score:.4f} | Tempo: {duration:.3f}s")
            # Mostra o metadado recuperado
            meta = top.payload.get('metadata', {})
            print(f"   📄 Fonte: {meta.get('title', 'N/A')} - {meta.get('discipline', 'N/A')}")
            print(f"   📝 Trecho: {top.payload.get('text_content', '')[:120]}...")
        else:
            print("   ⚠️ Sem resultados.")
            scores.append(0)

    avg = statistics.mean(scores) if scores else 0
    print("\n" + "="*30)
    print(f"📊 Média Geral: {avg:.4f}")
    if avg > 0.65: print("✅ RAG está saudável!")
    else: print("⚠️ RAG precisa de ajustes (verifique se os PDFs estão indexados).")

if __name__ == "__main__":
    run_validation()