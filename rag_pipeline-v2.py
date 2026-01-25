"""
RAG Pipeline - STABILNA VERZIJA (OpenAI gpt-4o-mini)
POPRAVLJENO: Evaluacija koristi direktni OpenAI klijent za JSON mode.
"""

import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Standardni OpenAI library (za evaluaciju)
import openai

# LlamaIndex imports (za RAG)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI # Preimenujemo da ne bude sukoba

# ================== KONFIGURACIJA ==================
load_dotenv()

class RAGConfig:
    # U .env stavi: OPENAI_API_KEY=sk-proj-...
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY nedostaje!")
    
    PDF_INPUT_DIR = "./Dokumenti"
    RESULTS_DIR = "./Rezultati" 
    QUESTION_DATASET_PATH = "./evaluation_questions.json"
    
    # Model
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    
    SLEEP_BETWEEN_QUERIES = 0.5 
    
     # Chunking Strategije
    CHUNKING_STRATEGIES = {
        "micro_overlap20": {
            "type": "fixed",
            "chunk_size": 128,
            "chunk_overlap": 20
        },
        "micro_overlap40": {
            "type": "fixed",
            "chunk_size": 128,
            "chunk_overlap": 40
        },
        "standard_overlap50": {
            "type": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50
        },
        "standard_overlap100": {
            "type": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 100
        },
        "macro_overlap100": {
            "type": "fixed",
            "chunk_size": 1024,
            "chunk_overlap": 100
        },
        "macro_overlap200": {
            "type": "fixed",
            "chunk_size": 1024,
            "chunk_overlap": 200
        },
        "semantic": {
            "type": "semantic",
            "buffer_size": 1,
            "breakpoint_percentile_threshold": 95
        }
    }

# ================== INICIJALIZACIJA ==================

def setup_environment():
    os.makedirs(RAGConfig.PDF_INPUT_DIR, exist_ok=True)
    os.makedirs(RAGConfig.RESULTS_DIR, exist_ok=True)

def initialize_models():
    print(f"\n[INIT] Inicijaliziram OpenAI: {RAGConfig.LLM_MODEL}")
    
    # 1. Embedding (Lokalno)
    embed_model = HuggingFaceEmbedding(
        model_name=RAGConfig.EMBEDDING_MODEL,
        cache_folder="./model_cache"
    )
    
    # 2. LLM (LlamaIndex Wrapper za RAG)
    llm = LlamaOpenAI(
        model=RAGConfig.LLM_MODEL, 
        temperature=0.1, 
        api_key=RAGConfig.OPENAI_API_KEY
    )
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    return embed_model, llm

# ================== POMOĆNE FUNKCIJE ==================

def load_documents():
    if not os.path.exists(RAGConfig.PDF_INPUT_DIR): raise FileNotFoundError("Nema dokumenata!")
    return SimpleDirectoryReader(input_dir=RAGConfig.PDF_INPUT_DIR, required_exts=[".pdf"], recursive=True).load_data()

def load_questions():
    if not os.path.exists(RAGConfig.QUESTION_DATASET_PATH): return []
    with open(RAGConfig.QUESTION_DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f).get("evaluation_questions", [])

def build_index(documents, strategy_name, strategy_config, embed_model):
    print(f"\n[INDEX] Gradim index: {strategy_name}")
    
    if strategy_config["type"] == "fixed":
        parser = SentenceSplitter(
            chunk_size=strategy_config["chunk_size"], 
            chunk_overlap=strategy_config["chunk_overlap"]
        )
    elif strategy_config["type"] == "semantic":
        parser = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=95, 
            embed_model=embed_model
        )
        
    nodes = parser.get_nodes_from_documents(documents)
    return VectorStoreIndex(nodes=nodes, embed_model=embed_model, show_progress=True)

# ================== EVALUACIJA (FIXED) ==================

def process_query(index, question, expected, llm_rag):
    """
    Ovdje koristimo DVA klijenta:
    1. 'llm_rag' (LlamaIndex) za generiranje odgovora iz dokumenata.
    2. 'eval_client' (Raw OpenAI) za ocjenjivanje (jer treba JSON mode).
    """
    
    # --- 1. RAG GENERACIJA (Koristi LlamaIndex) ---
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(question)
    context = "\n".join([n.text for n in nodes])
    
    start = time.time()
    # Generiranje odgovora
    response = index.as_query_engine(llm=llm_rag).query(question)
    generated_text = response.response
    latency = time.time() - start
    
    # --- 2. EVALUACIJA (Koristi direktni OpenAI client) ---
    # Kreiramo privremeni klijent samo za evaluaciju
    eval_client = openai.OpenAI(api_key=RAGConfig.OPENAI_API_KEY)
    
    eval_prompt = f"""
    You are an expert evaluator. Output a JSON object.
    
    Question: {question}
    Expected Answer: {expected}
    Generated Answer: {generated_text}
    Context Used: {context}
    
    Task: Rate these metrics from 0.0 to 1.0 based on the data above.
    
    Metrics:
    - faithfulness: Is the generated answer derived ONLY from the context?
    - context_relevancy: Does the retrieved context contain the answer?
    - answer_correctness: Is the generated answer factually correct compared to the expected answer?
    
    REQUIRED OUTPUT FORMAT (JSON ONLY):
    {{ "faithfulness": 0.5, "context_relevancy": 0.5, "answer_correctness": 0.5 }}
    """
    
    try:
        # Koristimo 'json_object' mode koji garantira da nema greške u formatu
        eval_res = eval_client.chat.completions.create(
            model=RAGConfig.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": eval_prompt}
            ],
            response_format={"type": "json_object"} 
        )
        scores = json.loads(eval_res.choices[0].message.content)
    except Exception as e:
        print(f"[EVAL ERROR] {e}")
        scores = {"faithfulness": 0, "context_relevancy": 0, "answer_correctness": 0}
        
    return {
        "question": question,
        "generated_answer": generated_text,
        "latency": latency,
        "faithfulness": scores.get("faithfulness", 0),
        "context_relevancy": scores.get("context_relevancy", 0),
        "answer_correctness": scores.get("answer_correctness", 0)
    }

# ================== MAIN ==================

def main():
    setup_environment()
    embed_model, llm_rag = initialize_models()
    documents = load_documents()
    questions = load_questions()
    
    if not questions: 
        print("Nema pitanja!")
        return

    all_results = []
    
    for name, config in RAGConfig.CHUNKING_STRATEGIES.items():
        print(f"\n--- STRATEGIJA: {name} ---")
        index = build_index(documents, name, config, embed_model)
        
        strategy_res = []
        for i, qa in enumerate(questions):
            print(f"[{i+1}/{len(questions)}] {qa['question'][:50]}...")
            
            res = process_query(index, qa["question"], qa["expected_answer"], llm_rag)
            res["strategy"] = name
            
            # Ispis rezultata u konzolu
            print(f"   -> F={res['faithfulness']} | R={res['context_relevancy']} | C={res['answer_correctness']}")
            
            strategy_res.append(res)
            time.sleep(RAGConfig.SLEEP_BETWEEN_QUERIES)
            
        all_results.extend(strategy_res)
    
    # Save
    if all_results:
        df = pd.DataFrame(all_results)
        t = datetime.now().strftime("%Y%m%d_%H%M")
        
        csv_path = f"{RAGConfig.RESULTS_DIR}/openai_results_{t}.csv"
        df.to_csv(csv_path, index=False)
        
        # Izračunaj prosjeke
        print("\n=== KONAČNI REZULTATI (PROSJEK) ===")
        summary = df.groupby('strategy')[['faithfulness', 'context_relevancy', 'answer_correctness', 'latency']].mean()
        print(summary)
        summary.to_csv(f"{RAGConfig.RESULTS_DIR}/summary_{t}.csv")
        
        print(f"\n[GOTOVO] Rezultati spremljeni u {csv_path}")

if __name__ == "__main__":
    main()