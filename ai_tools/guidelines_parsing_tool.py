# my_tools/rag_guidelines_tool.py
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import ChatOpenAI
from ai_tools.helpers import expand_query, cosine_similarity_numpy, format_sources_for_html
from ai_tools.phi_filter import detect_and_redact_phi
if os.path.exists("config.env"):
    load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")

# -------------------------------
# Lazy-loaded globals for standalone usage
# -------------------------------
_embeddings = None
_df = None
_embedding_model = None
_reranker = None
_llm_llama = None
_llm_langchain = None
_global_retriever = None

def _lazy_load(hybrid=True):
    global _embeddings, _df, _embedding_model, _reranker, _llm_llama, _llm_langchain, _global_retriever
    if _embeddings is None:
        _embeddings = np.load("data/processed/lp/summary_embeddings/embeddings.npy")
        _df = pd.read_csv("data/processed/lp/summary_embeddings/index.tsv", sep="\t")
        _embedding_model = OpenAIEmbedding()
        _llm_llama = OpenAI(model="gpt-4o-mini", temperature=0.0)
        _reranker = LLMRerank(llm=_llm_llama, top_n=2)
        _llm_langchain = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

    if hybrid and _global_retriever is None:
        global_index_path = "data/processed/lp/indices/Global"   # adjust path
        storage_context_arv = StorageContext.from_defaults(persist_dir=global_index_path)
        index_arv = load_index_from_storage(storage_context_arv)
        _global_retriever = VectorIndexRetriever(index=index_arv, similarity_top_k=3)

# -------------------------------
# Core RAG retrieve function
# -------------------------------
def rag_retrieve(query: str, llm= None, global_retriever = None, embeddings=None, df=None, embedding_model=None, reranker=None, hybrid=True) -> dict:
    """Perform RAG search, using passed resources or defaults if standalone."""
    # fallback to lazy-loaded objects if arguments not provided
    if embeddings is None or df is None or embedding_model is None or reranker is None or (hybrid and global_retriever is None):
        _lazy_load(hybrid=hybrid)
        llm = llm or _llm_langchain
        global_retriever = global_retriever or _global_retriever
        embeddings = _embeddings
        df = _df
        embedding_model = _embedding_model
        reranker = _reranker

    query_bundle = QueryBundle(query)
    expanded_query = expand_query(query, llm)
    query_embedding = embedding_model.get_text_embedding(expanded_query)
    
    similarities = cosine_similarity_numpy(query_embedding, embeddings)
    top_indices = similarities.argsort()[-3:][::-1]
    selected_paths = df.loc[top_indices, "vectorestore_path"].tolist()

    all_sources = []
    for path in selected_paths:
        storage_context = StorageContext.from_defaults(persist_dir=path)
        index = load_index_from_storage(storage_context)
        raw_retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        all_sources.extend(raw_retriever.retrieve(expanded_query))

    if hybrid:
        sources_arv = global_retriever.retrieve(expanded_query)
        all_sources.extend(sources_arv)

    # Deduplicate by node_id
    unique_sources = {}
    for src in all_sources:
        node_id = src.node.node_id
        if node_id not in unique_sources or src.score > unique_sources[node_id].score:
            unique_sources[node_id] = src
    deduped_sources = list(unique_sources.values())

    sources = reranker.postprocess_nodes(deduped_sources, query_bundle)

    if not sources:
        return {"answer": "No relevant information found.", "last_tool": "rag_retrieve"}

    retrieved_text = "\n\n".join([f"Source {i+1}: {s.text}" for i, s in enumerate(sources)])
    prompt = (
        "Answer the clinician's question using only the provided guideline excerpts.\n"
        "Include only information explicitly present in the sources.\n"
        "Return concise bullet points or short sentences.\n"
        "If the answer cannot be found, say: 'No relevant information found.'\n\n"
        f"Clinician question: {query}\n\nGuideline excerpts:\n{retrieved_text}"
    )

    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "rag_sources": format_sources_for_html(sources),
        "last_tool": "rag_retrieve"
    }

# -------------------------------
# Standalone wrapper
# -------------------------------
def rag_retrieve_standalone(query: str, hybrid=True) -> dict:
    """Run RAG tool independently, without needing external retriever or llm."""
    _lazy_load(hybrid=hybrid)

    # Dummy retriever that returns nothing (so only local vectors are used)
    class DummyRetriever:
        def retrieve(self, query):
            return []

    llm = _llm_langchain  # already created in _lazy_load
    query_redacted = detect_and_redact_phi(query)["redacted_text"]
    retriever = _global_retriever if hybrid else DummyRetriever()
    return rag_retrieve(query_redacted, llm, retriever, hybrid=hybrid)


# -------------------------------
# CLI entrypoint for testing
# -------------------------------
if __name__ == "__main__":
    q = input("Enter your query: ")
    h = input("Use hybrid retrieval? (y/n): ").strip().lower() == 'y'
    result = rag_retrieve_standalone(q, hybrid=h)
    print("\n=== Answer ===")
    print(result["answer"])

