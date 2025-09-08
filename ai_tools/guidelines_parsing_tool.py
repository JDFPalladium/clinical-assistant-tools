# my_tools/rag_guidelines_tool.py
import numpy as np
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from .helpers import expand_query, cosine_similarity_numpy, format_sources_for_html

# -------------------------------
# Lazy-loaded globals for standalone usage
# -------------------------------
_embeddings = None
_df = None
_embedding_model = None
_reranker = None
_llm_llama = None

def _lazy_load():
    global _embeddings, _df, _embedding_model, _reranker, _llm_llama
    if _embeddings is None:
        _embeddings = np.load("data/processed/lp/summary_embeddings/embeddings.npy")
        _df = pd.read_csv("data/processed/lp/summary_embeddings/index.tsv", sep="\t")
        _embedding_model = OpenAIEmbedding()
        _llm_llama = OpenAI(model="gpt-4o-mini", temperature=0.0)
        _reranker = LLMRerank(llm=_llm_llama, top_n=2)

# -------------------------------
# Core RAG retrieve function
# -------------------------------
def rag_retrieve(query: str, llm, global_retriever, embeddings=None, df=None, embedding_model=None, reranker=None):
    """Perform RAG search, using passed resources or defaults if standalone."""
    # fallback to lazy-loaded objects if arguments not provided
    if embeddings is None or df is None or embedding_model is None or reranker is None:
        _lazy_load()
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
def rag_retrieve_standalone(query: str, llm, global_retriever):
    """Lazy-loading wrapper for independent use."""
    return rag_retrieve(query, llm, global_retriever)
