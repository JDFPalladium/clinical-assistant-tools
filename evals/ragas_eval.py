# rag_eval_full.py
import os
import time
import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI

from ai_tools.helpers import expand_query, cosine_similarity_numpy, format_sources_for_html
from ai_tools.phi_filter import detect_and_redact_phi

from ragas.evaluation import evaluate
from ragas.metrics import faithfulness, answer_relevancy

if os.path.exists("config.env"):
    load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")

# -------------------------------
# Load embeddings & summaries
# -------------------------------
embeddings = np.load("data/processed/lp/summary_embeddings/embeddings.npy")
df_summaries = pd.read_csv("data/processed/lp/summary_embeddings/index.tsv", sep="\t")
embedding_model = OpenAIEmbedding()

# -------------------------------
# Load global index retriever
# -------------------------------
global_index_path = "data/processed/lp/indices/Global"
storage_context_arv = StorageContext.from_defaults(persist_dir=global_index_path)
global_index = load_index_from_storage(storage_context_arv)
global_retriever = VectorIndexRetriever(index=global_index, similarity_top_k=3)

# -------------------------------
# Instantiate all LLMs up front
# -------------------------------
llms_answer = {
    "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0.0),
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
}

llms_reranker = {
    "gpt-4o": OpenAI(model="gpt-4o", temperature=0.0),
    "gpt-4o-mini": OpenAI(model="gpt-4o-mini", temperature=0.0)
}

reranker_wrappers = {
    name: LLMRerank(llm=llm_reranker, top_n=2)
    for name, llm_reranker in llms_reranker.items()
}

# -------------------------------
# Core RAG evaluation
# -------------------------------
def run_rag_eval(test_queries, answer_llm_name="gpt-4o", reranker_llm_name="gpt-4o-mini"):
    answer_llm = llms_answer[answer_llm_name]
    reranker = reranker_wrappers[reranker_llm_name]

    results = []

    for query in test_queries:
        timings = {}
        tokens = {}

        # -----------------------
        # Step 1: Redact PHI
        # -----------------------
        start = time.time()
        query_redacted = detect_and_redact_phi(query)["redacted_text"]
        timings["phi_redaction"] = time.time() - start

        # -----------------------
        # Step 2: Query Expansion
        # -----------------------
        start = time.time()
        expanded_query = expand_query(query_redacted, answer_llm)
        # expanded_query = query_redacted 
        timings["query_expansion"] = time.time() - start

        # -----------------------
        # Step 3: Embedding & Chunk Retrieval
        # -----------------------
        start = time.time()
        query_vec = embedding_model.get_text_embedding(expanded_query)
        sims = cosine_similarity_numpy(query_vec, embeddings)
        top_paths = df_summaries.loc[sims.argsort()[-3:][::-1], "vectorestore_path"].tolist()

        all_nodes = []
        for path in top_paths:
            ctx = StorageContext.from_defaults(persist_dir=path)
            index = load_index_from_storage(ctx)
            retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
            all_nodes.extend(retriever.retrieve(expanded_query))

        all_nodes.extend(global_retriever.retrieve(expanded_query))
        timings["chunk_retrieval"] = time.time() - start

        # -----------------------
        # Step 4: Deduplicate & Rerank
        # -----------------------
        start = time.time()
        unique_nodes = {}
        for node in all_nodes:
            node_id = node.node.node_id
            if node_id not in unique_nodes or node.score > unique_nodes[node_id].score:
                unique_nodes[node_id] = node
        deduped_nodes = list(unique_nodes.values())

        sources = reranker.postprocess_nodes(deduped_nodes, QueryBundle(expanded_query))
        timings["reranking"] = time.time() - start

        if not sources:
            results.append({
                "question": query,
                "retrieved_contexts": [],
                "answer": "No relevant information found.",
                "timings": timings
            })
            continue

        rag_sources_formatted = format_sources_for_html(sources)

        # -----------------------
        # Step 5: Answer Generation
        # -----------------------
        start = time.time()
        retrieved_text = "\n\n".join([f"Source {i+1}: {s.text}" for i, s in enumerate(sources)])
        prompt = (
            "Answer the clinician's question using only the provided guideline excerpts.\n"
            "Include only information explicitly present in the sources.\n"
            "Respond with plain text only.\n"
            "If the answer cannot be found, say: 'No relevant information found.'\n\n"
            f"Clinician question: {query_redacted}\n\nGuideline excerpts:\n{retrieved_text}"
        )
        response = answer_llm.invoke(prompt)
        timings["answer_generation"] = time.time() - start

        results.append({
            "question": query,
            "retrieved_contexts": [s.text for s in sources],
            "answer": response.content,
            "timings": timings
        })

    # -------------------------------
    # Step 6: Compute RAGAS metrics
    # -------------------------------
    ragas_data = Dataset.from_list(results)
    eval_results = evaluate(ragas_data, metrics=[faithfulness, answer_relevancy])
    df_eval = eval_results.to_pandas()

    # Merge timings, sources back into final dataframe
    df_eval["timings"] = [r["timings"] for r in results]
    df_eval["retrieved_contexts"] = [r["retrieved_contexts"] for r in results]

    return df_eval

# -------------------------------
# CLI / run all combinations
# -------------------------------
if __name__ == "__main__":
    test_queries = [
        "What are important drug interactions with dolutegravir?",
        "How should PrEP be provided to adolescent girls?"
        # "When is cotrimoxazole prophylaxis indicated?",
        # "What are the guidelines for ART failure?",
        # "How do you manage HIV in pregnancy?",
        # "When should infants start ART?"
    ]

    for answer_llm_name in ["gpt-4o", "gpt-4o-mini"]:
        for reranker_llm_name in ["gpt-4o-mini"]:
            print(f"\n=== Running evaluation: answer={answer_llm_name}, reranker={reranker_llm_name} ===")
            df = run_rag_eval(test_queries, answer_llm_name, reranker_llm_name)
            csv_name = f"rag_eval_{answer_llm_name}_{reranker_llm_name}.csv"
            df.to_csv(csv_name, index=False)
            print(f"✅ Saved results to {csv_name}")
