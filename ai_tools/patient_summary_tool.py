import os
import re
import sqlite3
import json
import pandas as pd
from functools import lru_cache
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# load local modules
from ai_tools.patient_tool_helpers import patient_data_to_text, build_decoder_dict

# --- Load env vars ---
if os.path.exists("config.env"):
    load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")

# ------------------------
# Lazy loaders
# ------------------------
@lru_cache()
def get_main_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)

@lru_cache()
def get_rag_retriever():
    """
    Load retriever for guideline documents.
    """
    # Global retriever (load global index)
    global_index_path = "data/processed/lp/indices/Global"   # adjust to your real folder
    storage_context_arv = StorageContext.from_defaults(persist_dir=global_index_path)
    index_arv = load_index_from_storage(storage_context_arv)
    retriever = VectorIndexRetriever(index=index_arv, similarity_top_k=2)
    return retriever

# ------------------------
# Main function
# ------------------------
def patient_summary(llm, pk_hash: str) -> dict:
    """
    Generates a summary of patient records that flags items of concern for clinicians.
    """
    
    if not pk_hash:
        raise ValueError("pk_hash is required in state for SQL queries.")

    conn = sqlite3.connect("data/processed/site_database.sqlite")
    table_mappings = json.load(open("data/processed/table_mappings.json"))
    decoder = build_decoder_dict("data/processed/Site tables decoding map.csv")


    _, topic_dict = patient_data_to_text(pk_hash, conn, table_mappings, decoder)


    # --- Final prompt ---
    prompt = f"""
    You are a clinical assistant reviewing structured patient data.
    Your task is to summarize only clinical concerns requiring follow-up or action 
    (exclude routine facts).

    Output as a bullet list: • [Theme] – [Insight] (≤2 lines, actionable).

    1. **Medication & Treatment**

    Items of concern include:
    - Any history of adverse drug reactions
    - Medication non-adherence and challenges with adherence
    - History of missed appointments
    - Any history of treatment failure, regimen changes, or drug resistance

    {topic_dict.get("Medication & Treatment", [])}

    2. **Laboratory & Diagnostics**

    Items of concern include:
    - Any history of HIV viral non-suppression, low CD4 counts, or other abnormal lab results

    {topic_dict.get("Laboratory & Diagnostics", [])}

    3. **Clinical Events**
    
    Items of concern include:
    - Any history of complaints and diagnoses
    - Any history of hospitalizations, surgeries, or other significant clinical events

    {topic_dict.get("Clinical Events & Patient care", [])}

    4. **Psychosocial & Behavioral Risks**
    {topic_dict.get("Psychosocial & Behavioral Risks", [])}

    5. **Reproductive & Sexual Health**
    {topic_dict.get("Reproductive & Sexual Health", [])}

    If no issues, return: 
    • No clinical concerns identified.
    """
    print(prompt)

    response = llm.invoke(prompt)
    return {"answer": response.content}

# ------------------------
# Standalone wrapper
# ------------------------
def run_summary_standalone(pk_hash: str) -> dict:
    llm = get_main_llm()
    return patient_summary(llm=llm, pk_hash=pk_hash)


# ------------------------
# CLI entrypoint
# ------------------------
if __name__ == "__main__":
    pk = input("Enter your patient PK hash: ")
    result = run_summary_standalone(pk)
    print("\n--- Patient Summary ---")
    print(result["answer"])