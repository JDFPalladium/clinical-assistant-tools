# sql_tool.py
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

from ai_tools.phi_filter import detect_and_redact_phi
from ai_tools.schemas import table_descriptions 
from ai_tools.patient_tool_helpers import patient_data_to_text, build_decoder_dict

# --- Load env vars ---
if os.path.exists("config.env"):
    load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")


# ------------------------
# Utilities
# ------------------------
def check_guideline_needed(question, llm=None):
    """
    Ask an LLM whether guideline context is likely needed for this question.
    Returns True/False.
    """
    if llm is None:
        llm = get_summarizer_llm()  # lightweight 3.5 turbo
    
    prompt = (
        "You are a classifier that decides whether a clinical question requires consulting external clinical guidelines.\n\n"
        "Rules:\n"
        "- If the question is only about retrieving or summarizing patient-specific data (labs, visits, medications, demographics), answer NO.\n"
        "- If the question is about interpretation, management, treatment recommendations, or what action to take based on results, answer YES.\n\n"
        "Return only YES or NO. Do not include explanations.\n\n"
        f"Question: {question}\n"
        "Final answer (YES or NO):"
    )

    response = llm.invoke(prompt).content.strip().lower()
    return response.startswith("yes")

def build_summarization_prompt(question, sources):
    """
    Build a prompt to summarize guideline documents relevant to a question.
    
    Args:
        question (str): User question.
        sources (list of str): Text of the retrieved guideline documents.
    
    Returns:
        str: Prompt to send to the LLM.
    """
    prompt = (
        "You are a clinical assistant helping a healthcare provider answer a question using HIV/AIDS guidelines.\n\n"
        f"Question: {question}\n\n"
        "Below are excerpts from guideline documents that may be relevant:\n\n"
        f"{sources}\n\n"
        "Please summarize the most important points from the guideline excerpts that are relevant to the question. "
        "Focus on actionable clinical guidance, dosing, monitoring, and follow-up instructions. "
        "Do not include irrelevant details. "
        "Keep the summary concise, accurate, and suitable for clinical decision-making."
    )
    return prompt


def safe_json_load(s):
    """
    Strip ```json``` or ``` ``` from LLM output, then parse JSON.
    """
    # Remove code fences and optional language hints
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    return json.loads(s)

def build_table_selection_prompt():
    human_template = """
Question: "{question}"

Here are the patient tables and their descriptions:
{table_descriptions}

Return only the exact table names needed to answer the question, in JSON format like:
["Table Name 1", "Table Name 2"]

Do not include columns, explanations, or anything else.
"""
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    prompt = ChatPromptTemplate.from_messages([human_message])
    return prompt

def get_relevant_tables(question, table_descriptions, llm=None):
    prompt = build_table_selection_prompt()
    # Format with actual values
    messages = prompt.format_prompt(
        question=question,
        table_descriptions=json.dumps(table_descriptions, indent=2)
    ).to_messages()

    # Send to LLM
    response = llm(messages)
    tables = safe_json_load(response.content)

    return tables


# ------------------------
# Lazy loaders
# ------------------------
@lru_cache()
def get_summarizer_llm():
    return ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0125")

@lru_cache()
def get_main_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)

@lru_cache()
def get_rag_retriever(country="Kenya"):
    """
    Load retriever for guideline documents.
    """
    # Global retriever (load global index)
    global_index_path = f"data/{country}/processed/lp/indices/Global"   # adjust to your real folder
    storage_context_arv = StorageContext.from_defaults(persist_dir=global_index_path)
    index_arv = load_index_from_storage(storage_context_arv)
    retriever = VectorIndexRetriever(index=index_arv, similarity_top_k=2)
    return retriever


# ------------------------
# Main function
# ------------------------
def sql_chain(query: str, llm, pk_hash: str, country:str, search_guidelines = False, filter_tables = False) -> dict:
    """
    Retrieves patient data from SQL and summarizes it along with RAG guideline context.
    """
    summarizer_llm = get_summarizer_llm()
    
    if not pk_hash:
        raise ValueError("pk_hash is required in state for SQL queries.")

    conn = sqlite3.connect("data/processed/site_database.sqlite")
    table_mappings = json.load(open("data/processed/table_mappings.json"))
    decoder = build_decoder_dict("data/processed/Site tables decoding map.csv")

    # if filter_tables is True, the use LLM to select relevant tables
    if filter_tables:
        relevant_tables = get_relevant_tables(query, table_descriptions, llm) 
        filtered_tables = {k: v for k, v in table_descriptions.items() if k in relevant_tables}
    else:
        filtered_tables = table_descriptions

    patient_data = patient_data_to_text(pk_hash, conn, table_mappings, decoder, tables_to_include=filtered_tables)

    # if search_guidelines if True, then do RAG
    use_guidelines = False
    if search_guidelines:
        # Decide whether guidelines are needed
        use_guidelines = check_guideline_needed(query)
        print(f"Use guidelines: {use_guidelines}")

    # Only invoke RAG if needed
    if use_guidelines:
        retriever = get_rag_retriever(country=country)
        sources = retriever.retrieve(query)
        summarization_prompt = build_summarization_prompt(query, sources)
        guidelines_summary = summarizer_llm.invoke(summarization_prompt).content
    else:
        guidelines_summary = None

    print(patient_data)

    # --- Final prompt ---
    prompt = (
        "You are a clinical assistant. Given the user question, clinical guideline context, "
        "and summarized patient data below, answer the question accurately and concisely. "
        "Only use the provided data; do not guess or hallucinate. "
        "If essential patient information is missing, explain what is missing instead of guessing.\n\n"
        "Provide a bullet-point summary with no more than 5 bullets.\n\n"
        f"Question: {query}\n\n"
        f"Guideline Context: {guidelines_summary}\n\n"
        f"Patient Data:\n{patient_data}\n\n"
    )

    response = llm.invoke(prompt)
    return {"answer": response.content, "last_tool": "sql_chain"}


# ------------------------
# Standalone wrapper
# ------------------------
def run_sql_standalone(query: str, pk_hash: str, country:str, search_guidelines=True, filter_tables = False) -> dict:
    llm = get_main_llm()
    query_redacted = detect_and_redact_phi(query)["redacted_text"]
    return sql_chain(query=query_redacted,
                      llm=llm, 
                      pk_hash=pk_hash, 
                      country=country,
                      search_guidelines=search_guidelines,
                      filter_tables=filter_tables)


# ------------------------
# CLI entrypoint
# ------------------------
if __name__ == "__main__":
    q = input("Enter your query: ")
    pk = input("Enter your patient PK hash: ")
    sg = input("Search guidelines? (y/n): ").strip().lower() == 'y'
    ft = input("Filter tables? (y/n): ").strip().lower() == 'y'
    c = input("Enter country (default: Kenya): ").strip() or "Kenya"
    result = run_sql_standalone(q, pk, country=c, search_guidelines=sg, filter_tables=ft)
    print("\n--- SQL Tool Result ---")
    print(result["answer"])
