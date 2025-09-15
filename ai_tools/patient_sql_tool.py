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
from ai_tools.helpers import describe_relative_date
from ai_tools.schemas import table_descriptions

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

def patient_data_to_text(patient_id, conn, table_mappings, tables_to_include=None):
    """
    Pull all available data for one patient across multiple tables,
    filter/rename columns using the table_mappings, and return as text.

    Args:
        patient_id (str): The patient identifier to query.
        conn: Database connection object (e.g., SQLAlchemy or sqlite3).
        table_mappings (dict): Metadata dict with table and column mappings.
    
    Returns:
        str: JSON string with patient data.
    """

    patient_dict = {}

    if tables_to_include is not None:
        table_mappings = {k: v for k, v in table_mappings.items() if k in tables_to_include}

    for table_name, mapping in table_mappings.items():
        columns = list(mapping["columns"].keys())
        query = f"SELECT {', '.join(columns)} FROM {table_name} WHERE patient_id = ?"
        df = pd.read_sql(query, conn, params=(patient_id,))

        if df.empty:
            continue  # skip tables with no rows

        # Build rename mapping: {column_name: display_name}
        rename_map = {col: col_info["display_name"] for col, col_info in mapping["columns"].items()}
        df = df.rename(columns=rename_map)

        # drop columns that are entirely null
        df = df.dropna(axis=1, how="all")

        for col in df.columns:
            if "date" in col.lower():
                df[col] = df[col].apply(lambda x: describe_relative_date(pd.to_datetime(x)) if pd.notnull(x) else x)

        # convert to dict format
        records = df.to_dict(orient="records")

        # single row table → dict
        display_name = mapping["display_name"]
        if len(records) == 1:
            patient_dict[display_name] = records[0]
        else:
            patient_dict[display_name] = records

            # Convert patient_dict to compact text
        lines = []
        for table_name, content in patient_dict.items():
            lines.append(f"{table_name}:")
            if isinstance(content, dict):
                for k, v in content.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(content, list):
                for record in content:
                    record_lines = [f"{k}: {v}" for k, v in record.items()]
                    lines.append("  - " + ", ".join(record_lines))
            lines.append("")  # blank line between tables

        return "\n".join(lines)


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
def sql_chain(query: str, llm, pk_hash: str) -> dict:
    """
    Retrieves patient data from SQL and summarizes it along with RAG guideline context.
    """
    summarizer_llm = get_summarizer_llm()
    
    if not pk_hash:
        raise ValueError("pk_hash is required in state for SQL queries.")

    conn = sqlite3.connect("data/processed/site_database.sqlite")
    table_mappings = json.load(open("data/processed/table_mappings.json"))

    relevant_tables = get_relevant_tables(query, table_descriptions, llm) 
    filtered_tables = {k: v for k, v in table_descriptions.items() if k in relevant_tables}
    patient_json = patient_data_to_text(pk_hash, conn, table_mappings, tables_to_include=filtered_tables)

    # Decide whether guidelines are needed
    use_guidelines = check_guideline_needed(query)
    print(f"Use guidelines: {use_guidelines}")

    # Only invoke RAG if needed
    if use_guidelines:
        retriever = get_rag_retriever()
        sources = retriever.retrieve(query)
        summarization_prompt = build_summarization_prompt(query, sources)
        guidelines_summary = summarizer_llm.invoke(summarization_prompt).content
    else:
        guidelines_summary = None

    # --- Final prompt ---
    prompt = (
        "You are a clinical assistant. Given the user question, clinical guideline context, "
        "and summarized patient data below, answer the question accurately and concisely. "
        "Only use the provided data; do not guess or hallucinate. "
        "If essential patient information is missing, explain what is missing instead of guessing.\n\n"
        f"Question: {query}\n\n"
        f"Guideline Context: {guidelines_summary}\n\n"
        f"Patient Data (JSON):\n{patient_json}\n\n"
    )
    print(prompt)
    response = llm.invoke(prompt)
    return {"answer": response.content, "last_tool": "sql_chain"}


# ------------------------
# Standalone wrapper
# ------------------------
def run_sql_standalone(query: str, pk_hash: str):
    llm = get_main_llm()
    query_redacted = detect_and_redact_phi(query)["redacted_text"]
    return sql_chain(query=query_redacted, llm=llm, pk_hash=pk_hash)


# ------------------------
# CLI entrypoint
# ------------------------
if __name__ == "__main__":
    q = input("Enter your query: ")
    pk = input("Enter your patient PK hash: ")
    result = run_sql_standalone(q, pk)
    print("\n--- SQL Tool Result ---")
    print(result["answer"])
