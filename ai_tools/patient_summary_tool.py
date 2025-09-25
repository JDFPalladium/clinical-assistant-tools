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
    if not topic_dict:
        return {"answer": "No patient data found."}
    
    sections = "\n\n".join(
    f"{topic}\n\n{content}" for topic, content in topic_dict.items()



)
    # --- Final prompt ---
    prompt = f"""
You are a clinical assistant reviewing structured patient data.
Your task is to summarize only clinical concerns requiring follow-up or action 
(exclude routine facts).

Output as a bullet list: 
• [Priority: HIGH/MEDIUM/LOW] [Theme] - [Actionable Insight] (≤2 lines).

General Rules:
    - HIGH → Life-threatening, urgent, or major risk (e.g., repeated viral non-suppression, severe adverse drug reaction).
    - MEDIUM → Moderate but manageable concern requiring follow-up (e.g., missed visits, pregnancy without recent VL).
    - LOW → Mild concern, monitor only (e.g., single missed dose, mild side effects).
    - Report patterns/trends, not isolated normal values (e.g., “repeated missed visits” not “missed once”).
    - Only include concerning or abnormal findings.
    - If there is no major concern, return: • No impending risks detected.

Here is the structured patient data grouped by topic:
{sections}

Now generate the summary following these topic-specific rules:
1. Medication & Treatment

Items of concern include:
- Any history of adverse drug reactions
- Medication non-adherence and challenges with adherence
- History of missed appointments
- Any history of treatment failure, regimen changes, or drug resistance
- Regimen changes due to confirmed treatment failure.

Example Patient Output:
• HIGH Medication & Treatment - 2 unsuppressed VL results with regimen change → risk of confirmed treatment failure.
• MEDIUM Medication & Treatment - Missed last 3 clinic visits  → risk of ART interruption.
• LOW Medication & Treatment - Reported mild dizziness on ART but continues treatment.



2. Laboratory & Diagnostics

Items of concern include:
- Any history of HIV viral non-suppression, low CD4 counts, or other abnormal lab results
- History of unsuppressed or rising viral load
- Unsuppressed viral load with no adherence counselling
- Declining or critically low CD4 counts (below 200 cells/mm³).

Example Patient Output:
• HIGH Laboratory & Diagnostics- VL rising across 3 tests → uncontrolled HIV.
• MEDIUM Laboratory & Diagnostics - CD4 dropped from 450 → 300 in past year.
• LOW Laboratory & Diagnostics - Slightly elevated liver enzymes, no ART change required.



3. Clinical Events

Items of concern include:
- Any history of complaints and diagnoses
- Any history of hospitalizations, surgeries, or other significant clinical events.

Example Patient Output:
• HIGH Clinical Events – Hospitalized with cryptococcal meningitis last month.
• MEDIUM Clinical Events – 2 admissions in past year for pneumonia.
• LOW Clinical Events – Occasional mild cough, resolved without intervention.



4. Psychosocial & Behavioral Risks

Items of concern include:
- History of alcohol abuse linked to missed ART doses.
- Documented depression with no follow-up referral.
- Skipped visits due to fear of disclosure.
- Faith healing led to missed doses.

Example Patient Output:
• HIGH Psychosocial & Behavioral Risks - Alcohol abuse with repeated ART non-adherence.
• MEDIUM Psychosocial & Behavioral Risks - Documented depression, no mental health referral.
• LOW Psychosocial & Behavioral Risks - Skipped 1 visit due to stigma fears.



5. **Reproductive & Sexual Health**

Items of concern include:
- Current pregnancy or desire for pregnancy without recent VL or poor ART adherence.
- Pregnancy with poor ANC follow-up or complications (Gravida / Parity / Miscarriages).
- History of PMTCT default during breastfeeding.
- Breastfeeding while VL unsuppressed or unknown.

Example Patient Output:
• HIGH Reproductive & Sexual Health - Currently pregnant, last VL unknown → PMTCT risk.
• MEDIUM Reproductive & Sexual Health - History of 2 miscarriages, poor ANC attendance.
• LOW Reproductive & Sexual Health - Breastfeeding, VL suppressed but missed 1 ANC visit.



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