# sql_tool.py
import os
import sqlite3
import pandas as pd
from functools import lru_cache
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from langchain_openai import ChatOpenAI

from ai_tools.phi_filter import detect_and_redact_phi
from ai_tools.helpers import describe_relative_date

# --- Load env vars ---
if os.path.exists("config.env"):
    load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")


# ------------------------
# Utilities
# ------------------------
def safe(val):
    if pd.isnull(val) or val in ("", "NULL"):
        return "missing"
    return val

def extract_year(date_str):
    if pd.isnull(date_str) or date_str in ("", "NULL"):
        return "missing"
    try:
        return pd.to_datetime(date_str).year
    except (ValueError, TypeError):
        return "invalid date"


# ------------------------
# Lazy loaders
# ------------------------
@lru_cache()
def get_summarizer_llm():
    return ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0125")

@lru_cache()
def get_main_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

@lru_cache()
def get_rag_retriever():
    """
    Load retriever for guideline documents.
    """
    # Global retriever (load global index)
    global_index_path = "data/processed/lp/indices/Global"   # adjust to your real folder
    storage_context_arv = StorageContext.from_defaults(persist_dir=global_index_path)
    index_arv = load_index_from_storage(storage_context_arv)
    retriever = VectorIndexRetriever(index=index_arv, similarity_top_k=3)
    return retriever


# ------------------------
# Main function
# ------------------------
def sql_chain(query: str, llm, global_retriever, pk_hash: str) -> dict:
    """
    Retrieves patient data from SQL and summarizes it along with RAG guideline context.
    """
    summarizer_llm = get_summarizer_llm()

    # Retrieve guideline context
    sources = global_retriever.retrieve(query)
    summarization_prompt = (
        "You're a clinical assistant helping a provider answer a question using HIV/AIDS guidelines.\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed summary of the most relevant points to the user question from the following source texts.\n\n"
        f"{sources}"
    )
    guidelines_summary = summarizer_llm.invoke(summarization_prompt).content

    if not pk_hash:
        raise ValueError("pk_hash is required in state for SQL queries.")

    conn = sqlite3.connect("data/processed/patient_demonstration.sqlite")
    cursor = conn.cursor()

    # --- Visits ---
    cursor.execute("SELECT * FROM clinical_visits WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash})
    visits_data = pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description])

    def summarize_visits(df):
        if df.empty:
            return "No clinical visit data available."
        df = df.copy()
        df["VisitDate"] = pd.to_datetime(df["VisitDate"], errors="coerce")
        df["NextAppointmentDate"] = pd.to_datetime(df["NextAppointmentDate"], errors="coerce")

        summaries = []
        ordinal_map = {1: "First", 2: "Second", 3: "Third"}
        for idx, (_, row) in enumerate(df.sort_values("VisitDate", ascending=False).head(3).iterrows(), start=1):
            ordinal = ordinal_map.get(idx, f"{idx}th")
            summaries.append(
                f"{ordinal} most recent clinical visit, {describe_relative_date(row['VisitDate'])}: "
                f"WHO Stage {safe(row['WHOStage'])}, Weight {safe(row['Weight'])}kg, "
                f"NextAppointmentDate {describe_relative_date(safe(row['NextAppointmentDate']))}, "
                f"VisitType {safe(row['VisitType'])}, VisitBy {safe(row['VisitBy'])}, "
                f"Pregnant {safe(row['Pregnant'])}, Breastfeeding {safe(row['Breastfeeding'])}, "
                f"StabilityAssessment {safe(row['StabilityAssessment'])}, DifferentiatedCare {safe(row['DifferentiatedCare'])}, "
                f"Height {safe(row['Height'])}cm, Adherence {safe(row['Adherence'])}, BP {safe(row['BP'])}, "
                f"OI {safe(row['OI'])}, CurrentRegimen {safe(row['CurrentRegimen'])}"
            )
        return "\n".join(summaries)

    visits_summary = summarize_visits(visits_data)

    # --- Pharmacy ---
    cursor.execute("SELECT * FROM pharmacy WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash})
    pharmacy_data = pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description])

    def summarize_pharmacy(df):
        if df.empty:
            return "No pharmacy data available."
        df = df.copy()
        df["DispenseDate"] = pd.to_datetime(df["DispenseDate"], errors="coerce")
        df["ExpectedReturn"] = pd.to_datetime(df["ExpectedReturn"], errors="coerce")

        summaries = []
        ordinal_map = {1: "First", 2: "Second", 3: "Third"}
        for idx, (_, row) in enumerate(df.sort_values("DispenseDate", ascending=False).head(3).iterrows(), start=1):
            ordinal = ordinal_map.get(idx, f"{idx}th")
            summaries.append(
                f"{ordinal} most recent pharmacy visit, {describe_relative_date(row['DispenseDate'])}, "
                f"ExpectedReturn {describe_relative_date(row['ExpectedReturn'])}, Drug {safe(row['Drug'])}, "
                f"Duration {safe(row['Duration'])}, TreatmentType {safe(row['TreatmentType'])}, "
                f"RegimenLine {safe(row['RegimenLine'])}, "
                f"RegimenChangedSwitched {safe(row['RegimenChangedSwitched'])}, "
                f"RegimenChangeSwitchedReason {safe(row['RegimenChangeSwitchedReason'])}"
            )
        return "\n".join(summaries)

    pharmacy_summary = summarize_pharmacy(pharmacy_data)

    # --- Lab ---
    cursor.execute("SELECT * FROM lab WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash})
    lab_data = pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description])

    def summarize_lab(df):
        if df.empty:
            return "No lab data available."
        df = df.copy()
        df["OrderedbyDate"] = pd.to_datetime(df["OrderedbyDate"], errors="coerce")

        summaries = []
        ordinal_map = {1: "First", 2: "Second", 3: "Third"}
        for idx, (_, row) in enumerate(df.sort_values("OrderedbyDate", ascending=False).head(3).iterrows(), start=1):
            summaries.append(
                f"{ordinal_map.get(idx, idx)} most recent lab test, {describe_relative_date(row['OrderedbyDate'])}. "
                f"TestName {safe(row['TestName'])}, TestResult {safe(row['TestResult'])}"
            )
        return "\n".join(summaries)

    lab_summary = summarize_lab(lab_data)

    # --- Demographics ---
    cursor.execute("SELECT * FROM demographics WHERE PatientPKHash = :pk_hash", {"pk_hash": pk_hash})
    demographic_data = pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description])

    def summarize_demographics(df):
        if df.empty:
            return "No demographic data available."

        def calculate_age(dob):
            if pd.isnull(dob) or dob in ("", "NULL"):
                return "missing"
            try:
                dob = pd.to_datetime(dob)
                today = pd.to_datetime("today")
                return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            except (ValueError, TypeError):
                return "invalid date"

        row = df.iloc[0]
        return (
            f"Sex: {safe(row['Sex'])}\n"
            f"MaritalStatus: {safe(row['MaritalStatus'])}\n"
            f"EducationLevel: {safe(row['EducationLevel'])}\n"
            f"Occupation: {safe(row['Occupation'])}\n"
            f"OnIPT: {safe(row['OnIPT'])}\n"
            f"ARTOutcomeDescription: {safe(row['ARTOutcomeDescription'])}\n"
            f"StartARTDate: {describe_relative_date(pd.to_datetime(row['StartARTDate'], errors='coerce'))}\n"
            f"Age: {calculate_age(safe(row['DOB']))}"
        )

    demographic_summary = summarize_demographics(demographic_data)
    conn.close()

    # --- Final prompt ---
    prompt = (
        "You are a clinical assistant. Given the user question, clinical guideline context, "
        "and summarized patient data below, answer the question accurately and concisely. "
        "Only use the provided data; do not guess or hallucinate. "
        "If essential patient information is missing, explain what is missing instead of guessing.\n\n"
        f"Question: {query}\n\n"
        f"Guideline Context: {guidelines_summary}\n\n"
        f"Clinical Visits Summary:\n{visits_summary}\n\n"
        f"Pharmacy Summary:\n{pharmacy_summary}\n\n"
        f"Lab Summary:\n{lab_summary}\n\n"
        f"Demographic Summary:\n{demographic_summary}\n"
    )

    response = llm.invoke(prompt)
    return {"answer": response.content, "last_tool": "sql_chain"}


# ------------------------
# Standalone wrapper
# ------------------------
def run_sql_standalone(query: str, pk_hash: str):
    llm = get_main_llm()
    retriever = get_rag_retriever()
    query_redacted = detect_and_redact_phi(query)["redacted_text"]
    return sql_chain(query=query_redacted, llm=llm, global_retriever=retriever, pk_hash=pk_hash)


# ------------------------
# CLI entrypoint
# ------------------------
if __name__ == "__main__":
    q = input("Enter your query: ")
    pk = input("Enter your patient PK hash: ")
    result = run_sql_standalone(q, pk)
    print("\n--- SQL Tool Result ---")
    print(result["answer"])
