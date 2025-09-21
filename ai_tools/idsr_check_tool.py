# my_tools/idsr_tool.py
import os
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime
from typing import List
from unittest import case

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
# from ai_tools.phi_filter import detect_and_redact_phi
from ai_tools.prep_case import prep_case
from ai_tools.schemas import case_metadata

if os.path.exists("config.env"):
    load_dotenv("config.env")
os.environ.get("OPENAI_API_KEY")

# -------------------------------
# Lazy-loaded globals for standalone usage
# -------------------------------
_vectorstore = None
_tagged_documents = None
_keywords = None
_llm_langchain = None
_keyword_weights = None

def _lazy_load():
    global _vectorstore, _tagged_documents, _keywords, _llm_langchain, _keyword_weights
    
    if _vectorstore is None:
        # Load embeddings & vectorstore
        _vectorstore = FAISS.load_local(
            "data/processed/disease_vectorstore",
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )

        # Load tagged documents
        with open("data/processed/tagged_documents.json", "r", encoding="utf-8") as f:
            doc_dicts = json.load(f)
        _tagged_documents = [Document(**d) for d in doc_dicts]

        # Load keywords
        with open("data/processed/idsr_keywords.txt", "r", encoding="utf-8") as f:
            _keywords = [line.strip() for line in f if line.strip()]

        # Compute keyword weights
        keyword_doc_counts = Counter()
        total_docs = len(_tagged_documents)
        for tagged_doc in _tagged_documents:
            seen = set(tagged_doc.metadata.get("matched_keywords", []))
            for kw in seen:
                keyword_doc_counts[kw] += 1
        _keyword_weights = {
            kw: math.log(total_docs / (1 + count)) for kw, count in keyword_doc_counts.items()
        }

        # Load LLM
        _llm_langchain = ChatOpenAI(temperature=0.0, model="gpt-4o")

# -------------------------------
# Keyword scoring and extraction
# -------------------------------
class KeywordsOutput(BaseModel):
    keywords: List[str] = Field(description="List of relevant keywords extracted from the query")

def score_doc(doc_to_score, matched_keywords, keyword_weights):
    doc_keywords = set(doc_to_score.metadata.get("matched_keywords", []))
    overlap = doc_keywords & set(matched_keywords)
    return sum(keyword_weights.get(kw, 0) for kw in overlap)

def extract_keywords_with_gpt(query: str, llm, known_keywords: List[str]) -> List[str]:
    parser = PydanticOutputParser(pydantic_object=KeywordsOutput)
    prompt = ChatPromptTemplate.from_template(
        """
You are helping identify relevant medical concepts. 
Given this query: "{query}"

Select the most relevant 3-5 keywords from this list:
{keyword_list}

Only choose keywords that are **clearly supported by the text**. Do not include keywords based on loose associations or general knowledge.  
If a keyword seems possible but the text does not provide evidence, do NOT include it.

Return the matching keywords as a JSON object with a single key "keywords" whose value is a list of strings.

{format_instructions}
"""
    )
    chain = prompt | llm | parser
    output = chain.invoke({
        "query": query,
        "keyword_list": ", ".join(known_keywords),
        "format_instructions": parser.get_format_instructions(),
    })
    return output.keywords

def build_semantic_query(case: dict) -> str:
    complaints_list = [c["complaint"] for c in case.get("complaints", [])]
    complaints = ", ".join(complaints_list)
    notes = case.get("triage_notes", "")
    return f"Complaints: {complaints}. Notes: {notes}"

def hybrid_search_with_query_keywords(query, vstore, documents, keyword_list, llm, keyword_weights, top_k=3):
    semantic_hits_with_scores = vstore.similarity_search_with_score(query, k=top_k)
    semantic_hits = [doc for doc, score in semantic_hits_with_scores if score >= 0.65]
    # print the names of the docs in the semantic hits
    for doc in semantic_hits:
        print(f"Semantic match: {doc.metadata.get('disease_name', 'Unknown Disease')}")
    matched_keywords = extract_keywords_with_gpt(query, llm, keyword_list)

    keyword_hits = [
        doc for doc in documents
        if any(kw1 == kw2 for kw1 in doc.metadata.get("matched_keywords", []) for kw2 in matched_keywords)
    ]

    scored_docs = [(doc, score_doc(doc, matched_keywords, keyword_weights)) for doc in keyword_hits]
    ranked_docs = sorted(scored_docs, key=lambda x: -x[1])
    top_docs = [doc for doc, score in ranked_docs if score > 1.5]
    top_3_docs = top_docs[:3]
    # print the names of the docs in the top 3 keyword hits
    for doc in top_3_docs:
        print(f"Keyword match: {doc.metadata.get('disease_name', 'Unknown Disease')}")

    merged = {doc.page_content: doc for doc in semantic_hits + top_3_docs}
    # print doc names
    for doc in merged.values():
        print(f"Matched document: {doc.metadata.get('disease_name', 'Unknown Disease')}")
    return list(merged.values())

# -------------------------------
# Core tool function
# -------------------------------
def idsr_check(query, llm=None, sitecode=None):
    _lazy_load()
    llm = llm or _llm_langchain

    if isinstance(query, dict):
        query_dict = query
    elif isinstance(query, str):
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}\nInput was: {query[:100]}")
    else:
        raise TypeError(f"Query must be dict or JSON string, got {type(query)}")

    query = prep_case(query_dict, case_metadata)
    sem_query = build_semantic_query(query_dict)

    results = hybrid_search_with_query_keywords(
        sem_query, _vectorstore, _tagged_documents, _keywords, llm, _keyword_weights
    )

    # Location & epidemic info (SQLite)
    conn = sqlite3.connect("data/processed/location_data.sqlite")
    county_name, rainy_season, county_info, epidemic_info = "Unknown", "Unknown", [], []
    if sitecode:
        cursor = conn.cursor()
        cursor.execute("SELECT County FROM sitecode_county_xwalk WHERE Code = ?", (sitecode,))
        county = cursor.fetchone()
        if county:
            county_name = county[0]
            # Retrieve county disease info
            placeholders = ",".join("?" * len(results))
            disease_names = [doc.metadata.get("disease_name") for doc in results]
            placeholders = ",".join("?" * len(disease_names))
            cursor.execute(f"SELECT County, Disease, Prevalence, Notes FROM county_disease_info WHERE County = ? AND Disease IN ({placeholders})", (county_name, *disease_names))
            county_info = cursor.fetchall()
            # Epidemic info
            cursor.execute(f"SELECT Disease, EpidemicInfo FROM who_bulletin WHERE Disease IN ({placeholders})", disease_names)
            epidemic_info = cursor.fetchall()
            # Rainy season
            current_month = datetime.now().strftime("%B")
            cursor.execute("SELECT RainySeason FROM county_rainy_seasons WHERE County = ? and Month = ?", (county_name, current_month))
            rainy_row = cursor.fetchone()
            rainy_season = rainy_row[0] if rainy_row else "Unknown"
    conn.close()

    # Build prompt & invoke LLM
    disease_definitions = "\n\n".join([
        f"### Disease: {doc.metadata.get('disease_name', 'Unknown Disease')}:\n{doc.page_content}"
        for doc in results
    ])
    prompt_text = f"""
    Role & Context
    You are a medical assistant analyzing a clinical case in Kenya. You have:
    
    Disease definitions, County-level prevalence, seasonality, epidemic alerts, and rainy season status

    Instructions
    Compare the case to each disease definition, considering local prevalence, seasonality, and epidemic alerts.

    For each disease, classify as one of:
    - HIGH: strong alignment with the case and context
    - MEDIUM: possible but not strongly supported
    - LOW: unlikely

    Only include diseases that are HIGH. Do not include diseases that are MEDIUM or LOW. If no diseases are HIGH, return:
    NONE

    Keep reasoning to one concise line per HIGH disease.

    Clarifying Questions: include 2-3 if there are plausible matches; otherwise output NONE.
    Recommendation: single line if there are plausible matches; otherwise NONE.

## Case:
{query}

## Diseases:
{disease_definitions}

## Locational context:
In {county_name}, the current rainy season status is {rainy_season}.
County disease info: {county_info}
Epidemic info: {epidemic_info}
"""
    print(prompt_text)
    llm_response = llm.invoke(prompt_text)
    answer_text = llm_response.content.strip() if llm_response else "No relevant disease information found."

    no_relevant_matches = "NONE" in answer_text

    return {
        "answer": answer_text,
        "last_tool": "idsr_check",
        "possible_match_flag": no_relevant_matches,
        "context": results
    }

# -------------------------------
# Standalone wrapper
# -------------------------------
def idsr_check_standalone(query: str, sitecode=None):
    _lazy_load()
    return idsr_check(query, llm=_llm_langchain, sitecode=sitecode)

# -------------------------------
# CLI testing
# -------------------------------
if __name__ == "__main__":
    q = input("Enter your query: ")
    sc = input("Enter site code (optional): ")
    result = idsr_check_standalone(q, sc or None)
    print("\n=== Answer ===")
    print(result["answer"])
