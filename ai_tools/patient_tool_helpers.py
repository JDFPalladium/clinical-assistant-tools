from ai_tools.helpers import describe_relative_date
import pandas as pd

def build_decoder_dict(decoder_csv):
    """
    Convert decoder CSV into nested dict for fast lookup:
    { (table, column): { code: decoded_value } }
    """
    decoder_df = pd.read_csv(decoder_csv)
    decoder_map = {}
    for (table, col), group in decoder_df.groupby(["table_name", "column_name"]):
        decoder_map[(table, col)] = dict(zip(group["code"], group["value"]))
    return decoder_map

def patient_data_to_text(patient_id, conn, table_mappings, decoder, tables_to_include=None, num_obs=3):
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
    topic_dict = {}

    # if tables_to_include is given, filter table_mappings to the tables selected by LLM for relevance
    if tables_to_include is not None:
        table_mappings = {k: v for k, v in table_mappings.items() if k in tables_to_include}

    # now, iterate through tables, select columns, decode values, and add to patient_dict
    for table_name, mapping in table_mappings.items():
        columns = list(mapping["columns"].keys())
        # take top 3 rows only to limit size
        query = f"SELECT {', '.join(columns)} FROM {table_name} WHERE patient_id = ? LIMIT {num_obs}"
        df = pd.read_sql(query, conn, params=(patient_id,))

        if df.empty:
            continue  # skip tables with no rows

        # Decode coded values safely
        for col in df.columns:
            key = (table_name, col)
            if key in decoder:
                def safe_decode(x):
                    if pd.isna(x):
                        return x

                    # Convert float that is an integer (e.g., 165610.0) to int
                    if isinstance(x, float) and x.is_integer():
                        x = int(x)

                    # Look up in decoder using string key
                    return decoder.get(key, {}).get(str(x), x)

                df[col] = df[col].apply(safe_decode)

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

        # --- NEW: group by column topic ---
        for col, col_info in mapping["columns"].items():
            topic = col_info.get("topic", "Other")
            display_name = col_info["display_name"]
            if display_name in df.columns:
                values = df[display_name].dropna().tolist()
                for v in values:
                    topic_dict.setdefault(topic, []).append(f"{display_name}: {v}")


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

    compact_text = "\n".join(lines)

    return compact_text, topic_dict