import json
from datetime import datetime

def format_value(name, value, vtype, display_name):
    """Format a single variable value based on type and known units."""
    if value in [None, "", []]:
        return None

    # Numeric types (add units if known)
    if vtype == "Numeric":
        units = {
            "age": "years",
            "temperature": "°C",
            "pulse_rate": "bpm",
            "systolic": "mmHg",
            "diastolic": "mmHg",
            "respiratory_rate": "breaths/min",
            "oxygen_saturation": "%",
            "weight": "kg",
            "height": "cm",
            "bmi": "kg/m²",
            "muac": "cm",
            "menarche_age": "years",
            "menses_duration": "days",
        }

        unit = units.get(name, "")
        return f"{display_name}: {value}{' ' + unit if unit else ''}"
    
        # Checkbox: may be string, list, or missing
    if vtype == "checkbox":
        # Normalize to list
        if isinstance(value, str):
            value = [value]
        elif not isinstance(value, list):
            value = [str(value)]

        if not value:
            return None
        elif len(value) == 1:
            return value[0]
        else:
            return ", ".join(value)  # inline; could also use bullets

    # Dates
    elif vtype == "Date":
        try:
            val = datetime.fromisoformat(value).strftime("%d-%b-%Y")
            return f"{display_name}: {val}"
        except Exception:
            return f"{display_name}: {value}"

    # Text or categorical
    else:
        return f"{display_name}: {value}"


def prep_case(case_json: dict, variable_metadata: list[dict]) -> str:
    """
    Convert a case JSON object into a readable string for LLMs,
    applying metadata rules (conditions, display names, units, multiples).
    """
    summary_parts = []

    # --- Special handling for complaint block ---
    if case_json.get("complaint_today") == "Yes":
        complaints = case_json.get("complaints", [])
        if complaints:
            for idx, comp in enumerate(complaints, 1):
                c = comp.get("complaint")
                d = comp.get("duration")
                o = comp.get("onset_status")

                line = f"Complaint {idx}: {c}"
                if d is not None:
                    line += f" (Duration: {d} days)"
                if o:
                    line += f", Onset: {o}"

                summary_parts.append(line)
    else:
        if "complaint_today" in case_json:
            summary_parts.append("Patient having complaint today: No")

    # --- Special handling for adverse drug reaction block ---
    if case_json.get("adverse_drug_reactions") == "Yes":
        reactions = case_json.get("drug_reactions", [])
        if reactions:
            for idx, reaction in enumerate(reactions, 1):
                m = reaction.get("reaction_medicine")
                t = reaction.get("reaction_type")
                s = reaction.get("reaction_severity")
                d = reaction.get("reaction_date")
                a = reaction.get("reaction_action")

                line = f"Adverse Drug Reaction {idx}: {reaction.get('reaction')}"
                if m is not None:
                    line += f" (Medicine: {m})"
                if t:
                    line += f", Type: {t}"
                if s:
                    line += f", Severity: {s}"
                if d:
                    line += f", Date: {d}"
                if a:
                    line += f", Action: {a}"

                summary_parts.append(line)
    else:
        if "adverse_drug_reactions" in case_json:
            summary_parts.append("Patient having adverse drug reactions: No")

    # --- Loop through the rest of metadata ---
    for var in variable_metadata:
        name = var["Variable Name"]
        display_name = var["Display Name"]
        vtype = var.get("values", "")  # type info is in 'values' column

        # Skip complaint-related fields (already handled above)
        if name in ["complaint_today", "complaint", "duration", "onset_status", "complaints"]:
            continue

        if name not in case_json:
            continue

        value = case_json[name]
        formatted = format_value(name, value, vtype, display_name)
        if formatted:
            summary_parts.append(formatted)

    return "\n".join(summary_parts)


if __name__ == "__main__":
    from ai_tools.schemas import case_metadata, case_triage
    print(prep_case(case_triage, case_metadata))