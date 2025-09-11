import json
from datetime import datetime

def format_value(name, value, vtype, display_name):
    """Format a single variable value based on type and known units."""
    if value in [None, "", []]:
        return None

    # Numeric types (add units if known)
    if vtype == "Numeric":
        units = {
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
        }

        unit = units.get(name, "")
        return f"{display_name}: {value}{' ' + unit if unit else ''}"

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


def prep_triage_case(case_json: dict, variable_metadata: list[dict]) -> str:
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


triage_metadata = [
    {"Display Name": "Sex", "Variable Name": "sex", "values": "Categorical (Radio)", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Age", "Variable Name": "age", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "patient having complaint today", "Variable Name": "complaint_today", "values": "Categorical (Radio)", "Conditions": "", "Multiple Entries": "Multiple"},
    {"Display Name": "Complaint", "Variable Name": "complaint", "values": "Categorical", "Conditions": "if complaint_today = Yes", "Multiple Entries": ""},
    {"Display Name": "Duration in Days", "Variable Name": "duration", "values": "Numeric", "Conditions": "if complaint_today = Yes", "Multiple Entries": ""},
    {"Display Name": "Onset status", "Variable Name": "onset_status", "values": "Categorical", "Conditions": "if complaint_today = Yes", "Multiple Entries": ""},
    {"Display Name": "Patient Mobility Assessment", "Variable Name": "patient_mobility", "values": "Categorical (Radio)", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Any injury to the patient in the last 48 hours", "Variable Name": "injury", "values": "Categorical (Radio)", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Patient Level of Consciousness", "Variable Name": "consciousness", "values": "Categorical (Radio)", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Temperature in Celsius", "Variable Name": "temperature", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Pulse Rate", "Variable Name": "pulse_rate", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Systolic", "Variable Name": "systolic", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Diastolic", "Variable Name": "diastolic", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Respiratory Rate", "Variable Name": "respiratory_rate", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Oxygen Saturation", "Variable Name": "oxygen_saturation", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Weight in KG", "Variable Name": "weight", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Height in CM", "Variable Name": "height", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "BMI", "Variable Name": "bmi", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "MUAC", "Variable Name": "muac", "values": "Numeric", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Nutritional Status", "Variable Name": "nutrional_status", "values": "Categorical", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "LMP (Last Menstrual Period)", "Variable Name": "lmp", "values": "Date", "Conditions": "", "Multiple Entries": ""},
    {"Display Name": "Triage Notes", "Variable Name": "triage_notes", "values": "Text", "Conditions": "", "Multiple Entries": ""}
]

case_triage = '''
{
  "complaint_today": "Yes",
  "complaints": [
    {"complaint": "Vaginal bleeding", "duration": 3, "onset_status": "Sudden"},
    {"complaint": "Pain during intercourse", "duration": 7, "onset_status": "Gradual"}
  ],
  "sex": "Female",
  "age": 29,
  "patient_mobility": "Ambulatory",
  "injury": "No",
  "consciousness": "Alert",
  "temperature": 38.5,
  "pulse_rate": 88,
  "systolic": 120,
  "diastolic": 80,
  "respiratory_rate": 20,
  "oxygen_saturation": 95,
  "weight": 65,
  "height": 172,
  "bmi": 22.0,
  "muac": 12,
  "nutrional_status": "Moderate malnutrition",
  "lmp": "2025-08-15",
  "triage_notes": "Patient reporting pain during intercourse, family history of cervical cancer."
}
'''


if __name__ == "__main__":
    print(prep_triage_case(case_triage, triage_metadata))