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


case_metadata = [
    {"Display Name": "Sex", "Variable Name": "sex", "values": "Categorical"},
    {"Display Name": "Age", "Variable Name": "age", "values": "Numeric"},
    {"Display Name": "patient having complaint today", "Variable Name": "complaint_today", "values": "Categorical"},
    {"Display Name": "Complaint", "Variable Name": "complaint", "values": "Categorical"},
    {"Display Name": "Duration in Days", "Variable Name": "duration", "values": "Numeric"},
    {"Display Name": "Onset status", "Variable Name": "onset_status", "values": "Categorical"},
    {"Display Name": "Patient Mobility Assessment", "Variable Name": "patient_mobility", "values": "Categorical"},
    {"Display Name": "Any injury to the patient in the last 48 hours", "Variable Name": "injury", "values": "Categorical"},
    {"Display Name": "Patient Level of Consciousness", "Variable Name": "consciousness", "values": "Checkbox"},
    {"Display Name": "Temperature in Celsius", "Variable Name": "temperature", "values": "Numeric"},
    {"Display Name": "Pulse Rate", "Variable Name": "pulse_rate", "values": "Numeric"},
    {"Display Name": "Systolic", "Variable Name": "systolic", "values": "Numeric"},
    {"Display Name": "Diastolic", "Variable Name": "diastolic", "values": "Numeric"},
    {"Display Name": "Respiratory Rate", "Variable Name": "respiratory_rate", "values": "Numeric"},
    {"Display Name": "Oxygen Saturation", "Variable Name": "oxygen_saturation", "values": "Numeric"},
    {"Display Name": "Weight in KG", "Variable Name": "weight", "values": "Numeric"},
    {"Display Name": "Height in CM", "Variable Name": "height", "values": "Numeric"},
    {"Display Name": "BMI", "Variable Name": "bmi", "values": "Numeric"},
    {"Display Name": "MUAC", "Variable Name": "muac", "values": "Numeric"},
    {"Display Name": "Nutritional Status", "Variable Name": "nutrional_status", "values": "Categorical"},
    {"Display Name": "LMP (Last Menstrual Period)", "Variable Name": "lmp", "values": "Date"},
    {"Display Name": "Triage Notes", "Variable Name": "triage_notes", "values": "Text"},

    # --- Second form unique fields ---
    {"Display Name": "Clinical Notes", "Variable Name": "clinical_notes", "values": "Text"},
    {"Display Name": "Any Medication Taken Before the Visit", "Variable Name": "medication", "values": "Categorical"},
    {"Display Name": "Drug name", "Variable Name": "drug_name", "values": "Text"},
    {"Display Name": "Dosage", "Variable Name": "dosage", "values": "Numeric"},
    {"Display Name": "Frequency", "Variable Name": "frequency", "values": "Categorical"},
    {"Display Name": "Route of Administration", "Variable Name": "route_administration", "values": "Categorical"},
    {"Display Name": "Herbal Remedies", "Variable Name": "herbal", "values": "Categorical"},
    {"Display Name": "Blood Transfusion", "Variable Name": "blood_transfusion", "values": "Categorical"},
    {"Display Name": "Surgical history", "Variable Name": "surgical_history", "values": "Categorical"},
    {"Display Name": "Type of Surgery performed", "Variable Name": "surgery_type", "values": "Text"},
    {"Display Name": "Date of Surgery", "Variable Name": "surgery_date", "values": "Date"},
    {"Display Name": "Indication of Surgery", "Variable Name": "surgery_indication", "values": "Text"},
    {"Display Name": "Admission History", "Variable Name": "admission_history", "values": "Categorical"},
    {"Display Name": "Reason for Admission", "Variable Name": "admission_reason", "values": "Text"},
    {"Display Name": "Date of Admissions", "Variable Name": "admission_date", "values": "Date"},
    {"Display Name": "Adverse Drug Reactions", "Variable Name": "adverse_drug_reactions", "values": "Categorical"},
    {"Display Name": "Medicine Causing Reaction", "Variable Name": "reaction_medicine", "values": "Categorical"},
    {"Display Name": "Reaction", "Variable Name": "reaction_type", "values": "Categorical"},
    {"Display Name": "Severity", "Variable Name": "reaction_severity", "values": "Categorical"},
    {"Display Name": "Date of Onset", "Variable Name": "reaction_date", "values": "Date"},
    {"Display Name": "Action Taken", "Variable Name": "reaction_action", "values": "Categorical"},
    {"Display Name": "Ever had menses", "Variable Name": "menses_ever", "values": "Categorical"},
    {"Display Name": "Age at menarche", "Variable Name": "menarche_age", "values": "Numeric"},
    {"Display Name": "Menses Frequency", "Variable Name": "menses_frequency", "values": "Categorical"},
    {"Display Name": "Menstrual Flow Characteristics", "Variable Name": "menstrual_characteristics", "values": "Categorical"},
    {"Display Name": "Duration of Menses", "Variable Name": "menses_duration", "values": "Numeric"},
    {"Display Name": "History of intermenstrual vaginal bleeding", "Variable Name": "intermenstrual_bleeding", "values": "Categorical"},
    {"Display Name": "Associated Symptoms", "Variable Name": "intermenstrual_bleeding_symptoms", "values": "Text"},
    {"Display Name": "Reason for Amenorrhea", "Variable Name": "amenorrhea_reason", "values": "Categorical"},
    {"Display Name": "Other reason", "Variable Name": "amenorrhea_other", "values": "Text"},
    {"Display Name": "Previous Gynecological Survey", "Variable Name": "gynacological_survey", "values": "Categorical"},
    {"Display Name": "Name of Gynacological Survey", "Variable Name": "gynacological_name", "values": "Text"},
    {"Display Name": "Family Planning Status", "Variable Name": "fp_status", "values": "Categorical"},
    {"Display Name": "Family Planning Method", "Variable Name": "fp_method", "values": "Categorical"},
    {"Display Name": "Reason for Not Using Family Planning", "Variable Name": "fp_not_reason", "values": "Checkbox"},
    {"Display Name": "History of Trauma", "Variable Name": "trauma", "values": "Categorical"},
    {"Display Name": "Any primary doctor", "Variable Name": "primary_doctor", "values": "Categorical"},
    {"Display Name": "Developmental Milestone History", "Variable Name": "developmental_milestone", "values": "Categorical"},
    {"Display Name": "Milestone Delayed", "Variable Name": "milestone_delayed", "values": "Text"},
    {"Display Name": "Milestone Regressed", "Variable Name": "milestone_regressed", "values": "Text"},
    {"Display Name": "Mode of feeding", "Variable Name": "mode_feeding", "values": "Categorical"},
    {"Display Name": "General Examination Findings", "Variable Name": "exam_findings", "values": "Checkbox"},
    {"Display Name": "Exam Notes", "Variable Name": "exam_notes", "values": "Text"},
    {"Display Name": "Abnormal Gait Finding", "Variable Name": "abnormal_gait_finding", "values": "Categorical"},
    {"Display Name": "Cyanosis Type", "Variable Name": "cyanosis_type", "values": "Categorical"},
    {"Display Name": "Dehydration Type", "Variable Name": "dehydration_type", "values": "Categorical"},
    {"Display Name": "Finger Clubbing Grade", "Variable Name": "finger_clubbing_grade", "values": "Categorical"},
    {"Display Name": "Jaundice Grading", "Variable Name": "juandice_grading", "values": "Categorical"},
    {"Display Name": "Lymphadenopathy", "Variable Name": "Lymphadenopathy_site", "values": "Categorical"},
    {"Display Name": "Oedema Site", "Variable Name": "oedema_site", "values": "Categorical"},
    {"Display Name": "Pallor Location", "Variable Name": "pallor_location", "values": "Categorical"},
    {"Display Name": "Pallor Site", "Variable Name": "pallor_site", "values": "Categorical"},
    {"Display Name": "System Examination", "Variable Name": "system_examination", "values": "Categorical"},
    {"Display Name": "Systems Reviews", "Variable Name": "system_reviews", "values": "Checkbox"}
]

case_triage = {
  "complaint_today": "Yes",
  "complaints": [
    {"complaint": "Vaginal bleeding", "duration": 3, "onset_status": "Sudden"},
    {"complaint": "Pain during intercourse", "duration": 7, "onset_status": "Gradual"}
  ],
  "sex": "Female",
  "age": 29,
  "patient_mobility": "Ambulatory",
  "injury": "No",
  "consciousness": ["Alert"],
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

case_encounter = {
    "complaint_today": "Yes",
    "clinical_notes": "Patient reports intermittent abdominal pain over the past week.",
    "complaints": [
        {"complaint": "Vaginal bleeding", "duration": 3, "onset_status": "Sudden"},
        {"complaint": "Pain during intercourse", "duration": 7, "onset_status": "Gradual"}
    ],
    "medication": "Yes",
    "drug_name": "Paracetamol",
    "dosage": 500,
    "frequency": "Twice daily",
    "route_administration": "Oral",
    "herbal": "No",
    "blood_transfusion": "No",
    "surgical_history": "Yes",
    "surgery_type": "Appendectomy",
    "surgery_date": "2020-05-12",
    "surgery_indication": "Appendicitis",
    "admission_history": "Yes",
    "admission_reason": "Severe dehydration",
    "admission_date": "2021-03-20",
    "adverse_drug_reactions": "Yes",
    "drug_reactions": [
        {"reaction_medicine": "Amoxicillin", "reaction_type": "Rash", "reaction_severity": "Severe", "reaction_date": "2022-09-15", "reaction_action": "Stopped medication"}
    ],
    "menses_ever": "Yes",
    "menarche_age": 13,
    "lmp": "2025-08-15",
    "menses_frequency": "Regular (28 days)",
    "menstrual_characteristics": "Normal flow",
    "menses_duration": 5,
    "intermenstrual_bleeding": "No",
    "intermenstrual_bleeding_symptoms": "",
    "amenorrhea_reason": "Not Applicable",
    "amenorrhea_other": "",
    "gynacological_survey": "No",
    "gynacological_name": "",
    "fp_status": "On Family Planning",
    "fp_method": "Oral Contraceptive Pills",
    "fp_not_reason": [],
    "trauma": "No",
    "primary_doctor": "Yes",
    "developmental_milestone": "Normal",
    "milestone_delayed": "",
    "milestone_regressed": "",
    "mode_feeding": "Exclusive breastfeeding",
    "exam_findings": ["Pallor", "Oedema"],
    "exam_notes": "Mild pallor observed, slight pedal oedema.",
    "abnormal_gait_finding": "",
    "cyanosis_type": "",
    "dehydration_type": "",
    "finger_clubbing_grade": "",
    "juandice_grading": "",
    "Lymphadenopathy_site": "",
    "oedema_site": "Pedal",
    "pallor_location": "Conjunctiva",
    "pallor_site": "Generalized",
    "system_examination": "Abnormal",
    "system_reviews": ["Respiratory system", "Cardiovascular system"]
}


if __name__ == "__main__":
    print(prep_case(case_triage, case_metadata))