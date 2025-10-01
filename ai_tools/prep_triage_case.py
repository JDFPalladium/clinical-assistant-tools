import json
import pandas as pd

def decode_obs(obs_list, concept_map, value_map):
    """Turn obs block into a list of (concept_label, value_label_or_number)"""
    results = []
    for o in obs_list:
        if "groupMembers" in o:  # handle nested groups
            results.extend(decode_obs(o["groupMembers"], concept_map, value_map))
        else:
            concept_label = concept_map.get(o.get("concept"), o.get("concept"))
            val = o.get("value")
            if isinstance(val, str):
                val_label = value_map.get(val, val)
            else:
                val_label = val
            results.append((concept_label, val_label))

    return results


if __name__ == "__main__":
    from schemas import triage_actual
    mapping_df = pd.read_csv("../data/processed/TriageFormDecoding.csv")

    # Build lookup dictionaries
    concept_map = mapping_df.query("concept == 'concept'").set_index("key")["value"].to_dict()
    value_map   = mapping_df.query("concept == 'value'").set_index("key")["value"].to_dict()
    # Decode example payload    
    decoded_obs = decode_obs(triage_actual["obs"], concept_map, value_map)
    print(decoded_obs)