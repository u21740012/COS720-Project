def generate_simple_explanation(row: dict) -> str:
    reasons = []

    if row.get("num_printed_pages_off_hours", 0) > 0:
        reasons.append("printing activity outside normal working hours")

    if row.get("late_exit_flag", 0) == 1:
        reasons.append("late exit behaviour")

    if row.get("entry_during_weekend", 0) == 1:
        reasons.append("weekend access activity")

    if row.get("total_files_burned", 0) > 0:
        reasons.append("file burning activity")

    if row.get("burned_from_other", 0) > 0:
        reasons.append("burning files from another source")

    if row.get("num_entries", 0) > 8:
        reasons.append("unusually high access frequency")

    if row.get("num_unique_campus", 0) > 1:
        reasons.append("activity across multiple campuses")

    if row.get("is_abroad", 0) == 1:
        reasons.append("activity while abroad")

    if row.get("hostility_country_level", 0) >= 3:
        reasons.append("operation in a higher-risk country context")

    if not reasons:
        return "The system did not find any strong abnormal behavioural indicators in this record."

    return "This activity was flagged mainly because of " + ", ".join(reasons) + "."