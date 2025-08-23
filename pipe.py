import pandas as pd
import json
from datetime import datetime, date
import logging
import os
from io import StringIO

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Mock LLM classifier for ambiguous denial reasons
def mock_llm_classifier(denial_reason):
    """
    Mock classifier for ambiguous denial reasons.
    Returns True if retryable, False otherwise.
    """
    if denial_reason is None or pd.isna(denial_reason):
        return False  # null or NaN is non-retryable
    reason_lower = str(denial_reason).lower()
    if "incorrect procedure" in reason_lower:
        return False  # Based on example, assuming non-retryable
    if "form incomplete" in reason_lower:
        return True  # Assuming retryable
    if "not billable" in reason_lower:
        return False
    # Default to False for unknown
    return False


# Unified schema function
def normalize_record(record, source_system):
    """
    Normalize a single record to the unified schema.
    """
    normalized = {
        "claim_id": str(record.get("claim_id") or record.get("id", "unknown")).strip(),
        "patient_id": str(
            record.get("patient_id") or record.get("member", "unknown")
        ).strip(),
        "procedure_code": str(record.get("procedure_code") or "unknown").strip(),
        "denial_reason": (
            str(
                record.get("denial_reason") or record.get("error_msg", "unknown")
            ).strip()
            if not pd.isna(record.get("denial_reason") or record.get("error_msg"))
            else "unknown"
        ),
        "status": (
            str(record.get("status", "unknown")).lower().strip()
            if not pd.isna(record.get("status"))
            else "unknown"
        ),
        "submitted_at": record.get("submitted_at") or record.get("date"),
        "source_system": source_system,
    }

    # Normalize date to ISO format
    try:
        if isinstance(normalized["submitted_at"], str):
            if "T" in normalized["submitted_at"]:
                normalized["submitted_at"] = (
                    datetime.fromisoformat(
                        normalized["submitted_at"].replace("Z", "+00:00")
                    )
                    .date()
                    .isoformat()
                )
            else:
                normalized["submitted_at"] = (
                    datetime.strptime(normalized["submitted_at"], "%Y-%m-%d")
                    .date()
                    .isoformat()
                )
    except ValueError:
        logging.warning(
            f"Invalid date format for claim {normalized['claim_id']}: {normalized['submitted_at']}"
        )
        normalized["submitted_at"] = None

    return normalized


# Eligibility check
def is_eligible_for_resubmission(record, today=date(2025, 7, 30)):
    """
    Check if a claim is eligible for resubmission.
    """
    if record["status"] != "denied":
        return False, "Status not denied"

    if record["patient_id"] == "unknown":  # Null patient_id
        return False, "Patient ID is null"

    if record["submitted_at"] is None:
        return False, "Invalid submitted date"

    submitted_date = datetime.fromisoformat(record["submitted_at"]).date()
    days_ago = (today - submitted_date).days
    if days_ago <= 7:
        return False, "Submitted less than or equal to 7 days ago"

    retryable_reasons = {"Missing modifier", "Incorrect NPI", "Prior auth required"}
    non_retryable_reasons = {"Authorization expired", "Incorrect provider type"}

    reason = record["denial_reason"]
    if reason in retryable_reasons:
        return True, reason
    elif reason in non_retryable_reasons:
        return False, reason
    else:
        # Ambiguous - use classifier
        if mock_llm_classifier(reason):
            return True, f"Inferred retryable: {reason}"
        else:
            return False, f"Inferred non-retryable: {reason}"


# Main pipeline function (unchanged)
def run_pipeline(alpha_data, beta_data):
    """
    Run the ingestion pipeline.
    """
    claims = []
    failed_records = []

    # Process alpha (CSV)
    alpha_df = pd.read_csv(StringIO(alpha_data))
    for _, row in alpha_df.iterrows():
        try:
            normalized = normalize_record(row.to_dict(), "alpha")
            claims.append(normalized)
        except Exception as e:
            logging.error(f"Failed to process alpha record: {row.to_dict()} - {str(e)}")
            failed_records.append(
                {"source": "alpha", "record": row.to_dict(), "error": str(e)}
            )

    # Process beta (JSON)
    beta_list = json.loads(beta_data)
    for item in beta_list:
        try:
            normalized = normalize_record(item, "beta")
            claims.append(normalized)
        except Exception as e:
            logging.error(f"Failed to process beta record: {item} - {str(e)}")
            failed_records.append({"source": "beta", "record": item, "error": str(e)})

    # Eligibility
    eligible = []
    excluded = []
    for claim in claims:
        is_eligible, reason = is_eligible_for_resubmission(claim)
        if is_eligible:
            eligible.append(
                {
                    "claim_id": claim["claim_id"],
                    "resubmission_reason": reason,
                    "source_system": claim["source_system"],
                    "recommended_changes": f"Review and correct '{reason}' and resubmit",
                }
            )
        else:
            excluded.append(
                {
                    "claim_id": claim["claim_id"],
                    "exclusion_reason": reason,
                    "source_system": claim["source_system"],
                }
            )

    # Metrics
    total_processed = len(claims)
    from_alpha = sum(1 for c in claims if c["source_system"] == "alpha")
    from_beta = sum(1 for c in claims if c["source_system"] == "beta")
    flagged = len(eligible)
    excluded_count = len(excluded)

    logging.info(f"Total claims processed: {total_processed}")
    logging.info(f"From alpha: {from_alpha}")
    logging.info(f"From beta: {from_beta}")
    logging.info(f"Flagged for resubmission: {flagged}")
    logging.info(f"Excluded: {excluded_count}")

    # Save output
    with open("resubmission_candidates.json", "w") as f:
        json.dump(eligible, f, indent=4)

    # Save failed records
    if failed_records:
        with open("rejection_log.json", "w") as f:
            json.dump(failed_records, f, indent=4)

    return eligible


# Data (hardcoded from document)
alpha_csv = """claim_id,patient_id,procedure_code,denial_reason,submitted_at,status
A123,P001,99213,Missing modifier,2025-07-01,denied
A124,P002,99214,Incorrect NPI,2025-07-10,denied
A125,,99215,Authorization expired,2025-07-05,denied
A126,P003,99381,,2025-07-15,approved
A127,P004,99401,Prior auth required,2025-07-20,denied"""

beta_json = """[
  {
    "id": "B987",
    "member": "P010",
    "code": "99213",
    "error_msg": "Incorrect provider type",
    "date": "2025-07-03T00:00:00",
    "status": "denied"
  },
  {
    "id": "B988",
    "member": "P011",
    "code": "99214",
    "error_msg": "Missing modifier",
    "date": "2025-07-09T00:00:00",
    "status": "denied"
  },
  {
    "id": "B989",
    "member": "P012",
    "code": "99215",
    "error_msg": null,
    "date": "2025-07-10T00:00:00",
    "status": "approved"
  },
  {
    "id": "B990",
    "member": null,
    "code": "99401",
    "error_msg": "incorrect procedure",
    "date": "2025-07-01T00:00:00",
    "status": "denied"
  }
]"""

# Run the pipeline
if __name__ == "__main__":
    eligible_claims = run_pipeline(alpha_csv, beta_json)
    print("Eligible claims:", eligible_claims)
