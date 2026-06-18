import glob
import json
import unittest

from backend.safety_gate import enforce_answer_policy


BLOCKED_STATUSES = {"weak_support", "unsupported", "insufficient", "contradicted"}


def _load_artifact(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _verification_from_artifact(analysis):
    imaging = analysis.get("imaging") or {}
    imaging_status = imaging.get("status", "N/A") if isinstance(imaging, dict) else str(imaging)
    return {
        "risk_tier": analysis.get("risk_tier"),
        "risk_score": analysis.get("risk_score"),
        "risk_reasons": analysis.get("risk_reasons") or [],
        "kg": analysis.get("kg"),
        "nli": analysis.get("nli") or {},
        "rag_score": analysis.get("rag_score"),
        "rag_verified": analysis.get("rag_verified"),
        "rag_error": analysis.get("rag_error"),
        "citations": analysis.get("citations") or [],
        "imaging": imaging_status,
        "warnings": [],
        "suggestions": [],
        "matched_conditions": [],
    }


def _safety_from_artifact(analysis):
    return {
        "claims": [],
        "claim_verification": analysis.get("claim_verification") or [],
        "claims_summary": analysis.get("claims_summary") or {},
        "confidence": analysis.get("confidence") or {},
        "warnings": [],
    }


def _rag_from_artifact(analysis):
    safety = analysis.get("safety") or {}
    pre_generation = safety.get("pre_generation") or {}
    return {
        "citations": analysis.get("citations") or [],
        "safety_precheck": {
            "filtered_hits": [],
            "source_conflicts": pre_generation.get("source_conflicts") or [],
        },
    }


class SafetyGateArtifactMetricTests(unittest.TestCase):
    def test_curated_artifacts_do_not_expose_blocked_claims(self):
        paths = sorted(glob.glob("runtime_artifacts/*.json"))
        self.assertGreater(len(paths), 0, "expected saved runtime artifacts")

        exposed_blocked_claims = 0
        artifacts_with_claims = 0
        artifacts_with_blocked_claims = 0

        for path in paths:
            payload = _load_artifact(path)
            analysis = payload.get("analysis") or {}
            claim_rows = analysis.get("claim_verification") or []
            if not claim_rows:
                continue
            artifacts_with_claims += 1
            blocked = [
                row for row in claim_rows
                if row.get("status") in BLOCKED_STATUSES
            ]
            if blocked:
                artifacts_with_blocked_claims += 1

            gate = enforce_answer_policy(
                query=payload.get("query") or "",
                candidate_answer=payload.get("final_response") or "",
                verification=_verification_from_artifact(analysis),
                rag_result=_rag_from_artifact(analysis),
                safety_result=_safety_from_artifact(analysis),
            )
            if gate["answer_policy"]["decision"] == "answer":
                exposed_blocked_claims += len(blocked)

        self.assertGreater(artifacts_with_claims, 0, "expected artifacts with claim verification")
        self.assertGreater(artifacts_with_blocked_claims, 0, "expected at least one blocked-claim regression artifact")
        self.assertEqual(exposed_blocked_claims, 0)


if __name__ == "__main__":
    unittest.main()
