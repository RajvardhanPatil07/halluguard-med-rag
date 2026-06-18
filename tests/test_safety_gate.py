import unittest

from backend.safety_gate import (
    enforce_answer_policy,
    sanitize_hidden_candidate_safety,
    sanitize_hidden_candidate_verification,
)


def _rag_result(conflicts=None):
    return {
        "citations": [
            {
                "id": "E1",
                "source": "MedlinePlus",
                "condition": "appendicitis",
                "section": "overview",
                "text": "Appendicitis can cause pain that starts near the belly button and moves to the lower right abdomen. It can be a medical emergency.",
                "rank": 1,
            }
        ],
        "safety_precheck": {
            "filtered_hits": [
                {
                    "citation_id": "E1",
                    "source": "MedlinePlus",
                    "condition": "appendicitis",
                    "section": "overview",
                    "text": "Appendicitis can cause pain that starts near the belly button and moves to the lower right abdomen. It can be a medical emergency.",
                    "rank": 1,
                }
            ],
            "source_conflicts": conflicts or [],
        },
    }


def _verification(risk_tier="Tier 1", nli_label="Entailed", rag_verified=True, imaging="N/A"):
    return {
        "risk_tier": risk_tier,
        "risk_score": 0.0,
        "risk_reasons": [],
        "kg": "Match",
        "nli": {"label": nli_label, "confidence": 0.95, "claims": [{"claim": "raw draft claim"}]},
        "rag_score": 0.9,
        "rag_verified": rag_verified,
        "rag_error": None,
        "citations": [],
        "imaging": imaging,
        "warnings": [],
        "suggestions": [],
        "matched_conditions": [],
    }


def _safety(statuses, should_answer=True, should_refuse=False):
    rows = [
        {
            "claim_id": f"C{index}",
            "claim": f"raw candidate claim {index}",
            "status": status,
            "support_score": 0.9 if status == "supported" else 0.2,
            "contradiction_score": 0.9 if status == "contradicted" else 0.0,
            "best_citation_id": "E1",
            "reason": "test",
        }
        for index, status in enumerate(statuses, start=1)
    ]
    return {
        "claims": [{"claim_id": row["claim_id"], "text": row["claim"], "claim_type": "general"} for row in rows],
        "claim_verification": rows,
        "claims_summary": {"total": len(rows)},
        "confidence": {
            "score": 0.9 if should_answer else 0.3,
            "label": "high" if should_answer else "low",
            "should_answer": should_answer,
            "should_refuse": should_refuse,
            "reasons": [],
        },
        "warnings": [],
    }


class SafetyGateTests(unittest.TestCase):
    def test_supported_claims_show_candidate_answer(self):
        candidate = "Supported answer [E1]."
        result = enforce_answer_policy(
            query="appendicitis warning signs",
            candidate_answer=candidate,
            verification=_verification(),
            rag_result=_rag_result(),
            safety_result=_safety(["supported", "supported"]),
        )

        self.assertEqual(result["final_response"], candidate)
        self.assertEqual(result["answer_policy"]["decision"], "answer")
        self.assertFalse(result["answer_policy"]["raw_candidate_hidden"])

    def test_unsupported_claim_returns_evidence_summary(self):
        result = enforce_answer_policy(
            query="appendicitis warning signs",
            candidate_answer="This raw draft should not appear.",
            verification=_verification(),
            rag_result=_rag_result(),
            safety_result=_safety(["supported", "unsupported"]),
        )

        self.assertEqual(result["answer_policy"]["decision"], "evidence_summary")
        self.assertTrue(result["answer_policy"]["raw_candidate_hidden"])
        self.assertIn("What retrieved evidence supports", result["final_response"])
        self.assertNotIn("This raw draft should not appear", result["final_response"])

    def test_weak_claim_returns_evidence_summary(self):
        result = enforce_answer_policy(
            query="appendicitis warning signs",
            candidate_answer="Weakly supported draft.",
            verification=_verification(),
            rag_result=_rag_result(),
            safety_result=_safety(["weak_support"]),
        )

        self.assertEqual(result["answer_policy"]["decision"], "evidence_summary")
        self.assertEqual(result["answer_policy"]["blocked_claims_count"], 1)

    def test_supported_but_uncited_answer_returns_evidence_summary(self):
        result = enforce_answer_policy(
            query="appendicitis warning signs",
            candidate_answer="Supported answer without an inline citation.",
            verification=_verification(),
            rag_result=_rag_result(),
            safety_result=_safety(["supported"]),
        )

        self.assertEqual(result["answer_policy"]["decision"], "evidence_summary")
        self.assertIn("missing_inline_citations", result["answer_policy"]["reasons"])

    def test_contradicted_claim_refuses(self):
        result = enforce_answer_policy(
            query="guaranteed cure",
            candidate_answer="Unsafe draft.",
            verification=_verification(),
            rag_result=_rag_result(),
            safety_result=_safety(["contradicted"], should_answer=False),
        )

        self.assertEqual(result["answer_policy"]["decision"], "refuse")
        self.assertEqual(result["answer_policy"]["final_response_source"], "safety_refusal")
        self.assertIn("high-risk verification issue", result["final_response"])

    def test_no_claims_returns_insufficient_evidence_summary(self):
        result = enforce_answer_policy(
            query="rare disease management",
            candidate_answer="No claims draft.",
            verification=_verification(),
            rag_result=_rag_result(),
            safety_result=_safety([]),
        )

        self.assertEqual(result["answer_policy"]["decision"], "evidence_summary")
        self.assertIn("no_claims_verified", result["answer_policy"]["reasons"])

    def test_hidden_candidate_sanitizers_remove_raw_claim_text(self):
        policy = {
            "raw_candidate_hidden": True,
        }
        safety = sanitize_hidden_candidate_safety(_safety(["unsupported"]), policy)
        verification = sanitize_hidden_candidate_verification(_verification(), policy)

        self.assertNotIn("text", safety["claims"][0])
        self.assertNotIn("claim", safety["claim_verification"][0])
        self.assertNotIn("claim", verification["nli"]["claims"][0])


if __name__ == "__main__":
    unittest.main()
