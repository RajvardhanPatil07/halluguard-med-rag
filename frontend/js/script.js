/* ============================================================
   HalluGuard-Med - script.js
   Handles: navbar toggle, assistant chat, message rendering,
            analysis panel, radiology bar chart, image upload
   ============================================================ */

/* ---------- SVG icon helpers ---------- */
const SVG = {
  shield: `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.955 11.955 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z"/></svg>`,
  send:   `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.269 20.876L5.999 12zm0 0h7.5"/></svg>`,
  image:  `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z"/></svg>`,
  check:  `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`,
  xmark:  `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`,
  minus:  `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15 12H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`,
  warn:   `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/></svg>`,
  net:    `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5"/></svg>`,
  brain:  `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z"/></svg>`,
  steth:  `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"/></svg>`,
  rag:    `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125"/></svg>`,
  xray:   `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 4.875c0-.621.504-1.125 1.125-1.125h4.5c.621 0 1.125.504 1.125 1.125v4.5c0 .621-.504 1.125-1.125 1.125h-4.5A1.125 1.125 0 013.75 9.375v-4.5zM3.75 14.625c0-.621.504-1.125 1.125-1.125h4.5c.621 0 1.125.504 1.125 1.125v4.5c0 .621-.504 1.125-1.125 1.125h-4.5a1.125 1.125 0 01-1.125-1.125v-4.5zM13.5 4.875c0-.621.504-1.125 1.125-1.125h4.5c.621 0 1.125.504 1.125 1.125v4.5c0 .621-.504 1.125-1.125 1.125h-4.5A1.125 1.125 0 0113.5 9.375v-4.5z"/></svg>`,
};

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";
const BACKEND_URL_STORAGE_KEY = "halluguard_backend_url";

function normalizeBackendUrl(url) {
  return String(url || "").trim().replace(/\/+$/, "");
}

function getBackendBaseUrl() {
  const savedUrl = normalizeBackendUrl(localStorage.getItem(BACKEND_URL_STORAGE_KEY));
  return savedUrl || DEFAULT_BACKEND_URL;
}

function getBackendModeLabel() {
  const savedUrl = normalizeBackendUrl(localStorage.getItem(BACKEND_URL_STORAGE_KEY));
  return savedUrl ? "Using Kaggle backend" : "Using local backend";
}

async function getBackendResponse(query, imageDataUrl) {
  const formData = new FormData();
  formData.append("query", query || "[Medical image submitted]");

  if (imageDataUrl) {
    const imageBlob = await fetch(imageDataUrl).then(r => r.blob());
    formData.append("image", imageBlob, "medical-image.png");
  }

  const response = await fetch(`${getBackendBaseUrl()}/api/chat`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    const error = new Error(`Backend returned ${response.status}`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return response.json();
}

/* ---------- Report export integration: send completed HalluGuard output only ---------- */
async function downloadFullReport(payload, button) {
  if (!payload || !payload.analysis) return;
  const originalLabel = button ? button.innerHTML : "";
  if (button) {
    button.disabled = true;
    button.innerHTML = `${SVG.rag} Preparing report`;
  }

  try {
    const response = await fetch(`${getBackendBaseUrl()}/api/download-report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Report export failed with ${response.status}`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const link = document.createElement("a");
    link.href = url;
    link.download = `halluguard_report_${timestamp}.pdf`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    appendSystemWarning(
      "Report download failed.",
      "The analysis was preserved, but the PDF report could not be generated. Please check the backend report endpoint."
    );
  } finally {
    if (button) {
      button.disabled = false;
      button.innerHTML = originalLabel;
    }
  }
}

/* ---------- Risk tier config ---------- */
const TIER_CONFIG = {
  "Tier 1": { cls: "tier1", label: "Low Risk",      icon: SVG.check },
  "Tier 2": { cls: "tier2", label: "Moderate Risk", icon: SVG.warn  },
  "Tier 3": { cls: "tier3", label: "High Risk",     icon: SVG.xmark },
};

/* ---------- Check result config ---------- */
const CHECK_CONFIG = {
  "Match":        { cls: "green", icon: SVG.check },
  "Entailed":     { cls: "green", icon: SVG.check },
  "Neutral":      { cls: "gray",  icon: SVG.minus },
  "N/A":          { cls: "gray",  icon: SVG.minus },
  "Mismatch":     { cls: "red",   icon: SVG.xmark },
  "Contradicted": { cls: "red",   icon: SVG.xmark },
  "Analyzed":     { cls: "green", icon: SVG.check },
  "Error":        { cls: "red",   icon: SVG.xmark },
  "Insufficient Evidence": { cls: "gray", icon: SVG.warn },
  "Insufficient": { cls: "gray", icon: SVG.warn },
};

/* ---------- Bar color based on percentage ---------- */
function getBarColor(pct) {
  if (pct >= 60) return "#ef4444"; // red
  if (pct >= 40) return "#f97316"; // orange
  return "#22c55e";                // green
}

/* ---------- Build radiology bar chart panel ---------- */
function buildRadiologyPanel(imaging) {
  if (!imaging || imaging.status === "N/A") return "";
  if (imaging.status === "Error") {
    return `
      <div class="radiology-panel">
        <div class="radiology-panel__header">${SVG.xray} Image Findings - Error</div>
        <div class="radiology-panel__error">
          ${imaging.warnings && imaging.warnings.length
            ? imaging.warnings.join(" ")
            : "Imaging analysis failed."}
        </div>
      </div>`;
  }

  const scores = imaging.percentage_scores;
  const critical = imaging.critical || [];
  const findings = imaging.findings || [];
  const normalScore = imaging.normal_score;

  // Critical banner
  const criticalHtml = critical.length > 0 ? `
    <div class="radiology-critical">
      ${SVG.warn} Critical: ${critical.join(", ")}
    </div>` : "";

  // Normal score line
  const normalHtml = normalScore !== null ? `
    <div class="radiology-normal">
      Image model confidence - Normal: ${normalScore}%
    </div>` : "";

  const findingsHtml = findings.length ? `
    <div class="radiology-findings">
      ${findings.map(item => `<span>${escapeHtml(item)}</span>`).join("")}
    </div>` : "";

  // Bar chart rows
  let barsHtml = "";
  if (scores && Object.keys(scores).length > 0) {
    barsHtml = Object.entries(scores).map(([label, pct]) => {
      const color = getBarColor(pct);
      const isCritical = critical.includes(label);
      const critMark = isCritical
        ? `<span class="radiology-bar__critical-mark">[!]</span>`
        : "";
      return `
        <div class="radiology-bar__row">
          <div class="radiology-bar__label">${label}${critMark}</div>
          <div class="radiology-bar__track">
            <div class="radiology-bar__fill"
                 style="width:${Math.min(pct, 100)}%; background:${color}">
            </div>
          </div>
          <div class="radiology-bar__pct" style="color:${color}">${pct}%</div>
        </div>`;
    }).join("");
  } else {
    barsHtml = `<div class="radiology-no-findings">No significant findings detected above threshold.</div>`;
  }

  return `
    <div class="radiology-panel">
      <div class="radiology-panel__header">${SVG.xray} Image Findings</div>
      ${criticalHtml}
      ${normalHtml}
      ${findingsHtml}
      <div class="radiology-bars">${barsHtml}</div>
    </div>`;
}

function buildCitationsPanel(citations) {
  if (!citations || !citations.length) return "";
  const rows = citations.map(citation => {
    const score = citation.score != null ? `${(citation.score * 100).toFixed(0)}%` : "n/a";
    return `
      <div class="citation-item">
        <div class="citation-item__meta">
          <strong>${escapeHtml(citation.id || "Evidence")}</strong>
          <span>${escapeHtml(citation.condition || "medical evidence")}</span>
          <span>${escapeHtml(citation.section || "")}</span>
          <span>${score}</span>
        </div>
        <div class="citation-item__text">${escapeHtml(citation.text || "")}</div>
      </div>`;
  }).join("");
  return `
    <div class="evidence-panel">
      <div class="evidence-panel__title">${SVG.rag} Evidence</div>
      <div class="evidence-panel__items">${rows}</div>
    </div>`;
}

function buildReasonsPanel(reasons, riskScore) {
  const hasReasons = reasons && reasons.length;
  if (!hasReasons && riskScore == null) return "";
  const scoreHtml = riskScore != null
    ? `<div class="risk-score">Risk score: ${Number(riskScore).toFixed(2)}</div>`
    : "";
  const reasonsHtml = hasReasons
    ? reasons.map(reason => `<div class="panel-list__item">${escapeHtml(reason)}</div>`).join("")
    : `<div class="panel-list__item">No additional verification concerns reported.</div>`;
  return `
    <div>
      <div class="panel-list__title">${SVG.shield} Findings</div>
      ${scoreHtml}
      <div class="panel-list__items">${reasonsHtml}</div>
    </div>`;
}

function getConfidenceDecision(confidence) {
  if (!confidence) {
    return {
      cls: "review",
      icon: SVG.warn,
      label: "Review",
      title: "Safety confidence unavailable",
      text: "The response was generated, but confidence fusion data was not returned.",
    };
  }
  if (confidence.should_refuse) {
    return {
      cls: "refuse",
      icon: SVG.xmark,
      label: "Refuse",
      title: "Do not rely on this answer",
      text: "The safety layer found a contradiction, source conflict, or insufficient support.",
    };
  }
  if (!confidence.should_answer) {
    return {
      cls: "review",
      icon: SVG.warn,
      label: "Review",
      title: "Use only after review",
      text: "The answer is not fully supported. Check the citations and claim verification before using it.",
    };
  }
  return {
    cls: "answer",
    icon: SVG.check,
    label: "Answer",
    title: "Answer allowed",
    text: "The response passed the confidence fusion checks. Continue to treat it as clinical decision support only.",
  };
}

function buildConfidencePanel(analysis) {
  const confidence = analysis.confidence || {};
  const label = confidence.label || "unknown";
  const score = confidence.score != null ? `${Math.round(confidence.score * 100)}%` : "n/a";
  const decision = getConfidenceDecision(confidence);
  const reasons = confidence.reasons || [];
  const tags = reasons.length
    ? reasons.map(reason => `<span class="decision-tag">${escapeHtml(reason.replace(/_/g, " "))}</span>`).join("")
    : `<span class="decision-tag">No confidence blockers</span>`;

  return `
    <div class="safety-decision">
      <div class="confidence-card ${escapeHtml(label)}">
        <div class="confidence-card__top">
          <div>
            <div class="confidence-card__label">Confidence</div>
            <div class="confidence-card__value">${score}</div>
          </div>
          <div class="confidence-card__status ${decision.cls}">${decision.icon} ${escapeHtml(label)}</div>
        </div>
      </div>
      <div class="decision-panel ${decision.cls}">
        <div class="decision-panel__title">${decision.icon} ${decision.title}</div>
        <div class="decision-panel__text">${decision.text}</div>
        <div class="decision-panel__policy">
          Recommendation: ${decision.cls === "refuse"
            ? "do not rely on this answer without clinician review"
            : decision.cls === "review"
              ? "review supporting evidence before using this response"
              : "use as decision support with the standard medical disclaimer"}
        </div>
        <div class="decision-tags">${tags}</div>
      </div>
    </div>`;
}

function claimStatusIcon(status) {
  if (status === "supported") return SVG.check;
  if (status === "contradicted") return SVG.xmark;
  return SVG.warn;
}

function buildClaimVerificationPanel(claims) {
  if (!claims || !claims.length) return "";
  const supported = claims.filter(item => item.status === "supported").length;
  const rows = claims.map(item => {
    const status = item.status || "insufficient";
    const support = item.support_score != null ? `${Math.round(item.support_score * 100)}%` : "n/a";
    const citation = item.best_citation_id || "No citation";
    const evidence = item.best_evidence
      ? `<div class="claim-body__evidence"><strong>${escapeHtml(citation)}</strong>: ${escapeHtml(item.best_evidence)}</div>`
      : `<div class="claim-body__evidence"><strong>${escapeHtml(citation)}</strong></div>`;
    return `
      <div class="claim-row">
        <div class="claim-status">
          <div class="claim-status__pill ${escapeHtml(status)}">${claimStatusIcon(status)} ${escapeHtml(status)}</div>
          <div class="claim-status__score">${support} support</div>
        </div>
        <div class="claim-body">
          <div class="claim-body__text">${escapeHtml(item.claim || "")}</div>
          ${evidence}
        </div>
      </div>`;
  }).join("");

  return `
    <div class="safety-section">
      <div class="safety-section__header">
        <div class="safety-section__title">${SVG.check} Claim Verification</div>
        <div class="safety-section__meta">${supported}/${claims.length} supported</div>
      </div>
      <div class="claim-list">${rows}</div>
    </div>`;
}

function scoreColor(score) {
  if (score >= 0.7) return "#16a34a";
  if (score >= 0.4) return "#d97706";
  return "#dc2626";
}

function buildEvidenceScoresPanel(scores) {
  if (!scores || !scores.length) return "";
  const rows = scores.slice(0, 6).map(item => {
    const score = Number(item.final_score || 0);
    const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
    const color = scoreColor(score);
    return `
      <div class="evidence-quality__row">
        <div class="evidence-quality__id">${escapeHtml(item.citation_id || "EV")}</div>
        <div class="evidence-quality__track">
          <div class="evidence-quality__fill" style="width:${pct}%; background:${color}"></div>
        </div>
        <div class="evidence-quality__score">${pct}%</div>
        <div class="evidence-quality__badge ${item.passed ? "pass" : "fail"}">${item.passed ? "used" : "filtered"}</div>
      </div>`;
  }).join("");
  return `
    <div class="safety-section">
      <div class="safety-section__header">
        <div class="safety-section__title">${SVG.rag} Evidence Quality</div>
        <div class="safety-section__meta">pre-generation</div>
      </div>
      <div class="evidence-quality">${rows}</div>
    </div>`;
}

function buildSourceConflictsPanel(conflicts) {
  if (!conflicts || !conflicts.length) return "";
  const rows = conflicts.map(conflict => `
    <div class="conflict-item">
      <div class="conflict-item__top">
        <span>${escapeHtml(conflict.conflict_id || "Conflict")}</span>
        <span>${escapeHtml(conflict.severity || "review")}</span>
      </div>
      <div class="conflict-item__text">
        ${escapeHtml(conflict.citation_a || "A")} vs ${escapeHtml(conflict.citation_b || "B")}:
        ${escapeHtml(conflict.reason || "source disagreement")}
      </div>
    </div>`).join("");
  return `
    <div class="safety-section">
      <div class="safety-section__header">
        <div class="safety-section__title">${SVG.warn} Source Conflicts</div>
        <div class="safety-section__meta">${conflicts.length} detected</div>
      </div>
      <div class="conflict-list">${rows}</div>
    </div>`;
}

/* ---------- Build verification analysis panel ---------- */
function buildAnalysisPanel(data) {
  const { analysis, warnings = [], suggestions = [], image_uploaded = false } = data;
  if (!analysis) {
    return `
      <div class="analysis-panel">
        <div class="analysis-panel__header">${SVG.warn} Verification Analysis</div>
        <div class="analysis-panel__body">
          <div class="decision-panel review">
            <div class="decision-panel__title">${SVG.warn} Analysis unavailable</div>
            <div class="decision-panel__text">The backend returned a response without verification fields.</div>
          </div>
        </div>
      </div>`;
  }
  const tier = TIER_CONFIG[analysis.risk_tier] || TIER_CONFIG["Tier 2"];

  // Verification display value
  const nli = analysis.nli || {};
  const nliLabel = nli.label || "Neutral";
  const nliConfidence = nli.confidence != null
    ? ` (${(nli.confidence * 100).toFixed(0)}%)`
    : "";
  const nliDisplay = nliLabel + nliConfidence;

  // RAG display
  const ragScore = analysis.rag_score;
  const ragVerified = analysis.rag_verified;
  const ragDisplay = ragScore != null
    ? `${(ragScore * 100).toFixed(0)}%`
    : "Failed";
  const ragCfg = ragScore == null
    ? { cls: "red", icon: SVG.xmark }
    : ragVerified
      ? { cls: "green", icon: SVG.check }
      : { cls: "gray", icon: SVG.minus };

  // Imaging display for check card
  const imagingStatus = (analysis.imaging && analysis.imaging.status) || "N/A";

  // Check cards
  const checks = [
    { label: "Evidence Match",  value: analysis.kg,   icon: SVG.net,   display: analysis.kg },
    { label: "Verification",    value: nliLabel,       icon: SVG.brain, display: nliDisplay },
    { label: "RAG Score",       value: ragDisplay,     icon: SVG.rag,   display: ragDisplay, customCfg: ragCfg },
    { label: "Imaging",         value: imagingStatus,  icon: SVG.steth, display: imagingStatus },
  ];

  const checksHtml = checks.map(c => {
    const cfg = c.customCfg || CHECK_CONFIG[c.value] || CHECK_CONFIG["N/A"];
    return `
      <div class="check-card">
        <div class="check-card__icon">${c.icon}</div>
        <div class="check-card__label">${c.label}</div>
        <div class="check-card__value ${cfg.cls}">${cfg.icon} ${c.display}</div>
      </div>`;
  }).join("");

  const warningsHtml = warnings.length ? `
    <div>
      <div class="panel-list__title danger">${SVG.warn} Warnings</div>
      <div class="panel-list__items">
        ${warnings.map(w => `<div class="panel-list__item danger">${escapeHtml(w)}</div>`).join("")}
      </div>
    </div>` : "";

  const suggestionsHtml = suggestions.length ? `
    <div>
      <div class="panel-list__title safe">${SVG.check} Recommendations</div>
      <div class="panel-list__items">
        ${suggestions.map(s => `<div class="panel-list__item safe">${escapeHtml(s)}</div>`).join("")}
      </div>
    </div>` : "";

  // Radiology panel - only shown if image was uploaded
  const radiologyHtml = image_uploaded
    ? buildRadiologyPanel(analysis.imaging)
    : "";
  const safety = analysis.safety || {};
  const preGeneration = safety.pre_generation || {};
  const confidenceHtml = buildConfidencePanel(analysis);
  const claimsHtml = buildClaimVerificationPanel(analysis.claim_verification || []);
  const evidenceScoresHtml = buildEvidenceScoresPanel(preGeneration.evidence_scores || []);
  const conflictsHtml = buildSourceConflictsPanel(preGeneration.source_conflicts || []);
  const reasonsHtml = buildReasonsPanel(analysis.risk_reasons || [], analysis.risk_score);
  const citationsHtml = buildCitationsPanel(analysis.citations || data.citations || []);

  return `
    <div class="analysis-panel">
      <div class="analysis-panel__header">
        ${SVG.shield} Verification Analysis
      </div>
      <div class="analysis-panel__body">
        <div class="risk-tier ${tier.cls}">
          <div class="risk-tier__icon">${tier.icon}</div>
          <div>
            <div class="risk-tier__name">${analysis.risk_tier} - ${tier.label}</div>
            <div class="risk-tier__sub">Overall AI response risk assessment</div>
          </div>
        </div>
        ${confidenceHtml}
        <div class="checks-grid">${checksHtml}</div>
        ${radiologyHtml}
        ${claimsHtml}
        ${evidenceScoresHtml}
        ${conflictsHtml}
        ${reasonsHtml}
        ${citationsHtml}
        ${warningsHtml}
        ${suggestionsHtml}
      </div>
    </div>`;
}

/* ---------- Append user message ---------- */
function appendUserMessage(text, imageDataUrl) {
  const container = document.getElementById("chat-messages");
  if (!container) return;
  const div = document.createElement("div");
  div.className = "msg-user";
  const imgHtml = imageDataUrl
    ? `<div class="msg-user__image"><img src="${imageDataUrl}" alt="Uploaded scan"></div>`
    : "";
  div.innerHTML = `
    <div class="msg-user__inner">
      ${imgHtml}
      <div class="msg-user__bubble">${escapeHtml(text)}</div>
    </div>`;
  container.appendChild(div);
  scrollToBottom();
}

/* ---------- Typing indicator ---------- */
function appendTypingIndicator() {
  const container = document.getElementById("chat-messages");
  if (!container) return;
  const wrap = document.createElement("div");
  wrap.className = "msg-ai";
  wrap.id = "typing-indicator";
  wrap.innerHTML = `
    <div class="msg-ai__inner">
      <div class="msg-ai__header">
        <div class="msg-ai__avatar">${SVG.shield}</div>
        <span class="msg-ai__label">MedGemma Output</span>
      </div>
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  container.appendChild(wrap);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

/* ---------- Append AI message ---------- */
function appendAIMessage(data) {
  const container = document.getElementById("chat-messages");
  if (!container) return;
  const div = document.createElement("div");
  div.className = "msg-ai";
  const reportId = `report-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  div.innerHTML = `
    <div class="msg-ai__inner">
      <div class="msg-ai__header">
        <div class="msg-ai__avatar">${SVG.shield}</div>
        <span class="msg-ai__label">MedGemma Output</span>
      </div>
      <div class="msg-ai__response">${escapeHtml(data.final_response)}</div>
      ${buildAnalysisPanel(data)}
      ${data.analysis ? `
        <button class="report-download-btn" type="button" data-report-id="${reportId}">
          ${SVG.rag} Download Full Report
        </button>` : ""}
    </div>`;
  container.appendChild(div);
  if (data.analysis) {
    const reportButton = div.querySelector(`[data-report-id="${reportId}"]`);
    if (reportButton) {
      reportButton.addEventListener("click", () => downloadFullReport(data, reportButton));
    }
  }
  scrollToBottom();
}

/* ---------- Append system warning ---------- */
function appendSystemWarning(message, detail = "") {
  const container = document.getElementById("chat-messages");
  if (!container) return;
  const div = document.createElement("div");
  div.className = "msg-ai";
  const detailHtml = detail
    ? `<div class="system-warning__detail">${escapeHtml(detail)}</div>`
    : "";
  div.innerHTML = `
    <div class="msg-ai__inner">
      <div class="msg-ai__header">
        <div class="msg-ai__avatar">${SVG.warn}</div>
        <span class="msg-ai__label">System Warning</span>
      </div>
      <div class="system-warning">
        <div class="system-warning__title">${escapeHtml(message)}</div>
        ${detailHtml}
      </div>
    </div>`;
  container.appendChild(div);
  scrollToBottom();
}

/* ---------- Helpers ---------- */
function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function scrollToBottom() {
  const container = document.getElementById("chat-messages");
  if (container) container.scrollTop = container.scrollHeight;
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(Math.max(el.scrollHeight, 52), 140) + "px";
}

/* ============================================================
   NAVBAR
   ============================================================ */
function initNavbar() {
  const hamburger = document.getElementById("hamburger");
  const mobileMenu = document.getElementById("mobile-menu");
  if (!hamburger || !mobileMenu) return;
  hamburger.addEventListener("click", () => {
    mobileMenu.classList.toggle("open");
  });
}

/* ============================================================
   ASSISTANT PAGE
   ============================================================ */
function initAssistant() {
  const messagesEl        = document.getElementById("chat-messages");
  const inputEl           = document.getElementById("chat-input");
  const sendBtn           = document.getElementById("send-btn");
  const uploadBtn         = document.getElementById("upload-btn");
  const fileInput         = document.getElementById("file-input");
  const previewArea       = document.getElementById("image-preview-area");
  const previewImg        = document.getElementById("preview-img");
  const removeImgBtn      = document.getElementById("remove-img");
  const backendUrlEl      = document.getElementById("backend-url");
  const saveBackendUrlBtn = document.getElementById("save-backend-url");
  const backendStatusEl   = document.getElementById("backend-status");
  const chips             = document.querySelectorAll(".suggestion-chip");

  if (!messagesEl) return;

  let pendingImage = null;
  let isTyping = false;

  function setBackendStatus(text, state = "") {
    if (!backendStatusEl) return;
    backendStatusEl.textContent = text;
    backendStatusEl.classList.remove("ok", "error");
    if (state) backendStatusEl.classList.add(state);
  }

  function refreshBackendConfig() {
    const savedUrl = normalizeBackendUrl(localStorage.getItem(BACKEND_URL_STORAGE_KEY));
    if (backendUrlEl) backendUrlEl.value = savedUrl;
    setBackendStatus(getBackendModeLabel(), savedUrl ? "ok" : "");
  }

  refreshBackendConfig();

  if (saveBackendUrlBtn && backendUrlEl) {
    saveBackendUrlBtn.addEventListener("click", () => {
      const normalizedUrl = normalizeBackendUrl(backendUrlEl.value);
      if (normalizedUrl) {
        localStorage.setItem(BACKEND_URL_STORAGE_KEY, normalizedUrl);
      } else {
        localStorage.removeItem(BACKEND_URL_STORAGE_KEY);
      }
      refreshBackendConfig();
    });
  }

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      if (inputEl) {
        inputEl.value = chip.textContent.trim();
        autoResize(inputEl);
        updateSendState();
        inputEl.focus();
      }
    });
  });

  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = e => {
        pendingImage = e.target.result;
        if (previewImg)  previewImg.src = pendingImage;
        if (previewArea) previewArea.classList.add("visible");
        updateSendState();
      };
      reader.readAsDataURL(file);
    });
  }

  if (removeImgBtn) {
    removeImgBtn.addEventListener("click", () => {
      pendingImage = null;
      if (previewArea) previewArea.classList.remove("visible");
      if (fileInput)   fileInput.value = "";
      updateSendState();
    });
  }

  if (inputEl) {
    inputEl.addEventListener("input", () => {
      autoResize(inputEl);
      updateSendState();
    });
    inputEl.addEventListener("keydown", e => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  if (sendBtn) sendBtn.addEventListener("click", sendMessage);

  function setSendDisabled(disabled) {
    if (sendBtn) sendBtn.disabled = disabled;
  }

  function updateSendState() {
    const hasText = inputEl && inputEl.value.trim() !== "";
    setSendDisabled(!hasText && !pendingImage);
  }

  function sendMessage() {
    if (isTyping) return;
    const text = (inputEl ? inputEl.value.trim() : "") || "";
    if (!text && !pendingImage) return;

    const displayText = text || "[Medical image submitted]";
    const imgToSend = pendingImage;

    appendUserMessage(displayText, imgToSend);

    if (inputEl) { inputEl.value = ""; autoResize(inputEl); }
    pendingImage = null;
    if (previewArea) previewArea.classList.remove("visible");
    if (fileInput)   fileInput.value = "";

    const chipsWrap = document.querySelector(".suggestion-chips");
    if (chipsWrap) chipsWrap.style.display = "none";

    isTyping = true;
    setSendDisabled(true);
    appendTypingIndicator();

    getBackendResponse(text, imgToSend)
      .then(data => {
        removeTypingIndicator();
        setBackendStatus(getBackendModeLabel(), "ok");
        // Report export integration: preserve the user query alongside the
        // completed HalluGuard response for PDF generation.
        data.query = displayText;
        appendAIMessage(data);
      })
      .catch(error => {
        removeTypingIndicator();
        const apiError = error && error.payload && error.payload.error;
        if (apiError && apiError.code === "MEDGEMMA_UNAVAILABLE") {
          setBackendStatus("MedGemma unavailable", "error");
          appendSystemWarning(
            "MedGemma model unavailable. No medical answer generated.",
            apiError.message || "The backend refused to generate a response because the real model is unavailable."
          );
        } else {
          setBackendStatus("Backend unavailable", "error");
          appendSystemWarning(
            "Backend unavailable. No medical answer generated.",
            "The request could not reach the backend service. No substitute AI output was used."
          );
        }
      })
      .finally(() => {
        isTyping = false;
        updateSendState();
      });
  }
}

/* ============================================================
   INIT
   ============================================================ */
document.addEventListener("DOMContentLoaded", () => {
  initNavbar();
  initAssistant();
});
