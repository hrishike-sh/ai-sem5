(function(root){
  'use strict';
  const common=typeof module!=='undefined'&&module.exports?require('./violation_lens.js'):root.ViolationLens;
  const esc=common.escapeHTML,displayText=common.escapeReviewText;
  const list=value=>Array.isArray(value)?value:[];
  function renderEvidenceTable(view){
    const table=view?.evidence_table;
    if(!table)return '<p class="muted">Prepare numbered subjects to assemble the evidence table.</p>';
    const labels={suspected_issue:'Suspected issue',observed:'Observed',cannot_assess:'Needs review'};
    const rows=list(table.rows).map(row=>{
      const plate=row.plate;
      const candidates=list(plate?.candidates);
      const correction=plate&&row.subject_id===row.vehicle_id&&candidates.length===1?`<label>Correct reading <input maxlength="16" data-plate-vehicle="${esc(row.vehicle_id)}" data-plate-id="${esc(candidates[0].plate_id||'')}" value="${esc(plate.text||'')}" autocomplete="off"></label><button type="button" data-plate-save="${esc(row.vehicle_id)}" data-plate-id="${esc(candidates[0].plate_id||'')}">Save correction</button>`:'';
      const plateCell=plate?`<strong>${esc(plate.text||'Unreadable / unavailable')}</strong><small>${displayText(plate.corrected?'Reviewer correction; original OCR retained':plate.status==='low_confidence'?'Low-confidence detection':plate.status.replace(/_/g,' '))}</small>${plate.corrected?`<small>Original OCR: ${esc(candidates.map(c=>c.text||'(unreadable)').join(', '))}</small>`:''}${correction}`:'<span class="muted">No associated plate reading</span>';
      const scores=list(row.raw_model_scores).map(s=>`<div>${esc(s.source.replace(/_/g,' '))}: ${displayText(s.label)} <code title="${esc(Number.isFinite(s.model_score)?'Exact raw model score: '+String(s.model_score):'Score unavailable')}">${esc(Number.isFinite(s.model_score)?s.model_score.toFixed(3):'unavailable')}</code></div>`).join('')||'Not supplied';
      const refs=list(row.evidence_ids).map(id=>{
        const frame=list(view.frames).find(f=>f.id===id);
        return `<span title="${esc(id)}">${esc(frame?.role==='context'?'Context':row.subject_id+' crop')}</span>`;
      }).join(' · ');
      const issues=list(row.suspected_issues).map(i=>`<p>${displayText(i.text)}</p>`).join('')||'<span class="muted">No supported suspicion listed</span>';
      return `<tr><th scope="row">${esc(row.subject_id)}<small>${esc(row.kind)}</small>${row.vehicle_id&&row.vehicle_id!==row.subject_id?`<small>Apparent vehicle: ${esc(row.vehicle_id)}</small>`:''}${row.person_ids?.length&&row.kind!=='person'?`<small>Apparent riders: ${row.person_ids.map(esc).join(', ')}</small>`:''}</th><td>${plateCell}</td><td><span class="status">${esc(labels[row.assessment]||'Needs review')}</span>${issues}<details><summary>Observations</summary>${list(row.observations).map(o=>`<p>${displayText(o)}</p>`).join('')||'<p>No completed observations.</p>'}</details></td><td>${scores}</td><td>${refs}</td><td>${esc(row.review_status)}<small>Review disposition only</small></td></tr>`;
    }).join('');
    return `<p>${esc(table.summary?.suspected_issue_count||0)} distinct suspected issues · ${esc(table.summary?.cannot_assess_count||0)} subjects need review</p><div class="evidence-table-wrap"><table class="evidence-table"><thead><tr><th>Subject / vehicle</th><th>Plate reading</th><th>Assessment &amp; suspected issue</th><th>Raw model scores</th><th>Evidence</th><th>Review status</th></tr></thead><tbody>${rows||'<tr><td colspan="6">No numbered subjects.</td></tr>'}</tbody></table></div><p class="muted">${esc(table.score_notice)}</p><details><summary>Scope and limitations</summary>${list(table.limitations).map(t=>`<p>${displayText(t)}</p>`).join('')}</details>`;
  }
  const api={renderEvidenceTable};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ViolationEvidence=api;
})(typeof globalThis!=='undefined'?globalThis:this);
