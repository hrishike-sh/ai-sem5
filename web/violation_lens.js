(function (root) {
  'use strict';
  const ACTIVE = new Set(['queued', 'running', 'encoding', 'cancelling', 'cancel_requested', 'unknown']);
  const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const list = value => Array.isArray(value) ? value : [];
  const plain = value => typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '');
  function localURL(value, evidenceOrigin) {
    // The local file server has a separate random port. Trust only its configured origin.
    if (typeof value !== 'string' || value.startsWith('//') || /[\\\u0000-\u0020]/.test(value)) return '';
    try {
      const url = new URL(value, 'http://local.invalid');
      if(value.startsWith('/')&&url.origin==='http://local.invalid')return url.pathname+url.search+url.hash;
      const allowed=new URL(evidenceOrigin);
      return allowed.protocol==='http:'&&allowed.hostname==='127.0.0.1'&&allowed.origin===evidenceOrigin&&url.origin===allowed.origin&&!url.username&&!url.password?url.href:'';
    } catch { return ''; }
  }
  function timestamp(value) { return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value.toFixed(2) + ' s' : 'Time unavailable'; }
  function progress(value) { return typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null; }
  function authURL(value) { try { const url=new URL(value);return url.protocol==='https:'&&url.hostname==='auth.openai.com'&&url.pathname==='/api/accounts/authorize'&&!url.username&&!url.password&&!url.port?url.href:''; } catch { return ''; } }
  function runActive(run) { return ACTIVE.has(run?.status?.state || 'unknown'); }
  function priority(card) { return /helmet|wrong.?way|direction/i.test(card.rule || ''); }
  function visibleCards(cards, filter) { return list(cards).filter(card => filter === 'priority' ? priority(card) : filter === 'unreviewed' ? !['accept','reject'].includes(card.review?.decision) : true).slice().sort((a,b) => Number(priority(b)) - Number(priority(a))); }
  function disposition(review) { return review?.decision === 'accept' ? 'Human disposition: accepted for review record.' : review?.decision === 'reject' ? 'Human disposition: rejected for review record.' : 'Human disposition: not reviewed.'; }
  function ruleLabel(rule) { return String(rule || 'Unspecified candidate').replace(/[_-]+/g, ' ').replace(/^./, c => c.toUpperCase()); }
  function reviewPayload(view, cardID, decision, note) {
    if (!view?.run_id || !view.fingerprint || !list(view.cards).some(c => c.id === cardID)) throw Error('Reload the evidence before saving this decision.');
    if (!['accept','reject','reset'].includes(decision)) throw Error('Choose a valid human disposition.');
    return {id:view.run_id,card_id:cardID,fingerprint:view.fingerprint,decision,note:decision === 'reset' ? '' : String(note || '')};
  }
  function renderFrames(frames, evidenceOrigin) {
    return ['before','during','after'].map(role => {
      const frame = list(frames).find(f => (f.position || f.role) === role);
      const url = frame?.available === true ? localURL(frame.url,evidenceOrigin) : '';
      const label = role[0].toUpperCase() + role.slice(1);
      const content = url ? `<img src="${escapeHTML(url)}" alt="${label} evidence at ${escapeHTML(timestamp(frame.timestamp))}" loading="lazy">` : `<div class="frame-unavailable">${escapeHTML(frame?.reason || 'Evidence frame unavailable. No substitute frame is shown.')}</div>`;
      return `<figure>${content}<figcaption><strong>${label}</strong> · ${escapeHTML(timestamp(frame?.timestamp))}${frame?.id ? `<br>Frame ${escapeHTML(frame.id)}` : ''}</figcaption></figure>`;
    }).join('');
  }
  function renderCard(card, index, llm) {
    const facts = list(card.facts), missing = list(card.missing_facts);
    const factsHTML = facts.length ? `<ul>${facts.map(f => `<li>${escapeHTML(f.text || f.id || 'Recorded fact')}${f.value !== undefined ? ` <strong>${escapeHTML(plain(f.value))}</strong>` : ''}</li>`).join('')}</ul>` : '<p class="empty-facts">No supporting facts recorded.</p>';
    const missingHTML = missing.length ? `<ul>${missing.map(f => `<li>${escapeHTML(f.text || f.id || plain(f))}</li>`).join('')}</ul>` : '<p class="empty-facts">No missing facts listed. This does not certify completeness.</p>';
    const available = llm?.available === true && facts.length > 0;
    return `<article class="candidate" data-card-id="${escapeHTML(card.id)}" aria-labelledby="card-title-${index}">
      <div class="card-head"><div><h3 id="card-title-${index}">${escapeHTML(ruleLabel(card.rule))}</h3><p class="card-meta">${escapeHTML(timestamp(card.timestamp))} · Track ${escapeHTML(card.track_id ?? 'unavailable')}${card.synthetic===true?' · SYNTHETIC TEST EVIDENCE':''}</p>${card.recorded_verdict?`<p class="card-meta">Recorded pipeline label: ${escapeHTML(card.recorded_verdict)}</p>`:''}</div><span class="badge">Evidence status: ${escapeHTML(ruleLabel(card.status || 'unknown'))}</span></div>
      <div class="card-body"><p class="reason">${escapeHTML(card.reason || 'No reason recorded.')}</p><div class="facts-grid"><section><h4>Recorded facts</h4>${factsHTML}</section><section class="missing"><h4>Missing or uncertain</h4>${missingHTML}</section></div>
      <div class="evidence"><button type="button" class="secondary" data-action="frames">Load before / during / after</button><div class="frames" data-slot="frames"></div></div>
      <div class="review-decision"><h4>Your review</h4><p class="hint">Your disposition is saved separately. The machine finding and its evidence remain unchanged.</p><p class="disposition" data-slot="disposition">${escapeHTML(disposition(card.review))}</p><label for="note-${index}">Reviewer note (optional)</label><textarea id="note-${index}" data-slot="note" maxlength="4000" placeholder="What did you observe?">${escapeHTML(card.review?.note || '')}</textarea><div class="actions"><button type="button" data-action="accept">Accept candidate</button><button type="button" class="secondary" data-action="reject">Reject candidate</button><button type="button" class="secondary" data-action="reset">Reset review</button></div></div>
      <details class="explanation"><summary>Optional explanation of these facts</summary><p>${available ? 'Send the structured facts shown on this card to your ChatGPT plan for an optional explanation. Footage is not sent. This does not run a new detection or change the finding.' : 'No LLM explanation available. Deterministic facts remain available above.'}</p><p class="hint">${escapeHTML(llm?.message || 'No verified subscription adapter is connected. No credentials or API billing are required here.')}</p><button type="button" class="secondary" data-action="explain" ${available ? '' : 'disabled'}>Send these facts using my ChatGPT plan</button><div class="explanation-output" data-slot="explanation"></div></details>
      <p class="card-feedback" data-slot="feedback" role="status" aria-live="polite"></p></div></article>`;
  }
  const helpers = {escapeHTML,localURL,authURL,timestamp,progress,runActive,priority,visibleCards,disposition,reviewPayload,renderFrames,renderCard};
  if (typeof module !== 'undefined' && module.exports) module.exports = helpers;
  if (!root.document) return;

  const $ = id => root.document.getElementById(id);
  const token = root.document.querySelector('meta[name="traffic-token"]')?.content;
  const state = {config:null,media:null,run:null,review:null,llm:{available:false},busy:false,upload:null,generation:0,timer:null,cancelPending:false,library:[],before:null,libraryBusy:false,reviewBusy:false,drafts:new Map()};
  const storageKey = 'violation-lens-current-run';
  function notice(message, error = false) { $('notice').textContent = message; $('notice').hidden = !message; $('notice').classList.toggle('error', error); }
  function remember(id) { try { root.localStorage.setItem(storageKey, id); } catch { notice('This browser cannot remember the current run. Use Saved runs after a reload.'); } }
  async function api(path, data) {
    const options = data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-Traffic-Token':token},body:JSON.stringify(data)};
    const response = await root.fetch(path, options);
    let body; try { body = await response.json(); } catch { throw Error('The local server returned an unreadable response. Refresh to reconnect.'); }
    if (!response.ok) throw Error(body.error || body.message || `Request failed (${response.status}).`);
    return body;
  }
  function actions() {
    const active = state.run && runActive(state.run), locked = state.busy || !!state.upload || active;
    $('upload').disabled = !state.config || locked || !$('clip').files?.length || !$('rights').checked;
    $('clip').disabled = !!locked; $('rights').disabled = !!locked;
    $('start').disabled = !state.media || !!locked || state.config?.vehicle_tracking_ready !== true;
    $('cancel-run').hidden = !active; $('cancel-run').disabled = state.cancelPending;
    $('cancel-run').textContent = state.cancelPending ? 'Cancellation requested…' : 'Cancel detection';
    $('export-review').disabled = !state.review?.fingerprint || state.reviewBusy;
    for (const button of root.document.querySelectorAll('[data-open-run]')) button.disabled = !!state.busy || !!state.upload;
    const stage = state.review ? 'review' : state.media || state.run ? 'detect' : 'upload';
    for (const name of ['upload','detect','review']) { const el=$('step-'+name); if (name === stage) el.setAttribute('aria-current','step'); else el.removeAttribute('aria-current'); }
  }
  async function loadConfig() {
    try {
      state.config = await api('/api/config'); $('connection').textContent = 'Local server connected';
      const missing=[];
      if (state.config.vehicle_tracking_ready !== true) missing.push('Vehicle detection is unavailable. Install the verified YOLO11s model before starting detection.');
      if (state.config.helmet?.ready !== true) missing.push('Helmet detection is unavailable. Install the verified helmet model to assess helmet evidence.');
      $('setup').hidden = !missing.length; $('setup-message').textContent = missing.join(' ');
    } catch (error) { state.config=null; $('connection').textContent='Connection unavailable'; notice(error.message,true); }
    actions();
  }
  function clearReview() { state.review=null;state.drafts.clear(); $('cards').replaceChildren(); $('review-tools').hidden=true;$('provenance-notice').hidden=true; $('review-summary').textContent='Complete a detection run or open a saved run to inspect its evidence.'; }
  function showRun(run) {
    state.run=run; remember(run.id); $('source-name').textContent = typeof run.source === 'string' ? run.source : run.source?.name || 'Saved source';
    $('run-progress-wrap').hidden=false;
    const value=progress(run.status?.progress); $('run-state').textContent=ruleLabel(run.status?.state || 'unknown');
    if (value === null) $('run-progress').removeAttribute('value'); else $('run-progress').value=value;
    $('progress-label').textContent=value === null ? 'Progress unavailable' : Math.round(value)+'%';
    $('run-message').textContent=run.status?.message || (runActive(run) ? 'Waiting for the local worker…' : 'Run finished.');
    if (!runActive(run)) state.cancelPending=false;
    actions();
  }
  function schedulePoll(id, generation) { clearTimeout(state.timer); state.timer=root.setTimeout(()=>pollRun(id,generation),1800); }
  async function pollRun(id, generation) {
    try {
      const run=await api('/api/run?id='+encodeURIComponent(id));
      if (generation !== state.generation || state.run?.id !== id) return;
      showRun(run);
      if (runActive(run)) schedulePoll(id,generation);
      else if (run.status?.state === 'complete') { await loadReview(id,generation); loadLibrary(); }
      else { $('review-summary').textContent='Detection did not complete. Inspect its status above; saved evidence is not presented as a completed review.'; loadLibrary(); }
    } catch(error) { if(generation === state.generation) { $('run-message').textContent='Could not refresh status: '+error.message+' Use Refresh status to retry; your saved run is retained.'; $('connection').textContent='Status connection interrupted'; } }
  }
  async function openRun(id) {
    if(state.busy || state.upload) return;
    clearTimeout(state.timer); const generation=++state.generation; state.busy=true; state.media=null; state.cancelPending=false; clearReview(); notice(''); actions();
    try { const run=await api('/api/run?id='+encodeURIComponent(id)); if(generation!==state.generation)return; showRun(run); if(runActive(run))schedulePoll(id,generation); else if(run.status?.state==='complete')await loadReview(id,generation); }
    catch(error){if(generation===state.generation)notice('Could not open saved run: '+error.message,true);}
    finally{if(generation===state.generation){state.busy=false;actions();}}
  }
  function updateReviewSummary(shownCount) {
    const total=list(state.review?.cards).length, reviewed=list(state.review?.cards).filter(c=>['accept','reject'].includes(c.review?.decision)).length;
    $('review-summary').textContent=(state.review?.synthetic===true?'SYNTHETIC TEST EVIDENCE · ':'')+(total ? `${total} candidate${total===1?'':'s'} · ${reviewed} reviewed · ${shownCount} shown. Findings describe recorded evidence, not legal conclusions.` : 'No review candidates were recorded. This does not establish that no violation occurred.');
  }
  let shownCardCount=0;
  function drawCards() {
    const cards=visibleCards(state.review?.cards,$('rule-filter').value);
    $('provenance-notice').hidden=!state.review?.synthetic&&!state.review?.provenance_notice;
    $('provenance-notice').textContent=state.review?.provenance_notice||(state.review?.synthetic?'Synthetic test evidence. These fixtures demonstrate the review flow and do not establish real-world detection accuracy.':'');
    $('cards').innerHTML=cards.map((card,index)=>renderCard({...card,synthetic:card.synthetic===true||state.review?.synthetic===true,review:{...card.review,note:state.drafts.has(card.id)?state.drafts.get(card.id):card.review?.note}},index,state.llm)).join('');
    shownCardCount=cards.length;updateReviewSummary(shownCardCount);
  }
  async function loadReview(id,generation=state.generation) {
    $('review-summary').textContent='Loading recorded evidence…';
    try { const view=await api('/api/violation-review?id='+encodeURIComponent(id)); if(generation!==state.generation || state.run?.id!==id)return; if(view.run_id!==id || !view.fingerprint)throw Error('Evidence identity could not be verified.'); state.review=view; $('review-tools').hidden=false; drawCards(); actions(); }
    catch(error){if(generation===state.generation){notice('Could not load review: '+error.message,true);$('review-summary').textContent='Review unavailable. Refresh the run status to retry.';}}
  }
  function uploadClip() {
    const file=$('clip').files?.[0]; if(!file || !$('rights').checked || state.upload || state.busy || (state.run&&runActive(state.run)))return;
    if(!file.size){notice('Choose a nonempty video file.',true);return;}
    if(!/\.(mp4|mov|avi)$/i.test(file.name)){notice('Choose an MP4, MOV or AVI video clip. Still images cannot provide before-and-after motion evidence.',true);return;}
    clearTimeout(state.timer); ++state.generation; clearReview(); state.run=null;state.media=null;state.cancelPending=false; $('run-progress-wrap').hidden=true; notice('');
    const xhr=new root.XMLHttpRequest();state.upload=xhr;actions();$('upload-progress-wrap').hidden=false;$('upload-progress').value=0;$('upload-message').textContent='Uploading to the local server…';
    xhr.open('POST','/api/upload?name='+encodeURIComponent(file.name));xhr.setRequestHeader('X-Traffic-Token',token);
    xhr.upload.onprogress=event=>{if(event.lengthComputable){const amount=Math.round(event.loaded/event.total*100);$('upload-progress').value=amount;$('upload-message').textContent=`Uploading ${amount}%`;}};
    xhr.onload=()=>{try{const media=JSON.parse(xhr.responseText);if(xhr.status<200||xhr.status>=300)throw Error(media.error||'Upload failed.');if(!media.id||!media.source_revision)throw Error('Uploaded source identity is missing. Upload the clip again.');state.media=media;$('source-name').textContent=media.name||file.name;$('upload-message').textContent='Clip uploaded. Ready for detection.';}catch(error){notice(error.message,true);$('upload-message').textContent='Upload could not be completed.';}};
    xhr.onerror=()=>{notice('Upload interrupted. Check the local server and try again.',true);$('upload-message').textContent='Upload interrupted.';};
    xhr.onabort=()=>{$('upload-message').textContent='Upload cancelled. Choose Upload clip to try again.';};
    xhr.onloadend=()=>{state.upload=null;actions();};xhr.send(file);
  }
  async function startRun() {
    if($('start').disabled)return;
    const generation=++state.generation;state.busy=true;clearReview();notice('');actions();
    try { const run=await api('/api/job',{media:state.media.id,source_revision:state.media.source_revision,settings:state.config.defaults||{},masks:[],ai:true});if(generation!==state.generation)return;if(!run.id)throw Error('The job response has no run identifier.');showRun(run);schedulePoll(run.id,generation); }
    catch(error){notice('Detection submission could not be confirmed: '+error.message+' Check Saved runs before submitting again to avoid a duplicate job.',true);await loadLibrary();}
    finally{if(generation===state.generation){state.busy=false;actions();}}
  }
  async function cancelRun() {
    if(!state.run||!runActive(state.run)||state.cancelPending)return;
    const id=state.run.id,generation=state.generation;state.cancelPending=true;actions();
    try{await api('/api/cancel',{id});if(generation===state.generation){$('run-message').textContent='Cancellation requested. Waiting for the worker to stop…';schedulePoll(id,generation);}}
    catch(error){if(generation===state.generation){state.cancelPending=false;notice('Cancellation could not be confirmed: '+error.message,true);actions();}}
  }
  async function cardAction(event) {
    const button=event.target.closest('button[data-action]'),article=button?.closest('[data-card-id]');if(!button||!article||button.disabled)return;
    const id=article.dataset.cardId,card=list(state.review?.cards).find(c=>c.id===id);if(!card)return;
    const generation=state.generation,view=state.review,action=button.dataset.action,feedback=article.querySelector('[data-slot="feedback"]');
    const same=()=>generation===state.generation&&state.review?.run_id===view.run_id&&state.review?.fingerprint===view.fingerprint&&article.isConnected;
    const payload={id:view.run_id,card_id:id,fingerprint:view.fingerprint};
    const buttons=[...article.querySelectorAll('button')];const disabled=buttons.map(b=>b.disabled);buttons.forEach(b=>b.disabled=true);feedback.textContent='Working…';
    try {
      if(action==='frames') { const result=await api('/api/violation-frames',payload);if(!same())return;if(result.run_id!==view.run_id||result.card_id!==id||result.fingerprint!==view.fingerprint)throw Error('Frame evidence does not match this review. Reload evidence.');article.querySelector('[data-slot="frames"]').innerHTML=renderFrames(result.frames,state.config?.evidence_origin);feedback.textContent='Evidence frames loaded. Captions show actual sampled times.'; }
      else if(action==='explain') {
        if(!state.llm.available)throw Error('No LLM explanation available. Deterministic facts remain available.');
        const result=await api('/api/violation-explain',{...payload,consent:true});if(!same())return;if(result.card_id&&result.card_id!==id)throw Error('Explanation belongs to different evidence. Reload this review.');
        article.querySelector('[data-slot="explanation"]').textContent=result.explanation||result.text||result.message||'No LLM explanation available. Deterministic facts remain available.';feedback.textContent='Optional explanation response received; the machine finding is unchanged.';
      } else {
        const result=await api('/api/violation-decision',reviewPayload(view,id,action,article.querySelector('[data-slot="note"]').value));if(!same())return;
        if(result.run_id!==view.run_id||result.fingerprint!==view.fingerprint)throw Error('Review changed on the server. Reload evidence before continuing.');
        const updated=list(result.cards).find(c=>c.id===id);if(!updated)throw Error('Saved disposition could not be confirmed. Reload evidence.');
        // Only update the selected card: unsaved notes on other cards remain intact.
        const localCard=list(state.review.cards).find(c=>c.id===id);if(localCard)localCard.review=updated.review;state.drafts.delete(id);
        updateReviewSummary(shownCardCount);
        article.querySelector('[data-slot="disposition"]').textContent=disposition(updated.review);if(action==='reset')article.querySelector('[data-slot="note"]').value='';feedback.textContent='Human disposition saved. Recorded machine evidence is unchanged.';
      }
    } catch(error){if(same())feedback.textContent=error.message;}
    finally{if(same())buttons.forEach((b,index)=>b.disabled=disabled[index]);}
  }
  async function exportReview() {
    const view=state.review,generation=state.generation;if(!view||state.reviewBusy)return;state.reviewBusy=true;actions();
    try{const result=await api('/api/violation-export',{id:view.run_id,fingerprint:view.fingerprint});if(generation!==state.generation)return;const url=localURL(result.url,state.config?.evidence_origin);if(!url)throw Error('The export download address is invalid.');const link=root.document.createElement('a');link.href=url;link.download=result.filename||'violation-review.json';root.document.body.append(link);link.click();link.remove();notice('Review export prepared with recorded evidence and separate human dispositions.');}
    catch(error){if(generation===state.generation)notice('Export failed: '+error.message,true);}
    finally{state.reviewBusy=false;actions();}
  }
  async function loadLibrary(more=false) {
    if(state.libraryBusy)return;state.libraryBusy=true;$('reload-library').disabled=true;$('more-runs').disabled=true;
    try{const page=await api('/api/run-library'+(more&&state.before?'?before='+encodeURIComponent(state.before):''));state.library=more?state.library.concat(list(page.runs)):list(page.runs);state.before=page.next_before;
      $('saved-runs').innerHTML=state.library.map(run=>`<div class="saved-run"><div><p>${escapeHTML(typeof run.source==='string'?run.source:run.source?.name||'Saved source')}</p><small>${escapeHTML(typeof run.status==='string'?run.status:run.status?.state||'unknown')} · ${escapeHTML(run.id)}</small></div><button class="secondary" type="button" data-open-run="${escapeHTML(run.id)}">Open run</button></div>`).join('');
      $('library-message').textContent=state.library.length?'Open a run to restore its progress and review.':'No saved runs yet.';$('more-runs').hidden=!page.has_more||!page.next_before;
    }catch(error){$('library-message').textContent='Could not load saved runs: '+error.message;}
    finally{state.libraryBusy=false;$('reload-library').disabled=false;$('more-runs').disabled=false;actions();}
  }
  async function init() {
    $('clip').addEventListener('change',()=>{$('selected-file').textContent=$('clip').files?.[0]?.name||'No clip selected.';$('rights').checked=false;actions();});
    $('rights').addEventListener('change',actions);$('upload').addEventListener('click',uploadClip);$('cancel-upload').addEventListener('click',()=>state.upload?.abort());$('start').addEventListener('click',startRun);$('cancel-run').addEventListener('click',cancelRun);$('recheck').addEventListener('click',loadConfig);
    $('refresh-run').addEventListener('click',()=>{if(state.run){clearTimeout(state.timer);pollRun(state.run.id,state.generation);}});
    $('refresh-review').addEventListener('click',()=>{if(state.run)loadReview(state.run.id);});$('rule-filter').addEventListener('change',drawCards);$('cards').addEventListener('click',cardAction);$('cards').addEventListener('input',event=>{if(event.target.matches('[data-slot="note"]'))state.drafts.set(event.target.closest('[data-card-id]').dataset.cardId,event.target.value);});$('export-review').addEventListener('click',exportReview);
    $('connect-chatgpt').addEventListener('click',()=>{$('auth-confirmation').hidden=false;$('approve-auth').focus();});$('dismiss-auth').addEventListener('click',()=>{$('auth-confirmation').hidden=true;$('connect-chatgpt').focus();});
    $('approve-auth').addEventListener('click',async()=>{if($('approve-auth').disabled)return;$('approve-auth').disabled=true;$('auth-feedback').textContent='Preparing ChatGPT sign-in…';try{const result=await api('/api/violation-auth/start',{approve:true});const url=authURL(result.url);if(!url)throw Error('The server did not return a verified ChatGPT sign-in address.');root.location.assign(url);}catch(error){$('auth-feedback').textContent='Could not start sign-in: '+error.message;$('approve-auth').disabled=false;}});
    $('reload-library').addEventListener('click',()=>loadLibrary());$('more-runs').addEventListener('click',()=>loadLibrary(true));$('saved-runs').addEventListener('click',event=>{const button=event.target.closest('[data-open-run]');if(button&&!button.disabled)openRun(button.dataset.openRun);});
    await Promise.all([loadConfig(),loadLibrary(),api('/api/violation-llm').then(result=>{state.llm=result;}).catch(()=>{state.llm={available:false,message:'No verified subscription adapter is connected.'};})]);
    $('subscription-status').textContent=state.llm.available===true?'Using ChatGPT plan':'No LLM explanation available';$('subscription-message').textContent=state.llm.message||'Deterministic evidence remains available without a ChatGPT connection.';$('connect-chatgpt').hidden=state.llm.available===true;
    let id;try{id=root.localStorage.getItem(storageKey);}catch{}if(id)await openRun(id);actions();
    try{const url=new URL(root.location.href),signin=url.searchParams.get('signin');if(signin){url.searchParams.delete('signin');root.history.replaceState(null,'',url.pathname+url.search+url.hash);if(signin==='connected')notice('ChatGPT sign-in returned. '+(state.llm.available?'Your ChatGPT plan is connected. Choose a card to send its structured facts.':'The connection is not available yet. '+(state.llm.message||'Reload to check its status.')));else if(signin==='failed')notice('ChatGPT sign-in did not complete. Deterministic evidence remains available; choose Connect ChatGPT to try again.',true);}}catch{}
  }
  init();
})(typeof window !== 'undefined' ? window : globalThis);
