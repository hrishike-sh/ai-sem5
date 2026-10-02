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
  // Worker status.progress is a normalized fraction; the HTML progress element uses percent.
  function progress(value) { return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value * 100 : null; }
  function authURL(value) { try { const url=new URL(value);return url.protocol==='https:'&&url.hostname==='auth.openai.com'&&url.pathname==='/api/accounts/authorize'&&!url.username&&!url.password&&!url.port?url.href:''; } catch { return ''; } }
  function runActive(run) { return ACTIVE.has(run?.status?.state || 'unknown'); }
  function priority(card) { return /helmet|wrong.?way|direction/i.test(card.rule || ''); }
  function visibleCards(cards, filter) { return list(cards).filter(card => filter === 'priority' ? priority(card) : filter === 'unreviewed' ? !['accept','reject'].includes(card.review?.decision) : true).slice().sort((a,b) => Number(priority(b)) - Number(priority(a))); }
  function disposition(review) { return review?.decision === 'accept' ? 'Human disposition: accepted for review record.' : review?.decision === 'reject' ? 'Human disposition: rejected for review record.' : 'Human disposition: not reviewed.'; }
  function reviewSummary(view, shownCount) {
    const cards=list(view?.cards),total=cards.length,reviewed=cards.filter(c=>['accept','reject'].includes(c.review?.decision)).length,supported=cards.filter(c=>c.status==='supported').length;
    const truncated=view?.limits?.truncated===true;
    const count=value=>Number.isSafeInteger(value)&&value>=0?value:null;
    const nonapplicable=count(view?.omitted_nonapplicable_count??view?.limits?.nonapplicable_observations)||0;
    let summary=view?.synthetic===true?'SYNTHETIC TEST EVIDENCE · ':'';
    summary+=total||truncated?`${total} review card${total===1?'':'s'} returned · ${supported} supported candidate${supported===1?'':'s'} in this returned set · ${reviewed} reviewed · ${shownCount} shown by current filter. Findings describe recorded evidence, not legal conclusions.`:'0 supported candidates. No applicable review candidates were recorded. This does not establish that no violation occurred.';
    if(truncated) {
      const maximum=count(view.limits.max_cards),omitted=count(view.limits.omitted_observations);
      summary+=` Review display is capped${maximum===null?'':` at ${maximum} cards`}. ${omitted===null?'Additional':omitted+' additional'} recorded per-frame observations were omitted by the server limit; this is not a count of unique incidents. Supported counts apply only to the returned review cards, not the full clip.`;
    }
    if(nonapplicable>0)summary+=` ${truncated?'Separately, ':''}${nonapplicable} explicitly non-applicable observations omitted.`;
    return summary;
  }
  function ruleLabel(rule) { return String(rule || 'Unspecified candidate').replace(/[_-]+/g, ' ').replace(/^./, c => c.toUpperCase()); }
  function reviewPayload(view, cardID, decision, note) {
    if (!view?.run_id || !view.fingerprint || !list(view.cards).some(c => c.id === cardID)) throw Error('Reload the evidence before saving this decision.');
    if (!['accept','reject','reset'].includes(decision)) throw Error('Choose a valid human disposition.');
    return {id:view.run_id,card_id:cardID,fingerprint:view.fingerprint,decision,note:decision === 'reset' ? '' : String(note || '')};
  }
  function readiness(config) {
    if(config?.violation_readiness) {
      const value=config.violation_readiness;
      return {ready:value.ready===true,blockers:list(value.blockers).map(v=>plain(v?.message||v)),warnings:list(value.warnings).map(v=>plain(v?.message||v))};
    }
    return {ready:config?.vehicle_tracking_ready===true,blockers:config?.vehicle_tracking_ready===true?[]:['Vehicle detection is unavailable. Install the verified YOLO11s model before starting detection.'],warnings:config?.helmet?.ready===true?[]:['Helmet detection is unavailable. Install the verified helmet model to assess helmet evidence.']};
  }
  function renderFacts(facts, emptyMessage) {
    return list(facts).length?`<ul>${facts.map(f=>`<li>${escapeHTML(f.text||f.id||'Recorded fact')}${f.value!==undefined?` <strong>${escapeHTML(plain(f.value))}</strong>`:''}</li>`).join('')}</ul>`:`<p class="empty-facts">${escapeHTML(emptyMessage)}</p>`;
  }
  function validatePreparation(prepared, view, cardID, evidenceOrigin) {
    if(!prepared?.preparation_id||prepared.run_id!==view.run_id||prepared.card_id!==cardID||prepared.fingerprint!==view.fingerprint||!['images_and_facts','facts_only'].includes(prepared.mode))throw Error('Prepared evidence does not match this card. Prepare it again before sending.');
    const frames=list(prepared.frames).filter(frame=>frame.available===true);
    if(prepared.mode==='images_and_facts'&&(!frames.length||frames.some(frame=>!frame.id||!localURL(frame.url,evidenceOrigin))))throw Error('Prepared image previews are unavailable or invalid. Retry preparation, or explicitly choose structured facts only.');
    return prepared;
  }
  function preparedPayload(prepared, view, cardID) {
    if(!prepared?.preparation_id||prepared.run_id!==view.run_id||prepared.card_id!==cardID||prepared.fingerprint!==view.fingerprint||!['images_and_facts','facts_only'].includes(prepared.mode))throw Error('Prepare this card again before sending.');
    return {id:view.run_id,card_id:cardID,fingerprint:view.fingerprint,preparation_id:prepared.preparation_id,mode:prepared.mode,consent:true};
  }
  function renderPreparation(prepared, evidenceOrigin) {
    const images=prepared.mode==='images_and_facts';
    const frames=list(prepared.frames).map(frame=>{
      const url=frame.available===true?localURL(frame.url,evidenceOrigin):'';
      return `<figure>${url&&images?`<img data-prepared-frame src="${escapeHTML(url)}" alt="Prepared ${escapeHTML(frame.role||frame.position||'evidence')} frame at ${escapeHTML(timestamp(frame.timestamp))}">`:`<div class="frame-unavailable">${escapeHTML(frame.reason||'This frame will not be sent.')}</div>`}<figcaption><strong>${escapeHTML(ruleLabel(frame.role||frame.position||'Frame'))}</strong> · ${escapeHTML(timestamp(frame.timestamp))}<br>Frame ${escapeHTML(frame.id||'unavailable')}</figcaption></figure>`;
    }).join('');
    return `<div class="prepared-evidence"><h4>Review exactly what will be sent</h4><p>${images?'These server-prepared images and the structured facts below will be sent to ChatGPT. The full clip will not be sent.':'Structured facts only. No images will be sent to ChatGPT.'}</p>${frames?`<div class="frames">${frames}</div>`:''}<div class="facts-grid"><section><h4>Prepared facts</h4>${renderFacts(prepared.facts,'No supporting facts included.')}</section><section class="missing"><h4>Missing or uncertain</h4>${renderFacts(prepared.missing_facts,'No missing facts listed. This does not certify completeness.')}</section></div>${list(prepared.limitations).length?`<ul class="preparation-limitations">${prepared.limitations.map(item=>`<li>${escapeHTML(item)}</li>`).join('')}</ul>`:''}<p data-slot="preview-status" class="hint">${images?'Loading the exact prepared image previews…':'Facts-only preparation is ready for your consent.'}</p><button type="button" data-action="send-prepared" ${images?'disabled':''}>${images?'Send selected frames and facts to ChatGPT':'Send prepared structured facts only to ChatGPT'}</button></div>`;
  }
  function renderExplanation(result, prepared, saved=false) {
    const text=result.explanation||result.text||result.message||'No LLM explanation available. Deterministic facts remain available.';
    const observations=list(result.visual_observations),frameIDs=new Set(list(prepared?.frames).filter(frame=>frame.available===true).map(frame=>frame.id));
    const valid=observations.every(item=>frameIDs.has(item.frame_id));
    return `<div class="explanation-result"><h4>${saved?'Saved explanation':'Optional ChatGPT explanation'}${result.model?` · ${escapeHTML(result.model)}`:''}</h4>${saved?'<p class="hint">Restored from this run. No new ChatGPT request was sent.</p>':''}<p class="explanation-text">${escapeHTML(text)}</p>${result.vision_mode==='facts_only'||result.mode==='facts_only'?'<p class="hint">Structured facts only. No images were sent for this response.</p>':''}${observations.length?`<section class="visual-observations"><h4>Model observations of selected images</h4><p class="hint">These observations can be wrong. They are separate from the recorded detector finding and your review decision.</p>${valid?`<ul>${observations.map(item=>`<li>${escapeHTML(item.text||item.observation||'No observation text returned.')} <small>Frame ${escapeHTML(item.frame_id)}</small></li>`).join('')}</ul>`:'<p>Image observations could not be matched to the prepared frames and are not displayed. Prepare this evidence again before retrying.</p>'}${result.visual_notice?`<p class="hint">${escapeHTML(result.visual_notice)}</p>`:''}</section>`:''}${saved&&list(result.images_sent).length?`<details class="saved-image-details"><summary>Images used for this saved response</summary><ul>${result.images_sent.map(frame=>`<li>Frame ${escapeHTML(frame.id)} · ${escapeHTML(timestamp(frame.timestamp))} · ${escapeHTML(ruleLabel(frame.role||'Image'))}</li>`).join('')}</ul></details>`:''}</div>`;
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
  function renderCard(card, index, llm, savedExplanation) {
    const facts = list(card.facts), missing = list(card.missing_facts);
    const factsHTML = renderFacts(facts,'No supporting facts recorded.');
    const missingHTML = renderFacts(missing,'No missing facts listed. This does not certify completeness.');
    const available = llm?.available === true && facts.length > 0;
    return `<article class="candidate" data-card-id="${escapeHTML(card.id)}" aria-labelledby="card-title-${index}">
      <div class="card-head"><div><h3 id="card-title-${index}">${escapeHTML(ruleLabel(card.rule))}</h3><p class="card-meta">${escapeHTML(timestamp(card.timestamp))} · Track ${escapeHTML(card.track_id ?? 'unavailable')}${card.synthetic===true?' · SYNTHETIC TEST EVIDENCE':''}</p>${card.recorded_verdict?`<p class="card-meta">Recorded pipeline label: ${escapeHTML(card.recorded_verdict)}</p>`:''}</div><span class="badge">Evidence status: ${escapeHTML(ruleLabel(card.status || 'unknown'))}</span></div>
      <div class="card-body"><p class="reason">${escapeHTML(card.reason || 'No reason recorded.')}</p><div class="facts-grid"><section><h4>Recorded facts</h4>${factsHTML}</section><section class="missing"><h4>Missing or uncertain</h4>${missingHTML}</section></div>
      <div class="evidence"><button type="button" class="secondary" data-action="frames">Load before / during / after</button><div class="frames" data-slot="frames"></div></div>
      <details class="explanation" ${savedExplanation?'open':''}><summary>${savedExplanation?'Saved ChatGPT explanation':'Optional ChatGPT explanation'}</summary><p>${available ? 'Prepare a preview of the selected evidence before deciding whether to send it to your ChatGPT plan. An explanation does not run detection or change the recorded finding.' : 'No new LLM explanation available. Deterministic facts remain available above.'}</p><p class="hint">${escapeHTML(llm?.message || 'No verified subscription adapter is connected. No paid API fallback is used.')}</p><div class="actions"><button type="button" class="secondary" data-action="prepare" ${available?'':'disabled'}>Preview frames and facts for ChatGPT</button><button type="button" class="secondary" data-action="explain-facts" ${available?'':'disabled'}>Send structured facts only to ChatGPT</button></div><p class="hint">The facts-only option sends the recorded facts and missing context shown above, without images.</p><div data-slot="preparation"></div><div class="explanation-output" data-slot="explanation" role="status">${savedExplanation?renderExplanation(savedExplanation,{frames:list(savedExplanation.images_sent).map(frame=>({...frame,available:true}))},true):''}</div></details>
      <div class="review-decision"><h4>Your review</h4><p class="hint">Your disposition is saved separately. The machine finding and its evidence remain unchanged.</p><p class="disposition" data-slot="disposition">${escapeHTML(disposition(card.review))}</p><label for="note-${index}">Reviewer note (optional; saved with Accept or Reject)</label><textarea id="note-${index}" data-slot="note" maxlength="4000" placeholder="What did you observe?">${escapeHTML(card.review?.note || '')}</textarea><div class="actions"><button type="button" data-action="accept">Accept candidate</button><button type="button" class="secondary" data-action="reject">Reject candidate</button><button type="button" class="secondary" data-action="reset">Reset review</button></div></div>
      <p class="card-feedback" data-slot="feedback" role="status" aria-live="polite"></p></div></article>`;
  }
  const helpers = {escapeHTML,localURL,authURL,timestamp,progress,runActive,priority,visibleCards,disposition,reviewSummary,reviewPayload,readiness,validatePreparation,preparedPayload,renderPreparation,renderExplanation,renderFrames,renderCard};
  if (typeof module !== 'undefined' && module.exports) module.exports = helpers;
  if (!root.document) return;

  const $ = id => root.document.getElementById(id);
  const token = root.document.querySelector('meta[name="traffic-token"]')?.content;
  const state = {config:null,media:null,run:null,review:null,llm:{available:false},busy:false,upload:null,generation:0,timer:null,cancelPending:false,library:[],before:null,libraryBusy:false,reviewBusy:false,pendingCards:0,drafts:new Map(),preparations:new Map()};
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
    $('start').disabled = !state.media || !!locked || !readiness(state.config).ready;
    $('start').textContent=state.media&&state.run&&!active?'Run detection again':'Run detection';
    $('cancel-upload').hidden=!state.upload;
    $('next-action').textContent=state.upload?'Wait for the upload to finish before starting detection.':active?'Detection is in progress. You can cancel it or return to the saved run later.':state.busy?'Opening or submitting your run…':state.review?(state.media?'Inspect the candidate evidence below, then save your review decisions.':'Saved run restored for review. To start a new detection, choose and upload the clip again.'):state.media?(readiness(state.config).ready?'Your clip is ready. Choose Run detection.':'Resolve the setup items above, then choose Recheck setup.'):state.run?'Choose a new clip to start another run, or reopen a completed run below.':'Choose and upload a permitted clip to begin.';
    $('cancel-run').hidden = !active; $('cancel-run').disabled = state.cancelPending;
    $('cancel-run').textContent = state.cancelPending ? 'Cancellation requested…' : 'Cancel detection';
    $('export-review').disabled = !state.review?.fingerprint || state.reviewBusy || state.pendingCards>0;
    $('refresh-review').disabled=state.pendingCards>0; $('rule-filter').disabled=state.pendingCards>0;
    $('refresh-chatgpt').disabled=state.pendingCards>0;
    for (const button of root.document.querySelectorAll('[data-open-run]')) button.disabled = !!state.busy || !!state.upload;
    const stage = state.review ? 'review' : state.media || state.run ? 'detect' : 'upload';
    for (const name of ['upload','detect','review']) { const el=$('step-'+name); if (name === stage) el.setAttribute('aria-current','step'); else el.removeAttribute('aria-current'); }
  }
  async function loadConfig() {
    notice('');
    try {
      state.config = await api('/api/config'); $('connection').textContent = 'Local server connected';
      const ready=readiness(state.config),requiresSetup=!ready.ready||ready.blockers.length>0;
      $('setup').hidden = !requiresSetup&&!ready.warnings.length;
      $('setup-title').textContent=requiresSetup?'Local model setup needed':'Detection available with limitations';
      $('setup-message').textContent=[...ready.blockers,...ready.warnings].join(' ')||(requiresSetup?'Detection is not ready. Recheck setup for the current requirements.':'');
      $('setup-help').textContent=requiresSetup?'Model weights are not bundled. Follow the Models section in the project README to resolve the required setup items, then recheck setup. Restart the server if the setup instructions require it.':'Detection is available. Keep these scope limitations in mind when reviewing candidates.';
    } catch (error) { state.config=null; $('connection').textContent='Connection unavailable';$('setup').hidden=false;$('setup-title').textContent='Local server connection needed';$('setup-message').textContent=error.message;$('setup-help').textContent='Check that the app server is running, then choose Recheck setup. Existing saved runs are kept on the server.'; notice(error.message,true); }
    actions();
  }
  function clearReview() { state.review=null;state.drafts.clear();state.preparations.clear(); $('cards').replaceChildren(); $('review-tools').hidden=true;$('review-source').hidden=true;$('provenance-notice').hidden=true; $('review-summary').textContent='Complete a detection run or open a saved run to inspect its evidence.'; }
  function showRun(run) {
    $('connection').textContent='Local server connected';
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
    clearTimeout(state.timer); const generation=++state.generation; state.busy=true; state.media=null;state.run=null;$('run-progress-wrap').hidden=true;$('source-name').textContent='Opening saved source…'; state.cancelPending=false; clearReview(); notice(''); actions();
    try { const run=await api('/api/run?id='+encodeURIComponent(id)); if(generation!==state.generation)return; showRun(run); if(runActive(run))schedulePoll(id,generation); else if(run.status?.state==='complete')await loadReview(id,generation); }
    catch(error){if(generation===state.generation){$('source-name').textContent='Saved source could not be opened.';notice('Could not open saved run: '+error.message+' Refresh the saved-run list and try opening it again.',true);}}
    finally{if(generation===state.generation){state.busy=false;actions();}}
  }
  function updateReviewSummary(shownCount) {
    $('review-summary').textContent=reviewSummary(state.review,shownCount);
  }
  let shownCardCount=0;
  function drawCards() {
    state.preparations.clear();
    const cards=visibleCards(state.review?.cards,$('rule-filter').value);
    $('provenance-notice').hidden=!state.review?.synthetic&&!state.review?.provenance_notice;
    $('provenance-notice').textContent=state.review?.provenance_notice||(state.review?.synthetic?'Synthetic test evidence. These fixtures demonstrate the review flow and do not establish real-world detection accuracy.':'');
    $('review-source').hidden=false;$('review-source').textContent=`Source: ${typeof state.run?.source==='string'?state.run.source:state.run?.source?.name||'Unavailable'} · Run: ${state.review?.run_id||'Unavailable'}`;
    $('cards').innerHTML=cards.map((card,index)=>renderCard({...card,synthetic:card.synthetic===true||state.review?.synthetic===true,review:{...card.review,note:state.drafts.has(card.id)?state.drafts.get(card.id):card.review?.note}},index,state.llm,state.review?.explanations?.[card.id])).join('');
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
  function watchPreparedImages(article, prepared) {
    const record={prepared,ready:prepared.mode==='facts_only'};state.preparations.set(prepared.card_id,record);
    const button=article.querySelector('[data-action="send-prepared"]'),status=article.querySelector('[data-slot="preview-status"]');
    if(record.ready){button.disabled=false;return;}
    const images=[...article.querySelectorAll('img[data-prepared-frame]')];
    const check=()=>{
      if(state.preparations.get(prepared.card_id)!==record||!article.isConnected)return;
      const failed=images.some(img=>img.complete&&!img.naturalWidth);
      record.ready=images.length===list(prepared.frames).filter(frame=>frame.available===true).length&&images.every(img=>img.complete&&img.naturalWidth>0);
      button.disabled=!record.ready;
      status.textContent=failed?'One or more prepared images could not be displayed. Retry preparation, or explicitly choose structured facts only.':record.ready?'All prepared image previews are loaded. Review them before sending.':'Loading the exact prepared image previews…';
    };
    images.forEach(img=>{img.addEventListener('load',check);img.addEventListener('error',check);});check();
  }
  async function cardAction(event) {
    const button=event.target.closest('button[data-action]'),article=button?.closest('[data-card-id]');if(!button||!article||button.disabled)return;
    const id=article.dataset.cardId,card=list(state.review?.cards).find(c=>c.id===id);if(!card)return;
    const generation=state.generation,view=state.review,action=button.dataset.action,feedback=article.querySelector('[data-slot="feedback"]');
    const same=()=>generation===state.generation&&state.review?.run_id===view.run_id&&state.review?.fingerprint===view.fingerprint&&article.isConnected;
    const payload={id:view.run_id,card_id:id,fingerprint:view.fingerprint};
    const buttons=[...article.querySelectorAll('button')];const disabled=buttons.map(b=>b.disabled);buttons.forEach(b=>b.disabled=true);state.pendingCards++;actions();feedback.textContent='Working…';
    try {
      if(action==='frames') { const result=await api('/api/violation-frames',payload);if(!same())return;if(result.run_id!==view.run_id||result.card_id!==id||result.fingerprint!==view.fingerprint)throw Error('Frame evidence does not match this review. Reload evidence.');article.querySelector('[data-slot="frames"]').innerHTML=renderFrames(result.frames,state.config?.evidence_origin);feedback.textContent='Evidence frames loaded. Captions show actual sampled times.'; }
      else if(action==='prepare') {
        if(!state.llm.available)throw Error('No LLM explanation available. Deterministic facts remain available.');
        state.preparations.delete(id);article.querySelector('[data-slot="preparation"]').replaceChildren();
        const result=await api('/api/violation-prepare',payload);if(!same())return;
        validatePreparation(result,view,id,state.config?.evidence_origin);
        article.querySelector('[data-slot="preparation"]').innerHTML=renderPreparation(result,state.config?.evidence_origin);watchPreparedImages(article,result);
        feedback.textContent='Evidence prepared locally. Nothing has been sent to ChatGPT. Review the preview, then choose whether to send it.';
      }
      else if(action==='explain-facts'||action==='send-prepared') {
        if(!state.llm.available)throw Error('No LLM explanation available. Deterministic facts remain available.');
        const record=state.preparations.get(id);
        if(action==='send-prepared'&&!record?.ready)throw Error('Load and inspect all prepared image previews before sending. You can choose structured facts only instead.');
        const request=action==='send-prepared'?preparedPayload(record.prepared,view,id):{...payload,consent:true,mode:'facts_only'};
        feedback.textContent=action==='send-prepared'&&record.prepared.mode==='images_and_facts'?'Sending the previewed frames and facts to ChatGPT…':'Sending structured facts only to ChatGPT…';
        const result=await api('/api/violation-explain',request);if(!same())return;if(result.card_id&&result.card_id!==id)throw Error('Explanation belongs to different evidence. Reload this review.');
        if(result.state==='explained'||result.explanation){state.review.explanations=state.review.explanations||{};state.review.explanations[id]=result;}
        article.querySelector('[data-slot="explanation"]').innerHTML=renderExplanation(result,action==='send-prepared'?record.prepared:null);feedback.textContent='Optional explanation response received; the machine finding is unchanged.';
      } else {
        const submittedNote=article.querySelector('[data-slot="note"]').value;
        const result=await api('/api/violation-decision',reviewPayload(view,id,action,submittedNote));if(!same())return;
        if(result.run_id!==view.run_id||result.fingerprint!==view.fingerprint)throw Error('Review changed on the server. Reload evidence before continuing.');
        const updated=list(result.cards).find(c=>c.id===id);if(!updated)throw Error('Saved disposition could not be confirmed. Reload evidence.');
        // Only update the selected card: unsaved notes on other cards remain intact.
        const localCard=list(state.review.cards).find(c=>c.id===id);if(localCard)localCard.review=updated.review;
        const editedWhileSaving=action!=='reset'&&article.querySelector('[data-slot="note"]').value!==submittedNote;
        if(!editedWhileSaving)state.drafts.delete(id);
        updateReviewSummary(shownCardCount);
        article.querySelector('[data-slot="disposition"]').textContent=disposition(updated.review);if(action==='reset')article.querySelector('[data-slot="note"]').value='';feedback.textContent=editedWhileSaving?'Disposition saved with the earlier note. Your newer note is still unsaved; choose Accept or Reject again to save it.':'Human disposition saved. Recorded machine evidence is unchanged.';
      }
    } catch(error){if(same())feedback.textContent=action==='prepare'?'Could not prepare image evidence: '+error.message+' Retry preparation, or explicitly choose structured facts only.':error.message;}
    finally{state.pendingCards--;actions();if(same())buttons.forEach((b,index)=>b.disabled=disabled[index]);}
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
  async function loadLLM(refreshCards=false) {
    $('refresh-chatgpt').disabled=true;
    try{state.llm=await api('/api/violation-llm');}catch{state.llm={available:false,message:'Could not check the ChatGPT connection. Check the local server, then choose Recheck connection. Deterministic facts remain available.'};}
    $('subscription-status').textContent=state.llm.available===true?'Using ChatGPT plan':'No LLM explanation available';$('subscription-message').textContent=state.llm.message||'Deterministic evidence remains available without a ChatGPT connection.';$('connect-chatgpt').hidden=state.llm.available===true;
    $('refresh-chatgpt').disabled=false;
    if(refreshCards&&state.review&&state.pendingCards===0)drawCards();
  }
  async function init() {
    $('clip').addEventListener('change',()=>{$('selected-file').textContent=$('clip').files?.[0]?.name||'No clip selected.';$('rights').checked=false;actions();});
    $('rights').addEventListener('change',actions);$('upload').addEventListener('click',uploadClip);$('cancel-upload').addEventListener('click',()=>state.upload?.abort());$('start').addEventListener('click',startRun);$('cancel-run').addEventListener('click',cancelRun);$('recheck').addEventListener('click',loadConfig);
    $('refresh-run').addEventListener('click',()=>{if(state.run){clearTimeout(state.timer);pollRun(state.run.id,state.generation);}});
    $('refresh-review').addEventListener('click',()=>{if(state.run)loadReview(state.run.id);});$('rule-filter').addEventListener('change',drawCards);$('cards').addEventListener('click',cardAction);$('cards').addEventListener('input',event=>{if(event.target.matches('[data-slot="note"]'))state.drafts.set(event.target.closest('[data-card-id]').dataset.cardId,event.target.value);});$('export-review').addEventListener('click',exportReview);
    $('connect-chatgpt').addEventListener('click',()=>{$('auth-confirmation').hidden=false;$('approve-auth').focus();});$('dismiss-auth').addEventListener('click',()=>{$('auth-confirmation').hidden=true;$('connect-chatgpt').focus();});
    $('refresh-chatgpt').addEventListener('click',()=>loadLLM(true));
    $('approve-auth').addEventListener('click',async()=>{if($('approve-auth').disabled)return;$('approve-auth').disabled=true;$('auth-feedback').textContent='Preparing ChatGPT sign-in…';try{const result=await api('/api/violation-auth/start',{approve:true});const url=authURL(result.url);if(!url)throw Error('The server did not return a verified ChatGPT sign-in address.');root.location.assign(url);}catch(error){$('auth-feedback').textContent='Could not start sign-in: '+error.message;$('approve-auth').disabled=false;}});
    $('reload-library').addEventListener('click',()=>loadLibrary());$('more-runs').addEventListener('click',()=>loadLibrary(true));$('saved-runs').addEventListener('click',event=>{const button=event.target.closest('[data-open-run]');if(button&&!button.disabled)openRun(button.dataset.openRun);});
    await Promise.all([loadConfig(),loadLibrary(),loadLLM()]);
    let id;try{id=root.localStorage.getItem(storageKey);}catch{}if(id)await openRun(id);actions();
    try{const url=new URL(root.location.href),signin=url.searchParams.get('signin');if(signin){url.searchParams.delete('signin');root.history.replaceState(null,'',url.pathname+url.search+url.hash);if(signin==='connected')notice('ChatGPT sign-in returned. '+(state.llm.available?'Your ChatGPT plan is connected. Choose a card to send its structured facts.':'The connection is not available yet. '+(state.llm.message||'Reload to check its status.')));else if(signin==='failed')notice('ChatGPT sign-in did not complete. Deterministic evidence remains available; choose Connect ChatGPT to try again.',true);}}catch{}
  }
  init();
})(typeof window !== 'undefined' ? window : globalThis);
