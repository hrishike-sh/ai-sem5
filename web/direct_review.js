(function(root){
  'use strict';
  const common=typeof module!=='undefined'&&module.exports?require('./violation_lens.js'):root.ViolationLens;
  const {escapeHTML:esc,escapeReviewText:displayText,reviewText,localURL,timestamp}=common;
  const list=value=>Array.isArray(value)?value:[];
  const kindLabels=Object.freeze({
    person_on_two_wheeler:'Rider visible',
    bare_head_of_rider_visible:'Uncovered head visible',
    helmet_on_rider_visible:'Helmet visible on rider',
    crowded_rider_load_visible:'Crowded rider arrangement',
    rider_position_appears_unstable:'Possibly unstable rider position',
    cargo_appears_unsecured:'Possibly unsecured cargo',
    rider_association_unclear:'Unmatched rider',
    head_protection_unclear:'Head protection needs review',
    no_relevant_road_user_visible:'No relevant road user clearly visible',
    helmet_possible:'Possible missing helmet',
    unsafe_rider_load_possible:'Possible unsafe rider arrangement',
    other_visible_concern:'Possible cargo safety concern'
  });
  function sourceKind(name){return /\.(mp4|mov|avi)$/i.test(name||'')?'video':/\.(jpe?g|png)$/i.test(name||'')?'image':null;}
  function selectedTime(value,kind){if(kind==='image')return 0;if(String(value).trim()==='')throw Error('Enter a video timestamp in seconds.');const time=Number(value);if(!Number.isFinite(time)||time<0)throw Error('Use a finite, non-negative video timestamp.');return time;}
  function validateView(view,origin){
    if(!view||typeof view.review_id!=='string'||!view.review_id||typeof view.fingerprint!=='string'||!view.fingerprint)throw Error('The prepared review identity is missing. Prepare the source again.');
    const frames=list(view.frames);
    if(!frames.length||frames.some(f=>!f.id||!f.sha256)||new Set(frames.map(f=>f.id)).size!==frames.length)throw Error('The prepared image identity is incomplete. Prepare the source again.');
    // Historical reviews may retain unavailable assets; they remain readable, but cannot be sent.
    return {...view,original_frame:view.original_frame?{...view.original_frame,url:localURL(view.original_frame.url,origin)}:null,frames:frames.map(f=>({...f,url:localURL(f.url,origin)}))};
  }
  function evidenceLabel(frame,annotations){
    if(frame.role==='context')return 'Context';
    if(frame.role==='subject_crop'){
      const subjects=list(frame.subject_ids).filter(id=>typeof id==='string'&&/^[PV][1-9][0-9]*$/.test(id));
      if(!subjects.length)for(const subject of list(annotations?.subjects))if(/^[PV][1-9][0-9]*$/.test(subject.id)&&list(subject.evidence_ids).includes(frame.id))subjects.push(subject.id);
      return subjects.length===1?subjects[0]+' crop':'Subject crop';
    }
    return frame.id;
  }
  function evidenceReference(id,frames,annotations){const frame=list(frames).find(f=>f.id===id);return frame?`<span title="${esc(id)}">${esc(evidenceLabel(frame,annotations))}</span>`:'';}
  function renderPreview(view){return list(view.frames).map(f=>`<figure>${f.available!==false&&f.url?`<img src="${esc(f.url)}" data-direct-frame="${esc(f.id)}" alt="Exact prepared image: ${esc(evidenceLabel(f,view.annotations))}">`:'<div class="frame-unavailable">'+esc(f.reason||'Prepared image unavailable. Reopen or prepare the source again.')+'</div>'}<figcaption><strong title="${esc(f.id)}">${esc(evidenceLabel(f,view.annotations))}</strong><br>Actual timestamp: ${esc(timestamp(f.timestamp))}<br>${esc(f.width)} x ${esc(f.height)} pixels</figcaption></figure>`).join('');}
  function renderResult(result,frames,annotations){
    if(!result)return '';
    if(Array.isArray(result.subject_results))return renderSubjects(result,frames,annotations);
    const ids=new Set(list(frames).map(f=>f.id));
    const entries=items=>list(items).map(item=>{
      const refs=list(item.evidence_ids);
      if(!refs.length||refs.some(id=>!ids.has(id)))return '<li>Assessment item unavailable: its evidence references do not match this review.</li>';
      const uncertainty=list(item.uncertainty_text).length?item.uncertainty_text:list(item.uncertainties);
      const label=Object.prototype.hasOwnProperty.call(kindLabels,item.kind)?kindLabels[item.kind]:'Visual assessment';
      return `<li><strong>${esc(label)}</strong><p>${displayText(item.text)}</p><small>Evidence: ${refs.map(esc).join(', ')}</small>${uncertainty.length?`<p class="missing">Uncertainty: ${uncertainty.map(displayText).join(' - ')}</p>`:''}</li>`;
    }).join('');
    return `<h3>Optional LLM interpretation</h3><p>Interpretation is separate from the primary local specialist evidence above.</p><p class="hint">Saved model output - LLM</p><p>These are fallible visual interpretations of the selected image. They do not establish legal guilt, speed, or permitted travel direction.</p><h4>Visible observations</h4>${list(result.observations).length?`<ul>${entries(result.observations)}</ul>`:'<p>No structured observations were returned.</p>'}<h4>Suspected issues requiring review</h4>${list(result.suspected_issues).length?`<ul>${entries(result.suspected_issues)}</ul>`:'<p>No suspected issues were returned. This does not establish that the scene is free of violations.</p>'}<h4>Limitations</h4><ul>${list(result.limitations).map(text=>`<li>${displayText(text)}</li>`).join('')}</ul>`;
  }
  function subjectName(subject){return `${subject.kind.charAt(0).toUpperCase()+subject.kind.slice(1)} ${subject.id||'(new)'}`;}
  function renderSubjects(result,frames,annotations){
    const ids=new Set(list(frames).map(f=>f.id));
    const statuses={unclear:'Needs review',no_visible_violation:'No visible violation for this subject',suspected_issue:'Suspected issue - human review required'};
    const entries=items=>list(items).map(item=>{const refs=list(item.evidence_ids);return !refs.length||refs.some(id=>!ids.has(id))?'<li>Assessment item unavailable: evidence does not match this review.</li>':`<li>${displayText(item.text)}<small>Evidence: ${refs.map(id=>evidenceReference(id,frames,annotations)).join(', ')}${list(item.related_subject_ids).length?' - Related subjects: '+item.related_subject_ids.map(esc).join(', '):''}</small>${list(item.uncertainty_text).length?`<p class="missing">${item.uncertainty_text.map(displayText).join(' - ')}</p>`:''}</li>`;}).join('');
    return `<h3>Optional LLM interpretation by subject</h3><p>Interpretation is separate from the primary local specialist evidence above.</p><p class="hint">LLM. Interpretations require human review.</p>${result.subject_results.map(subject=>`<section class="subject-result"><h4>${esc(list(annotations?.subjects).find(s=>s.id===(subject.subject_id||subject.id))?subjectName(list(annotations.subjects).find(s=>s.id===(subject.subject_id||subject.id))):(subject.subject_id||subject.id))}</h4><p><strong>${esc(statuses[subject.status]||'Needs review - status unavailable')}</strong></p><p>Role: ${esc(({rider:'Rider',pedestrian:'Pedestrian',person_unclear:'Person - role needs review',vehicle:'Vehicle'})[subject.role]||'Needs review')}${subject.associated_vehicle_id?` - Associated vehicle: ${esc(subject.associated_vehicle_id)}`:''}</p><h5>Observations</h5>${list(subject.observations).length?`<ul>${entries(subject.observations)}</ul>`:'<p>No observations recorded for this subject.</p>'}<h5>Suspected violations</h5>${list(subject.suspected_issues).length?`<ul>${entries(subject.suspected_issues)}</ul>`:'<p>No suspected violations recorded for this subject. This is not proof of compliance.</p>'}${list(subject.uncertainty_text).length?`<p class="missing">Uncertainty: ${subject.uncertainty_text.map(displayText).join(' - ')}</p>`:''}</section>`).join('')}<h4>Limitations</h4><ul>${list(result.limitations).map(text=>`<li>${displayText(text)}</li>`).join('')}</ul>`;
  }
  function localEvidence(view){return view?.local_evidence||view?.annotations?.local_evidence;}
  function localReady(view){return ['complete','unavailable'].includes(localEvidence(view)?.state);}
  function renderLocalEvidence(view){
    const evidence=localEvidence(view),subjects=list(view?.annotations?.subjects);
    if(!evidence)return '<p>Run helmet check to check helmets on the current numbered subjects locally.</p>';
    const statuses={helmet:'Helmet detected',no_helmet:'No helmet detected',unclear:'Needs review',low_confidence:'Low-confidence detection',not_observed:'Not observed',not_applicable:'Not applicable',unavailable:'Unavailable'};
    const cards=subjects.filter(s=>s.kind==='person').map(subject=>{
      const item=list(evidence.subjects).find(s=>s.id===subject.id),helmet=item?.helmet;
      const detections=list(helmet?.detections).filter(d=>['helmet','no_helmet'].includes(d.label)&&Number.isFinite(d.model_score)&&d.model_score>=0&&d.model_score<=1);
      return `<section class="local-subject"><h4>${esc(subjectName(subject))}</h4><p class="local-status">${esc(helmet?.status==='unclear'&&new Set(detections.map(d=>d.label)).size>1?'Conflicting predictions':statuses[helmet?.status]||'Unavailable')}</p>${detections.length?`<ul>${detections.map(d=>`<li><strong>${d.label==='helmet'?'Helmet':'No helmet'}</strong> <span class="model-score" title="Exact raw model score: ${esc(String(d.model_score))}">${esc(d.model_score.toFixed(3))}</span> <small>raw model score</small></li>`).join('')}</ul>`:'<p>No usable specialist detection for this person. This does not establish whether a helmet is present.</p>'}<p class="hint">${displayText(helmet?.reason||evidence.reason||'Evidence unavailable.')}</p>${list(item?.vehicle_candidates).length?`<p class="hint">Nearby vehicle candidates: ${item.vehicle_candidates.map(v=>esc(v.id)).join(', ')}. Geometry only; rider association unconfirmed.</p>`:''}</section>`;
    }).join('');
    const vehicles=subjects.filter(s=>['motorcycle','bicycle','car','bus','truck'].includes(s.kind));
    const rows=['motorcycle','bicycle','car','bus','truck'].map(kind=>{const group=vehicles.filter(s=>s.kind===kind);return `<tr><th scope="row">${esc(kind)}</th><td>${group.length}</td><td>${group.filter(s=>s.origin==='detector').length}</td><td>${group.filter(s=>s.origin==='manual').length}</td></tr>`;}).join('');
    return `<p class="hint">Helmet check - ${evidence.state==='complete'?'Check complete':'Unavailable'}. ${displayText(evidence.reason||'')}</p><p>Scores are raw model confidence, not accuracy or proof of a violation. Missing detections never mean "no helmet."</p><div class="local-subjects">${cards||'<p>No numbered person boxes in this image.</p>'}</div><h4>Vehicles in current numbered boxes: ${vehicles.length}</h4><p class="hint">Single selected image only, not a unique vehicle count across the clip. Manual and detector boxes may miss or duplicate vehicles.</p><div class="box-table-wrap"><table class="box-table"><thead><tr><th>Class</th><th>Count</th><th>Detector boxes</th><th>Manual boxes</th></tr></thead><tbody>${rows}</tbody></table></div>${view.annotations?.detector?.truncated||evidence.vehicle_counts?.truncated?'<p class="missing">Proposals were truncated to 12 boxes. Counts cover only retained, current boxes.</p>':''}${list(evidence.limitations).length?`<ul class="hint">${evidence.limitations.map(text=>`<li>${esc(text)}</li>`).join('')}</ul>`:''}`;
  }
  function vehicleEvidence(view){return view?.vehicle_type_evidence||view?.annotations?.vehicle_type_evidence;}
  function renderVehicleEvidence(view){
    const evidence=vehicleEvidence(view);if(!evidence)return '<p>Vehicle subtype evidence has not been recorded for this image.</p>';
    if(evidence.state==='unavailable')return '<p><strong>Unavailable</strong></p><p>'+displayText(evidence.reason)+'</p><p>No subtype conclusions or proposal counts are available from this check.</p>';
    const score=d=>Number.isFinite(d.model_score)?`<span class="model-score" title="Exact raw model score: ${esc(String(d.model_score))}">${esc(d.model_score.toFixed(3))}</span>`:'Unavailable';
    const statuses={candidate:'Subtype candidate',low_confidence:'Low-confidence detection',unclear:'Needs review',not_observed:'Not observed',unavailable:'Unavailable'};
    const entries=items=>list(items).map(d=>`<li>${esc(d.label)} ${score(d)} <small>raw model score</small>${d.reason?`<p class="hint">${d.reason==='ambiguous_vehicle_overlap'?'Overlaps multiple numbered vehicle boxes; association unconfirmed.':'No matching numbered vehicle box; association unconfirmed.'}</p>`:''}</li>`).join('');
    const cards=list(evidence.subjects).filter(row=>row.status!=='not_applicable').map(row=>`<section class="local-subject"><h4>${esc(row.id)}</h4><p class="local-status">${esc(statuses[row.status]||'Unavailable')}</p>${list(row.candidates).length?`<ul>${entries(row.candidates)}</ul>`:'<p>No usable subtype candidate. This does not establish that no vehicle is present.</p>'}</section>`).join('');
    return `<p class="hint">Vehicle subtype check - ${displayText(evidence.reason)}</p><p>Subtype labels and ID matches are candidates requiring visual review. Raw scores are not calibrated accuracy.</p><div class="local-subjects">${cards}</div><h4>Unmatched proposals</h4>${list(evidence.unassigned_detections).length?`<ul>${entries(evidence.unassigned_detections)}</ul>`:'<p>No unmatched proposals recorded.</p>'}<h4>Subtype proposal counts: ${esc(evidence.candidate_counts?.total??0)}</h4><p class="hint">Current image detector proposals with score at least 0.5; not verified objects, manual box counts, or unique video traffic. Lower-score candidates above are excluded from this count.</p><ul class="feature-values">${Object.entries(evidence.candidate_counts?.by_type||{}).map(([label,count])=>`<li><span>${esc(label)}</span><strong>${esc(count)}</strong></li>`).join('')}</ul><ul class="hint">${list(evidence.limitations).map(text=>`<li>${esc(text)}</li>`).join('')}</ul>`;
  }
  function canSend(view,loaded,consent,available,busy){return !!view&&localReady(view)&&!view.model_result&&list(view.annotations?.subjects).length>0&&view.source_readiness?.available!==false&&!busy&&available===true&&consent===true&&list(view.frames).length>0&&view.frames.every(f=>f.available!==false&&f.url&&loaded.has(f.id));}
  const helpers={sourceKind,selectedTime,validateView,renderPreview,renderResult,canSend,localEvidence,localReady,renderLocalEvidence,vehicleEvidence,renderVehicleEvidence,renderPlates};
  if(typeof module!=='undefined'&&module.exports)module.exports=helpers;
  if(!root.document)return;
  const $=id=>root.document.getElementById(id),token=root.document.querySelector('meta[name="traffic-token"]')?.content;
  if(!$('direct-panel'))return;
  const state={view:null,media:null,file:null,config:null,available:false,busy:false,upload:null,generation:0,loaded:new Set(),libraryBusy:false,libraryPromise:null,mutationPromise:null,libraryRefreshQueued:false,draft:[],dirty:false,editingResult:false,drag:null,selectedFile:null,autoQueued:false,autoStarted:false,backendUnsupported:false,vehicleBackendUnsupported:false,features:{}};
  const storageKey='violation-lens-direct-review';
  function message(text,error=false){$('direct-message').textContent=text;$('direct-message').classList.toggle('error',error);}
  async function api(path,data){
    const generation=state.generation;
    const request=async()=>{const response=await root.fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Traffic-Token':token},body:JSON.stringify(data)});let value;try{value=await response.json();}catch{throw Error('The server returned an unreadable response. Recheck the connection and retry.');}if(!response.ok)throw Error(value.error||value.message||'The request failed.');return value;};
    if(data===undefined)return request();
    if(state.libraryPromise)await state.libraryPromise;
    if(state.mutationPromise)try{await state.mutationPromise;}catch{}
    if(generation!==state.generation)throw Error('The source changed before this request started.');
    const pending=request();state.mutationPromise=pending;
    try{return await pending;}finally{if(state.mutationPromise===pending)state.mutationPromise=null;}
  }
  const featureNames={localization:'Subject localization',helmet:'Helmet evidence',vehicles:'Vehicle class counts',types:'Vehicle subtype evidence',plates:'Number plate reading'};
  function feature(key,status,detail){state.features[key]={status,detail};renderFeatures();}
  function renderFeatures(){
    for(const key of Object.keys(featureNames)){
      const value=state.features[key]||{status:'waiting',detail:'Waiting for an image.'};
      $('direct-feature-'+key).dataset.status=value.status;
      $('direct-feature-'+key+'-status').textContent=({waiting:'Waiting',running:'Processing locally',complete:'Complete',error:'Failed',unavailable:'Unavailable',unsupported:'Not applicable',stale:'Needs saved corrections'})[value.status]||value.status;
      $('direct-feature-'+key+'-detail').textContent=reviewText(value.detail);
    }
    $('direct-report-summary').innerHTML=Object.entries(featureNames).map(([key,name])=>{const value=state.features[key]||{status:'waiting',detail:'Waiting for an image.'};return `<li><strong>${esc(name)}</strong>: ${esc(value.status)}. ${displayText(value.detail)}</li>`;}).join('');
    $('direct-report').hidden=!state.view;
    if(root.ViolationEvidence)$('direct-evidence-table').innerHTML=(state.dirty?'<p class="notice">Unsaved box edits. The table below is the last saved evidence; save corrections to rebuild it.</p>':'')+root.ViolationEvidence.renderEvidenceTable(state.view);
    const rows=list(state.view?.annotations?.subjects);$('direct-feature-localization-output').innerHTML=rows.length?'<p class="hint">Current IDs: '+rows.map(row=>esc(row.id)).join(', ')+'</p>':'';
    $('direct-feature-vehicles-output').innerHTML=state.view?.annotations?'<ul class="feature-values">'+['motorcycle','bicycle','car','bus','truck'].map(kind=>'<li><span>'+esc(kind)+'</span><strong>'+rows.filter(row=>row.kind===kind).length+'</strong></li>').join('')+'</ul>':'';
  }
  function featuresFromView(view){
    const subjects=list(view.annotations?.subjects),evidence=localEvidence(view),types=vehicleEvidence(view),people=subjects.filter(s=>s.kind==='person').length,vehicles=subjects.length-people;
    state.features=view.annotations?{
      localization:{status:view.annotations.detector?.state==='unavailable'?'unavailable':'complete',detail:(view.annotations.detector?.state==='unavailable'?(view.annotations.detector.reason||'Subject detector unavailable. Add boxes manually.')+' ':'')+`${subjects.length} current numbered boxes (${people} people). Review IDs and correct missed or incorrect boxes.`},
      vehicles:{status:'complete',detail:`${vehicles} vehicle boxes in this selected image. ${subjects.filter(s=>s.kind!=='person'&&s.origin==='manual').length} manual; ${subjects.filter(s=>s.kind!=='person'&&s.origin==='detector').length} detector. Not a clip traffic count.${view.annotations.detector?.truncated?' Detector proposals were truncated.':''}`},
      helmet:{status:evidence?(evidence.state==='unavailable'?'unavailable':people?'complete':'unsupported'):'waiting',detail:evidence?(evidence.reason||'Saved local evidence restored.'):'Waiting for the local helmet specialist.'}
    }:{localization:{status:'waiting',detail:'Image prepared. Waiting for subject localization.'},helmet:{status:'waiting',detail:'Needs numbered person boxes.'},vehicles:{status:'waiting',detail:'Needs numbered vehicle boxes.'}};
    state.features.types={status:types?(types.state==='complete'?'complete':'unavailable'):'waiting',detail:types?(types.reason||'Saved subtype evidence restored.'):'Waiting for the local vehicle subtype specialist.'};
    const plates=view.plate_evidence;state.features.plates={status:plates?(plates.state==='complete'?'complete':'unavailable'):'waiting',detail:plates?(plates.reason||'Saved plate evidence restored. Unreadable text is not inferred.'):'Waiting for local plate detection and OCR.'};
    const report=view.feature_report;if(report?.features){const mapping={localization:'localization',helmet:'helmet',vehicles:'vehicle_counts',types:'vehicle_types',plates:'plates'};for(const [key,source] of Object.entries(mapping)){const saved=report.features[source];if(saved&&['unavailable','pending','needs_input','not_applicable','failed'].includes(saved.state))state.features[key]={status:({pending:'waiting',needs_input:'waiting',not_applicable:'unsupported',failed:'error'})[saved.state]||saved.state,detail:saved.summary||list(saved.errors).join(' ')};}}
    renderFeatures();
  }
  function startQueuedImage(){if(state.autoQueued&&!state.autoStarted&&!state.busy&&!state.libraryBusy&&state.config){state.autoStarted=true;state.autoQueued=false;return prepare();}}
  function selectSource(file){
    if(file&&file===state.selectedFile)return;
    const previousUpload=state.upload;state.generation++;state.upload=null;state.busy=false;previousUpload?.abort();
    state.selectedFile=file||null;state.media=null;state.file=null;state.backendUnsupported=false;state.vehicleBackendUnsupported=false;state.autoStarted=false;clearView();
    const kind=sourceKind(file?.name);state.autoQueued=kind==='image';state.features={};renderFeatures();
    $('direct-upload-progress').hidden=true;$('direct-filename').textContent=file?.name||'No source selected.';$('direct-time-wrap').hidden=kind!=='video';$('direct-time').value='0';
    message(kind==='image'?'Starting all local checks for this image.':kind==='video'?'Choose a timestamp, then prepare this video frame.':'Choose a JPG, PNG, MP4, MOV or AVI file.',!kind);actions();return startQueuedImage();
  }
  function actions(){
    const file=state.selectedFile||$('direct-file').files?.[0],kind=sourceKind(file?.name),view=state.view;
    $('direct-prepare').disabled=state.busy||!state.config||!kind;
    $('direct-prepare').textContent=state.media&&file===state.file?'Prepare selected frame':'Upload and preview';
    for(const id of ['direct-time','direct-note'])$(id).disabled=state.busy;
    $('direct-cancel').hidden=!state.upload;$('direct-library-refresh').disabled=state.busy||state.libraryBusy;
    $('direct-plate-run').disabled=state.busy||state.dirty||!view?.annotations||view.source_readiness?.available===false;
    for(const input of $('direct-evidence-table').querySelectorAll('input,button'))input.disabled=state.busy||state.dirty||view?.source_readiness?.available===false;
    $('direct-local-run').disabled=state.busy||state.dirty||state.backendUnsupported||!view?.annotations||view.source_readiness?.available===false;
    $('direct-local-run').textContent=view?.model_result?'Run helmet check (clears LLM review)':'Run helmet check';
    $('direct-vehicle-run').disabled=state.busy||state.dirty||state.vehicleBackendUnsupported||!view?.annotations||view.source_readiness?.available===false;
    $('direct-vehicle-run').textContent=view?.model_result?'Run subtype check (clears LLM review)':'Run subtype check';
    $('direct-vehicle-rerun-notice').hidden=!view?.model_result;
    $('direct-local-rerun-notice').hidden=!view?.model_result;
    $('direct-send').disabled=state.dirty||!canSend(view,state.loaded,true,state.available,state.busy);
    const editingDisabled=state.busy||!view||(!!view.model_result&&!state.editingResult)||view.source_readiness?.available===false;
    $('direct-annotate').disabled=editingDisabled||!!view?.annotations;
    $('direct-save-boxes').disabled=editingDisabled||!state.dirty;
    $('direct-add-box').disabled=editingDisabled||state.draft.length>=12;
    $('direct-new-kind').disabled=editingDisabled;
    for(const input of $('direct-boxes').querySelectorAll('input,select,button'))input.disabled=editingDisabled;
    $('direct-send').hidden=!!view?.model_result;
    for(const action of ['accept','reject','reset'])$('direct-'+action).disabled=state.busy||state.dirty||!view?.model_result;
    $('direct-export').hidden=!view;$('direct-export').disabled=state.busy||state.dirty||!view;
    $('direct-edit-boxes').hidden=!view?.model_result||state.editingResult;
    $('direct-edit-boxes').disabled=state.busy||view?.source_readiness?.available===false;
    $('direct-cancel-box-edit').hidden=!state.editingResult;
    $('direct-cancel-box-edit').disabled=state.busy;
    for(const button of $('direct-library').querySelectorAll('[data-direct-open]'))button.disabled=state.busy;
  }
  function clearView(){$('direct-drop-preview').hidden=true;$('direct-drop-preview').src='';state.view=null;state.drag=null;state.dirty=false;state.editingResult=false;state.loaded.clear();for(const id of ['direct-evidence','direct-result','direct-decision','direct-local','direct-report','direct-vehicle-evidence','direct-plate-evidence'])$(id).hidden=true;}
  function remember(id){try{root.localStorage.setItem(storageKey,id);}catch{message('Review saved on the server. This browser cannot remember it; reopen it from saved image reviews.');}}
  const subjectKinds=['person','motorcycle','bicycle','car','bus','truck'];
  function drawEditor(redrawTable=true){
    if(redrawTable)$('direct-boxes').innerHTML=state.draft.map((s,i)=>`<tr><th scope="row">${esc(subjectName(s))}</th><td><select aria-label="Subject ${i+1} type" data-box-index="${i}" data-box-field="kind">${subjectKinds.map(kind=>`<option value="${kind}"${s.kind===kind?' selected':''}>${esc(kind)}</option>`).join('')}</select></td>${s.bbox.map((n,j)=>`<td><input type="number" min="0" step="1" value="${esc(n)}" aria-label="Subject ${i+1} ${['left','top','right','bottom'][j]}" data-box-index="${i}" data-box-field="${j}"></td>`).join('')}<td><button type="button" class="secondary" data-box-delete="${i}">Delete</button></td></tr>`).join('');
    const original=state.view?.original_frame||state.view?.frames[0];
    const width=state.view?.annotations?.image_width||original?.width,height=state.view?.annotations?.image_height||original?.height;
    $('direct-box-overlay').innerHTML=width&&height?state.draft.map(s=>`<span class="subject-box" style="left:${100*s.bbox[0]/width}%;top:${100*s.bbox[1]/height}%;width:${100*(s.bbox[2]-s.bbox[0])/width}%;height:${100*(s.bbox[3]-s.bbox[1])/height}%"><b>${esc(s.id||'New')}</b></span>`).join(''):'';
  }
  function edited(redrawTable=true){state.dirty=true;for(const key of Object.keys(featureNames))feature(key,'stale','Boxes changed. Save corrections to rebuild the combined report.');$('direct-local-output').innerHTML='<p>Boxes changed. Save corrections to rerun the local checks.</p>';$('direct-result').hidden=true;$('direct-vehicle-output').innerHTML='<p>Boxes changed. Save corrections to rerun subtype checks.</p>';drawEditor(redrawTable);$('direct-preview-status').textContent='Boxes changed. Save corrections to rebuild the exact context and crop previews before sending.';actions();}
  async function annotate(manual){
    const view=state.view;if(!view||state.busy||(view.model_result&&!(manual&&state.editingResult))||(!manual&&view.annotations))return;
    const generation=state.generation;state.busy=true;feature('localization','running',manual?'Saving corrected subject boxes.':'Detecting and numbering subjects in the selected image.');feature('helmet','waiting','Waiting for numbered person boxes.');feature('vehicles','waiting','Waiting for numbered vehicle boxes.');actions();message(manual?'Saving boxes and preparing exact crops locally.':'Detecting and numbering subjects locally. All checks stay on this computer.');
    try{const data={review_id:view.review_id,fingerprint:view.fingerprint};if(manual)data.subjects=state.draft.map(s=>({...s,bbox:[...s.bbox]}));const result=await api('/api/direct-annotate',data);if(generation!==state.generation||state.view!==view)return;if(result.review_id!==view.review_id)throw Error('The annotation response belongs to another review. Reopen this review.');draw(result);await runLocalModels(true);if(generation===state.generation)await runVehicleModels(true);if(generation===state.generation)await runPlateModels(true);state.libraryRefreshQueued=true;}
    catch(error){if(generation===state.generation&&state.view===view){feature('localization','error',error.message+' Add or correct boxes manually and save them.');feature('helmet','unavailable','No current numbered person boxes; save manual corrections to continue.');feature('vehicles','unavailable','No current numbered vehicle boxes; save manual corrections to continue.');$('direct-local').hidden=false;$('direct-local-output').innerHTML='<p>Subject localization failed. No helmet or vehicle conclusions are available. Add boxes manually to continue. This request error is not saved in the exported evidence.</p>';message(error.message+' You can add or correct boxes manually and save them.',true);}}
    finally{if(generation===state.generation){state.busy=false;actions();if(state.libraryRefreshQueued)await loadLibrary();}}
  }
  async function runLocalModels(fromAnnotation=false){
    const view=state.view;if(!view||(!fromAnnotation&&(state.busy||state.backendUnsupported))||state.dirty||!view.annotations)return;
    const generation=state.generation;state.busy=true;actions();message('Checking helmets locally. All checks stay on this computer.');feature('helmet','running','Checking the same selected image with the local specialist.');
    $('direct-local').hidden=false;$('direct-local-output').innerHTML='<p role="status">Checking helmets locally...</p>';
    try{
      const result=await api('/api/direct-local-models',{review_id:view.review_id,fingerprint:view.fingerprint});
      if(generation!==state.generation||state.view!==view)return;
      if(result.review_id!==view.review_id)throw Error('The local evidence belongs to another review. Reopen this review.');
      if(!localReady(result))throw Error('The local check did not return a completed or unavailable result.');
      draw(result);message(localEvidence(result).state==='complete'?'Local checks saved. Review the scores and exact previews. LLM explanation is optional.':'Local specialist unavailable. Review the reason below; retry locally or request an optional LLM interpretation with consent.');state.libraryRefreshQueued=true;
    }catch(error){if(generation===state.generation&&state.view===view){const unsupported=error.message==='Unknown operation.';state.backendUnsupported=unsupported;const detail=unsupported?'The running server needs the updated backend loaded. Retrying cannot fix this.':error.message+' Retry with Run helmet check.';feature('helmet',unsupported?'unavailable':'error',detail);$('direct-local-output').innerHTML='<p>'+esc(detail)+'</p><p>Subject localization and current numbered vehicle counts remain available above. No new helmet evidence was saved. This request error is shown here only; the export contains the saved evidence and its last recorded status.</p>';message(detail,true);}}
    finally{if(generation===state.generation&&!fromAnnotation){state.busy=false;actions();if(state.libraryRefreshQueued)await loadLibrary();}}
  }
  async function runVehicleModels(fromAnnotation=false){
    const view=state.view;if(!view||(!fromAnnotation&&(state.busy||state.vehicleBackendUnsupported))||state.dirty||!view.annotations)return;
    const generation=state.generation,priorHelmet=state.features.helmet;state.busy=true;feature('types','running','Checking vehicle subtype proposals on the same selected image.');actions();message('Checking vehicle subtypes locally. All checks stay on this computer.');
    $('direct-vehicle-evidence').hidden=false;$('direct-vehicle-output').innerHTML='<p>Checking vehicle subtype proposals locally...</p>';
    try{
      const result=await api('/api/direct-vehicle-types',{review_id:view.review_id,fingerprint:view.fingerprint});
      if(generation!==state.generation||state.view!==view)return;
      if(result.review_id!==view.review_id)throw Error('The subtype evidence belongs to another review. Reopen this review.');
      if(!['complete','unavailable'].includes(vehicleEvidence(result)?.state))throw Error('The subtype check did not return a completed or unavailable result.');
      draw(result);if(priorHelmet&&['error','unavailable'].includes(priorHelmet.status)){state.features.helmet=priorHelmet;renderFeatures();}
      message('Local report updated. Review each feature result, including unavailable checks. LLM explanation is optional.');state.libraryRefreshQueued=true;
    }catch(error){if(generation===state.generation&&state.view===view){const unsupported=error.message==='Unknown operation.';state.vehicleBackendUnsupported=unsupported;const detail=unsupported?'The running server needs the updated backend loaded. Retrying cannot fix this.':error.message+' Retry with Run subtype check.';feature('types',unsupported?'unavailable':'error',detail);$('direct-vehicle-output').innerHTML='<p>'+esc(detail)+'</p><p>Other completed features remain available. This request error is shown here only; the export contains saved evidence.</p>';message(detail,true);}}
    finally{if(generation===state.generation&&!fromAnnotation){state.busy=false;actions();if(state.libraryRefreshQueued)await loadLibrary();}}
  }
  async function runPlateModels(fromAnnotation=false){
    const view=state.view;if(!view||(!fromAnnotation&&state.busy)||state.dirty||!view.annotations)return;
    const generation=state.generation,previousFeatures={...state.features};state.busy=true;feature('plates','running','Finding plate candidates and reading visible text locally.');actions();
    try{const result=await api('/api/direct-plates',{review_id:view.review_id,fingerprint:view.fingerprint});if(generation!==state.generation||state.view!==view)return;if(result.review_id!==view.review_id||!['complete','unavailable'].includes(result.plate_evidence?.state))throw Error('The plate check did not return a completed or unavailable result.');draw(result);for(const key of ['helmet','types'])if(previousFeatures[key]?.status==='error')state.features[key]=previousFeatures[key];renderFeatures();state.libraryRefreshQueued=true;message('Local checks saved. Inspect the image and evidence table. LLM interpretation is optional.');}
    catch(error){if(generation===state.generation&&state.view===view){feature('plates','error',error.message+' Retry using Run plate check.');message(error.message,true);}}
    finally{if(generation===state.generation&&!fromAnnotation){state.busy=false;actions();if(state.libraryRefreshQueued)await loadLibrary();}}
  }
  async function correctPlate(event){
    const button=event.target.closest('[data-plate-save]'),view=state.view;if(!button||!button.dataset.plateSave||!view||state.busy||state.dirty||view.source_readiness?.available===false)return;
    const input=Array.from($('direct-evidence-table').querySelectorAll('input[data-plate-vehicle]')).find(i=>i.dataset.plateVehicle===button.dataset.plateSave&&i.dataset.plateId===button.dataset.plateId);if(!input)return;
    const generation=state.generation;state.busy=true;actions();
    try{const result=await api('/api/direct-plate-correction',{review_id:view.review_id,fingerprint:view.fingerprint,vehicle_id:button.dataset.plateSave,plate_id:button.dataset.plateId,text:input.value});if(generation!==state.generation||state.view!==view)return;if(result.review_id!==view.review_id)throw Error('The plate correction belongs to another review.');draw(result);message('Plate correction saved separately from original OCR. Prior LLM assessment and review decision were cleared.');state.libraryRefreshQueued=true;}
    catch(error){if(generation===state.generation)message(error.message,true);}
    finally{if(generation===state.generation){state.busy=false;actions();if(state.libraryRefreshQueued)await loadLibrary();}}
  }
  function renderPlates(view){
    const evidence=view.plate_evidence;if(!evidence)return '<p>Plate detection and OCR have not run on this image.</p>';
    const entries=list(evidence.subjects).filter(s=>s.status!=='not_applicable').map(s=>`<section class="local-subject"><h4>${esc(s.id)}</h4><p>${displayText(s.status==='low_confidence'?'Low-confidence detection':s.status.replace(/_/g,' '))}</p>${list(s.candidates).map(c=>`<p><strong>${esc(c.text||'Unreadable')}</strong><br>Detector score: ${esc(c.detector_score??'unavailable')} · OCR score: ${esc(c.ocr_score??'unavailable')}<br><small>Plate box in source pixels: ${list(c.box).map(esc).join(', ')}</small></p>`).join('')||'<p>No readable plate candidate.</p>'}</section>`).join('');
    const unassigned=list(evidence.unassigned_detections),reasonLabels={ambiguous_vehicle_overlap:'Ambiguous vehicle overlap',no_numbered_vehicle_match:'No numbered vehicle match'};
    const unmatched=unassigned.length?`<h4>Unassigned plate proposals</h4><p class="hint">These candidates are not linked to any vehicle. OCR text is unverified and may be incomplete or wrong.</p><div class="local-subjects">${unassigned.map(c=>`<section class="local-subject"><h4>${esc(c.plate_id||'Unassigned plate')}</h4><p><strong>Unassigned — ${esc(reasonLabels[c.reason]||'Association needs review')}</strong></p><p>OCR candidate: ${esc(c.text||'Unreadable')}<br>Detector score: ${esc(c.detector_score??'unavailable')} · OCR score: ${esc(c.ocr_score??'unavailable')}</p><p class="hint">Plate box in source pixels: ${list(c.box).map(esc).join(', ')}</p></section>`).join('')}</div>`:'<p class="hint">No unassigned plate proposals recorded.</p>';
    const width=evidence.source?.width,height=evidence.source?.height,source=view.original_frame||view.frames[0],candidates=list(evidence.subjects).flatMap(s=>list(s.candidates).map(c=>({...c,subject:s.id}))).concat(unassigned.map(c=>({...c,subject:'Unassigned'})));
    const overlay=width>0&&height>0&&source?.url&&candidates.length?`<svg class="plate-map" viewBox="0 0 ${esc(width)} ${esc(height)}" role="img" aria-label="Local plate candidate boxes on the source image"><image href="${esc(source.url)}" width="${esc(width)}" height="${esc(height)}"/>${candidates.filter(c=>list(c.box).length===4&&c.box.every(Number.isFinite)).map(c=>`<g><rect x="${c.box[0]}" y="${c.box[1]}" width="${c.box[2]-c.box[0]}" height="${c.box[3]-c.box[1]}" fill="none" stroke="#ffcf4a" stroke-width="${Math.max(2,width/500)}"/><text x="${c.box[0]}" y="${Math.max(16,c.box[1]-6)}" fill="#ffcf4a" stroke="#172e33" stroke-width=".6" font-size="${Math.max(16,width/65)}">${esc(c.subject)} plate</text></g>`).join('')}</svg>`:'';
    return `<p>${displayText(evidence.reason||'Local plate check complete.')}</p>${overlay}<div class="local-subjects">${entries}</div>${unmatched}<p class="hint">Scores are raw model output, not calibrated accuracy. Corrections in the table retain the original OCR.</p>`;
  }
  function draw(view){
    state.view=validateView(view,state.config?.evidence_origin);const current=state.view,generation=state.generation;featuresFromView(current);
    state.loaded.clear();$('direct-evidence').hidden=false;
    state.dirty=false;state.editingResult=false;state.draft=list(current.annotations?.subjects).map(s=>({id:s.id,kind:s.kind,bbox:[...s.bbox],origin:s.origin}));
    $('direct-annotation-editor').hidden=!!current.model_result;
    const original=current.original_frame||current.frames[0],context=current.frames.find(f=>f.role==='context')||original;$('direct-drop-preview').hidden=!context?.url||context.available===false;$('direct-drop-preview').src=context?.available!==false?context?.url||'':'';$('direct-editor-image').src=original?.available!==false?original?.url||'':'';
    $('direct-annotation-notice').textContent=(current.annotations?.notice||'Detect subjects locally or draw missing boxes. A person box does not establish a rider role.')+(current.annotations?.detector?.truncated?' The detector returned more than 12 proposals; only its 12 highest-confidence proposals were retained. Review the image for missing subjects. At most 12 boxes can be submitted: delete or replace boxes to include missed subjects.':'');
    drawEditor();
    $('direct-plate-evidence').hidden=!current.annotations;$('direct-plate-output').innerHTML=renderPlates(current);
    $('direct-local').hidden=!current.annotations;$('direct-local-output').innerHTML=renderLocalEvidence(current);
    $('direct-vehicle-evidence').hidden=!current.annotations;$('direct-vehicle-output').innerHTML=renderVehicleEvidence(current);
    $('direct-source').textContent=`Source: ${current.source?.name||'Saved source'} - Image review: ${current.review_id}`;
    if(!state.selectedFile)$('direct-filename').textContent='Saved review source: '+(current.source?.name||'Saved source')+'. Choose a file to start a new review.';
    $('direct-source-warning').hidden=current.source_readiness?.available!==false;
    $('direct-source-warning').textContent=current.source_readiness?.available===false?'Historical assessment and your decision remain available. Current original frames cannot be verified. '+(current.source_readiness.reason||'Source unavailable.') : '';
    $('direct-provenance').hidden=!current.synthetic;
    $('direct-provenance').textContent=current.synthetic?'Synthetic source. This image review does not demonstrate real-world detection accuracy.':'';
    $('direct-preview').innerHTML=renderPreview(current);
    $('direct-preview-status').textContent='Loading the exact prepared image. Inspect it before choosing Send to LLM.';
    const images=Array.from($('direct-preview').querySelectorAll('img[data-direct-frame]'));
    for(const image of images){
      const ready=()=>{if(generation!==state.generation||state.view!==current)return;if(image.naturalWidth>0)state.loaded.add(image.dataset.directFrame);else state.loaded.delete(image.dataset.directFrame);$('direct-preview-status').textContent=current.frames.every(f=>state.loaded.has(f.id))?'All prepared images loaded. These are the exact images selected for sending.':'A prepared image could not load. Prepare it again before sending.';actions();};
      image.addEventListener('load',ready);image.addEventListener('error',()=>{if(state.view!==current)return;state.loaded.delete(image.dataset.directFrame);$('direct-preview-status').textContent='A prepared image could not load. Prepare it again before sending.';actions();});if(image.complete)ready();
    }
    if(!images.length)$('direct-preview-status').textContent='The prepared image is unavailable. Saved assessments can still be reviewed.';
    $('direct-result').hidden=!current.model_result;$('direct-result').innerHTML=renderResult(current.model_result,current.frames,current.annotations);
    $('direct-decision').hidden=!current.model_result;
    $('direct-disposition').textContent=current.review?.decision==='accept'?'You accepted this assessment.':current.review?.decision==='reject'?'You rejected this assessment.':'No human disposition saved.';
    $('direct-note').value=current.review?.note||'';remember(current.review_id);actions();
  }
  async function connection(){
    const results=await Promise.allSettled([api('/api/config'),api('/api/violation-llm')]);
    state.config=results[0].status==='fulfilled'?results[0].value:null;
    const llm=results[1].status==='fulfilled'?results[1].value:null;state.available=llm?.available===true;
    $('direct-account').textContent=state.available?'LLM connection ready. Review the images and OpenAI sharing notice before sending.':'LLM image review needs your ChatGPT connection. Use Connect ChatGPT above, then Recheck connection. '+(llm?.message||'');
    if(!state.config)message('Could not connect to the local server. Check that it is running, then choose Recheck connection.',true);actions();startQueuedImage();
  }
  function upload(file){return new Promise((resolve,reject)=>{
    const xhr=new root.XMLHttpRequest(),generation=state.generation;state.upload=xhr;$('direct-upload-progress').hidden=false;$('direct-upload-progress').value=0;actions();
    xhr.open('POST','/api/upload?name='+encodeURIComponent(file.name));xhr.setRequestHeader('X-Traffic-Token',token);xhr.setRequestHeader('Content-Type','application/octet-stream');
    xhr.upload.onprogress=event=>{if(generation===state.generation&&event.lengthComputable){$('direct-upload-progress').value=Math.round(event.loaded/event.total*100);message('Uploading source: '+$('direct-upload-progress').value+'%.');}};
    xhr.onload=()=>{let body;try{body=JSON.parse(xhr.responseText);}catch{return reject(Error('The upload response could not be read. Retry upload.'));}if(xhr.status<200||xhr.status>=300)return reject(Error(body.error||'Upload failed. Retry upload.'));resolve(body);};
    xhr.onerror=()=>reject(Error('Upload failed. Check the local server and retry.'));xhr.onabort=()=>reject(Error('Upload cancelled. You can choose another source or retry.'));xhr.send(file);
  });}
  async function prepare(){
    if($('direct-prepare').disabled||state.busy)return;
    const file=state.selectedFile||$('direct-file').files?.[0];let time;try{time=selectedTime($('direct-time').value,sourceKind(file?.name));}catch(error){message(error.message,true);return;}
    const generation=++state.generation;state.busy=true;feature('localization','running','Uploading and preparing the shared source image.');actions();message('Preparing your local preview. All checks stay on this computer.');
    try{
      if(state.libraryPromise)await state.libraryPromise;
      if(state.mutationPromise){feature('localization','waiting','Waiting for the previous local check to finish before uploading this replacement.');try{await state.mutationPromise;}catch{}}
      if(generation!==state.generation)return;
      if(!state.media||state.file!==file){const media=await upload(file);if(generation!==state.generation)return;if(!media.id||!media.source_revision)throw Error('Upload identity is incomplete. Retry upload.');state.media=media;state.file=file;}
      state.upload=null;$('direct-upload-progress').hidden=true;actions();
      const view=await api('/api/direct-prepare',{media:state.media.id,source_revision:state.media.source_revision,timestamp:time});if(generation!==state.generation)return;draw(view);message('Preview prepared locally. Starting checks for the same selected image.');state.libraryRefreshQueued=true;state.busy=false;actions();await annotate(false);
    }catch(error){if(generation===state.generation){feature('localization','error',error.message);feature('helmet','unavailable','Source preparation failed.');feature('vehicles','unavailable','Source preparation failed.');message(error.message,true);}}
    finally{if(generation===state.generation){state.upload=null;state.busy=false;$('direct-upload-progress').hidden=true;actions();if(state.libraryRefreshQueued)await loadLibrary();}}
  }
  async function open(id){if(state.busy)return;const generation=++state.generation;state.busy=true;clearView();actions();message('Opening saved image review. No new model request is made.');try{const view=await api('/api/direct-review?id='+encodeURIComponent(id));if(generation!==state.generation)return;if(view.review_id!==id)throw Error('The returned review does not match the selected review.');draw(view);message(view.model_result?'Saved LLM assessment restored. No new model request was made.':'Prepared review restored. Inspect its image before choosing to send.');}catch(error){if(generation===state.generation)message(error.message,true);}finally{if(generation===state.generation){state.busy=false;actions();if(state.libraryRefreshQueued)await loadLibrary();}}}
  async function mutate(action){
    const view=state.view;if(!view||state.busy)return;
    if(action==='analyze'&&(state.dirty||!canSend(view,state.loaded,true,state.available,false)))return;
    if(action==='export'&&state.dirty)return;
    if(!['analyze','export'].includes(action)&&(!view.model_result||state.dirty))return;
    const generation=state.generation;state.busy=true;actions();message(action==='analyze'?'LLM is reviewing the selected image. Keep this page open while the request completes.':'Saving your local review.');
    try{
      const data={review_id:view.review_id,fingerprint:view.fingerprint};
      if(action==='analyze')data.consent=true;else if(action!=='export'){data.decision=action;data.note=$('direct-note').value;}
      const result=await api('/api/direct-'+(action==='analyze'?'analyze':action==='export'?'export':'decision'),data);
      if(generation!==state.generation||state.view!==view)return;
      if(action==='export'){const url=localURL(result.url,state.config?.evidence_origin);if(!url)throw Error('The export download address is invalid.');const link=root.document.createElement('a');link.href=url;link.download=result.filename||'llm-image-review.json';root.document.body.append(link);link.click();link.remove();message('Review JSON exported with local evidence, any LLM interpretation, and separate human disposition.');}
      else{if(result.review_id!==view.review_id||result.fingerprint!==view.fingerprint)throw Error('The returned review identity changed. Reopen the saved review before continuing.');draw(result);message(action==='analyze'?'LLM assessment saved. Read the uncertainty and add your review decision.':'Your disposition was saved separately from the model assessment.');loadLibrary();}
    }catch(error){if(generation===state.generation)message(error.message+(action==='analyze'?' No alternate model or paid API fallback will be used. Reopen the saved review before retrying if the result is uncertain.':''),true);}
    finally{if(generation===state.generation){state.busy=false;actions();if(state.libraryRefreshQueued)await loadLibrary();}}
  }
  function loadLibrary(){
    if(state.busy){state.libraryRefreshQueued=true;return state.libraryPromise||Promise.resolve();}
    if(state.libraryPromise)return state.libraryPromise;
    state.libraryBusy=true;state.libraryRefreshQueued=false;$('direct-library-refresh').disabled=true;
    const pending=(async()=>{try{const response=await api('/api/direct-library');const reviews=Array.isArray(response)?response:list(response.reviews);$('direct-library').innerHTML=reviews.map(item=>`<div class="saved-run"><div><p>${esc(item.source?.name||'Saved image review')}</p><small>${esc(item.review_id)}${item.synthetic?' - Synthetic':''}</small></div><button type="button" class="secondary" data-direct-open="${esc(item.review_id)}">Open image review</button></div>`).join('');$('direct-library-message').textContent=reviews.length?'Reopening shows saved output without a new model request.':'No saved image reviews yet.';}catch(error){$('direct-library-message').textContent='Could not load image reviews: '+error.message;}finally{state.libraryBusy=false;actions();}})();
    state.libraryPromise=pending;
    return pending.finally(()=>{if(state.libraryPromise===pending)state.libraryPromise=null;startQueuedImage();});
  }
  $('direct-file').addEventListener('change',()=>selectSource($('direct-file').files?.[0]));
  for(const name of ['dragenter','dragover'])$('direct-drop-zone').addEventListener(name,event=>{event.preventDefault();$('direct-drop-zone').classList.toggle('drag-active',true);});
  $('direct-drop-zone').addEventListener('dragleave',()=> $('direct-drop-zone').classList.toggle('drag-active',false));
  $('direct-drop-zone').addEventListener('drop',event=>{event.preventDefault();$('direct-drop-zone').classList.toggle('drag-active',false);const files=Array.from(event.dataTransfer?.files||[]);if(files.length!==1){message('Drop one image or video at a time.',true);return;}return selectSource(files[0]);});
  $('direct-time').addEventListener('input',()=>{if(state.busy)return;state.generation++;clearView();message('Timestamp changed. Prepare the selected frame before sending.');actions();});

  $('direct-prepare').addEventListener('click',prepare);$('direct-cancel').addEventListener('click',()=>state.upload?.abort());$('direct-send').addEventListener('click',()=>mutate('analyze'));
  $('direct-edit-boxes').addEventListener('click',()=>{if(state.busy||!state.view?.model_result||state.view.source_readiness?.available===false)return;state.editingResult=true;$('direct-annotation-editor').hidden=false;$('direct-annotation-editor').open=true;message('Editing boxes. The current assessment remains saved until you save corrections. Saving changes clears that assessment and your review decision.');actions();});
  $('direct-cancel-box-edit').addEventListener('click',()=>{if(state.busy||!state.editingResult)return;draw(state.view);message('Box edits discarded. The saved assessment and decision are unchanged.');});
  $('direct-vehicle-run').addEventListener('click',()=>runVehicleModels());
  $('direct-plate-run').addEventListener('click',()=>runPlateModels());
  $('direct-evidence-table').addEventListener('click',correctPlate);
  $('direct-local-run').addEventListener('click',()=>runLocalModels());
  $('direct-annotate').addEventListener('click',()=>annotate(false));$('direct-save-boxes').addEventListener('click',()=>annotate(true));
  $('direct-add-box').addEventListener('click',()=>{if(state.busy||!state.view||(state.view.model_result&&!state.editingResult)||state.draft.length>=12)return;const f=state.view.original_frame||state.view.frames[0];state.draft.push({kind:$('direct-new-kind').value||'person',bbox:[0,0,Math.max(1,Math.round(f.width/4)),Math.max(1,Math.round(f.height/4))],origin:'manual'});edited();});
  function updateBox(event,redrawTable){if(state.busy||(state.view?.model_result&&!state.editingResult))return;const input=event.target,i=Number(input.dataset.boxIndex),field=input.dataset.boxField,s=state.draft[i];if(!s||field===undefined)return;if(field==='kind'){if((s.kind==='person')!==(input.value==='person'))delete s.id;s.kind=input.value;}else s.bbox[Number(field)]=Number(input.value);s.origin='manual';edited(redrawTable);}
  $('direct-boxes').addEventListener('input',event=>updateBox(event,false));
  $('direct-boxes').addEventListener('change',event=>updateBox(event,true));
  $('direct-boxes').addEventListener('click',event=>{if(state.busy||(state.view?.model_result&&!state.editingResult))return;const button=event.target.closest('[data-box-delete]');if(!button||button.dataset.boxDelete===undefined)return;state.draft.splice(Number(button.dataset.boxDelete),1);edited();});
  function point(event){const image=$('direct-editor-image'),rect=image.getBoundingClientRect(),f=state.view.original_frame||state.view.frames[0];return [Math.round(Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width))*f.width),Math.round(Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))*f.height)];}
  $('direct-box-canvas').addEventListener('pointerdown',event=>{if(state.busy||!state.view||(state.view.model_result&&!state.editingResult)||!$('direct-editor-image').naturalWidth)return;state.drag=point(event);event.currentTarget.setPointerCapture?.(event.pointerId);event.preventDefault();});
  $('direct-box-canvas').addEventListener('pointerup',event=>{if(!state.drag||state.busy)return;if(state.draft.length>=12){state.drag=null;message('At most 12 boxes can be submitted. Delete or replace a box to include another subject.',true);return;}const start=state.drag,end=point(event);state.drag=null;const bbox=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];if(bbox[2]-bbox[0]<2||bbox[3]-bbox[1]<2)return;state.draft.push({kind:$('direct-new-kind').value||'person',bbox,origin:'manual'});edited();});
  $('direct-box-canvas').addEventListener('pointercancel',()=>{state.drag=null;});
  for(const action of ['accept','reject','reset','export'])$('direct-'+action).addEventListener('click',()=>mutate(action));
  $('direct-library-refresh').addEventListener('click',loadLibrary);$('direct-library').addEventListener('click',event=>{const button=event.target.closest('[data-direct-open]');if(button&&!button.disabled)open(button.dataset.directOpen);});
  $('refresh-chatgpt').addEventListener('click',connection);
  (async()=>{await connection();await loadLibrary();let id;try{id=root.localStorage.getItem(storageKey);}catch{}if(id&&!state.selectedFile)await open(id);actions();})();
})(typeof window!=='undefined'?window:globalThis);
