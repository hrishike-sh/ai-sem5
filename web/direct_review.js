(function(root){
  'use strict';
  const common=typeof module!=='undefined'&&module.exports?require('./violation_lens.js'):root.ViolationLens;
  const {escapeHTML:esc,localURL,timestamp}=common;
  const list=value=>Array.isArray(value)?value:[];
  const kindLabels=Object.freeze({
    person_on_two_wheeler:'Rider visible',
    bare_head_of_rider_visible:'Uncovered head visible',
    helmet_on_rider_visible:'Helmet visible on rider',
    crowded_rider_load_visible:'Crowded rider arrangement',
    rider_position_appears_unstable:'Possibly unstable rider position',
    cargo_appears_unsecured:'Possibly unsecured cargo',
    rider_association_unclear:'Rider association unclear',
    head_protection_unclear:'Head protection unclear',
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
      return `<li><strong>${esc(label)}</strong><p>${esc(item.text)}</p><small>Evidence: ${refs.map(esc).join(', ')}</small>${uncertainty.length?`<p class="missing">Uncertainty: ${uncertainty.map(esc).join(' · ')}</p>`:''}</li>`;
    }).join('');
    return `<h3>Astra image assessment</h3><p class="hint">Saved model output · ${esc(result.model||'Model not reported')}</p><p>These are fallible visual interpretations of the selected image. They do not establish legal guilt, speed, or permitted travel direction.</p><h4>Visible observations</h4>${list(result.observations).length?`<ul>${entries(result.observations)}</ul>`:'<p>No structured observations were returned.</p>'}<h4>Suspected issues requiring review</h4>${list(result.suspected_issues).length?`<ul>${entries(result.suspected_issues)}</ul>`:'<p>No suspected issues were returned. This does not establish that the scene is free of violations.</p>'}<h4>Limitations</h4><ul>${list(result.limitations).map(text=>`<li>${esc(text)}</li>`).join('')}</ul>`;
  }
  function subjectName(subject){return `${subject.kind.charAt(0).toUpperCase()+subject.kind.slice(1)} ${subject.id||'(new)'}`;}
  function renderSubjects(result,frames,annotations){
    const ids=new Set(list(frames).map(f=>f.id));
    const statuses={unclear:'Unclear — evidence is insufficient',no_visible_violation:'No visible violation for this subject',suspected_issue:'Suspected issue — human review required'};
    const entries=items=>list(items).map(item=>{const refs=list(item.evidence_ids);return !refs.length||refs.some(id=>!ids.has(id))?'<li>Assessment item unavailable: evidence does not match this review.</li>':`<li>${esc(item.text)}<small>Evidence: ${refs.map(id=>evidenceReference(id,frames,annotations)).join(', ')}${list(item.related_subject_ids).length?' · Related subjects: '+item.related_subject_ids.map(esc).join(', '):''}</small>${list(item.uncertainty_text).length?`<p class="missing">${item.uncertainty_text.map(esc).join(' · ')}</p>`:''}</li>`;}).join('');
    return `<h3>Astra assessment by subject</h3><p class="hint">${esc(result.model||'Saved model output')}. Interpretations require human review.</p>${result.subject_results.map(subject=>`<section class="subject-result"><h4>${esc(list(annotations?.subjects).find(s=>s.id===(subject.subject_id||subject.id))?subjectName(list(annotations.subjects).find(s=>s.id===(subject.subject_id||subject.id))):(subject.subject_id||subject.id))}</h4><p><strong>${esc(statuses[subject.status]||'Unclear — status unavailable')}</strong></p><p>Role: ${esc(({rider:'Rider',pedestrian:'Pedestrian',person_unclear:'Person — role unclear',vehicle:'Vehicle'})[subject.role]||'Unclear')}${subject.associated_vehicle_id?` · Associated vehicle: ${esc(subject.associated_vehicle_id)}`:''}</p><h5>Observations</h5>${list(subject.observations).length?`<ul>${entries(subject.observations)}</ul>`:'<p>No observations recorded for this subject.</p>'}<h5>Suspected violations</h5>${list(subject.suspected_issues).length?`<ul>${entries(subject.suspected_issues)}</ul>`:'<p>No suspected violations recorded for this subject. This is not proof of compliance.</p>'}${list(subject.uncertainty_text).length?`<p class="missing">Uncertainty: ${subject.uncertainty_text.map(esc).join(' · ')}</p>`:''}</section>`).join('')}<h4>Limitations</h4><ul>${list(result.limitations).map(text=>`<li>${esc(text)}</li>`).join('')}</ul>`;
  }
  function canSend(view,loaded,consent,available,busy){return !!view&&!view.model_result&&list(view.annotations?.subjects).length>0&&view.source_readiness?.available!==false&&!busy&&available===true&&consent===true&&list(view.frames).length>0&&view.frames.every(f=>f.available!==false&&f.url&&loaded.has(f.id));}
  const helpers={sourceKind,selectedTime,validateView,renderPreview,renderResult,canSend};
  if(typeof module!=='undefined'&&module.exports)module.exports=helpers;
  if(!root.document)return;
  const $=id=>root.document.getElementById(id),token=root.document.querySelector('meta[name="traffic-token"]')?.content;
  if(!$('direct-panel'))return;
  const state={view:null,media:null,file:null,config:null,available:false,busy:false,upload:null,generation:0,loaded:new Set(),libraryBusy:false,draft:[],dirty:false,editingResult:false,drag:null};
  const storageKey='violation-lens-direct-review';
  function message(text,error=false){$('direct-message').textContent=text;$('direct-message').classList.toggle('error',error);}
  async function api(path,data){const response=await root.fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Traffic-Token':token},body:JSON.stringify(data)});let value;try{value=await response.json();}catch{throw Error('The server returned an unreadable response. Recheck the connection and retry.');}if(!response.ok)throw Error(value.error||value.message||'The request failed.');return value;}
  function actions(){
    const file=$('direct-file').files?.[0],kind=sourceKind(file?.name),view=state.view;
    $('direct-prepare').disabled=state.busy||!state.config||!kind||!$('direct-rights').checked;
    $('direct-prepare').textContent=state.media&&file===state.file?'Prepare selected frame':'Upload and preview';
    for(const id of ['direct-file','direct-time','direct-rights','direct-consent','direct-note'])$(id).disabled=state.busy;
    $('direct-cancel').hidden=!state.upload;
    $('direct-send').disabled=state.dirty||!canSend(view,state.loaded,$('direct-consent').checked,state.available,state.busy);
    const editingDisabled=state.busy||!view||(!!view.model_result&&!state.editingResult)||view.source_readiness?.available===false;
    $('direct-annotate').disabled=editingDisabled||!!view?.annotations;
    $('direct-save-boxes').disabled=editingDisabled||!state.dirty;
    $('direct-add-box').disabled=editingDisabled||state.draft.length>=12;
    $('direct-new-kind').disabled=editingDisabled;
    for(const input of $('direct-boxes').querySelectorAll('input,select,button'))input.disabled=editingDisabled;
    $('direct-send').hidden=!!view?.model_result;
    $('direct-consent').closest('label').hidden=!!view?.model_result;
    for(const action of ['accept','reject','reset','export'])$('direct-'+action).disabled=state.busy||state.dirty||!view?.model_result;
    $('direct-edit-boxes').hidden=!view?.model_result||state.editingResult;
    $('direct-edit-boxes').disabled=state.busy||view?.source_readiness?.available===false;
    $('direct-cancel-box-edit').hidden=!state.editingResult;
    $('direct-cancel-box-edit').disabled=state.busy;
    for(const button of $('direct-library').querySelectorAll('[data-direct-open]'))button.disabled=state.busy;
  }
  function clearView(){state.view=null;state.drag=null;state.dirty=false;state.editingResult=false;state.loaded.clear();$('direct-consent').checked=false;for(const id of ['direct-evidence','direct-result','direct-decision'])$(id).hidden=true;}
  function remember(id){try{root.localStorage.setItem(storageKey,id);}catch{message('Review saved on the server. This browser cannot remember it; reopen it from saved image reviews.');}}
  const subjectKinds=['person','motorcycle','bicycle','car','bus','truck'];
  function drawEditor(redrawTable=true){
    if(redrawTable)$('direct-boxes').innerHTML=state.draft.map((s,i)=>`<tr><th scope="row">${esc(subjectName(s))}</th><td><select aria-label="Subject ${i+1} type" data-box-index="${i}" data-box-field="kind">${subjectKinds.map(kind=>`<option value="${kind}"${s.kind===kind?' selected':''}>${esc(kind)}</option>`).join('')}</select></td>${s.bbox.map((n,j)=>`<td><input type="number" min="0" step="1" value="${esc(n)}" aria-label="Subject ${i+1} ${['left','top','right','bottom'][j]}" data-box-index="${i}" data-box-field="${j}"></td>`).join('')}<td><button type="button" class="secondary" data-box-delete="${i}">Delete</button></td></tr>`).join('');
    const original=state.view?.original_frame||state.view?.frames[0];
    const width=state.view?.annotations?.image_width||original?.width,height=state.view?.annotations?.image_height||original?.height;
    $('direct-box-overlay').innerHTML=width&&height?state.draft.map(s=>`<span class="subject-box" style="left:${100*s.bbox[0]/width}%;top:${100*s.bbox[1]/height}%;width:${100*(s.bbox[2]-s.bbox[0])/width}%;height:${100*(s.bbox[3]-s.bbox[1])/height}%"><b>${esc(s.id||'New')}</b></span>`).join(''):'';
  }
  function edited(redrawTable=true){state.dirty=true;$('direct-consent').checked=false;drawEditor(redrawTable);$('direct-preview-status').textContent='Boxes changed. Save corrections to rebuild the exact context and crop previews before sending.';actions();}
  async function annotate(manual){
    const view=state.view;if(!view||state.busy||(view.model_result&&!(manual&&state.editingResult))||(!manual&&view.annotations))return;
    const generation=state.generation;state.busy=true;$('direct-consent').checked=false;actions();message(manual?'Saving boxes and preparing exact crops locally.':'Detecting and numbering subjects locally. No image is being sent to ChatGPT.');
    try{const data={review_id:view.review_id,fingerprint:view.fingerprint};if(manual)data.subjects=state.draft.map(s=>({...s,bbox:[...s.bbox]}));const result=await api('/api/direct-annotate',data);if(generation!==state.generation||state.view!==view)return;if(result.review_id!==view.review_id)throw Error('The annotation response belongs to another review. Reopen this review.');draw(result);message(result.annotations?.notice||'Numbered subjects prepared. Check for missed or incorrect boxes and inspect every outbound image.');loadLibrary();}
    catch(error){if(generation===state.generation&&state.view===view)message(error.message+' You can add or correct boxes manually and save them.',true);}
    finally{if(generation===state.generation){state.busy=false;actions();}}
  }
  function draw(view){
    state.view=validateView(view,state.config?.evidence_origin);const current=state.view,generation=state.generation;
    state.loaded.clear();$('direct-consent').checked=false;$('direct-evidence').hidden=false;
    state.dirty=false;state.editingResult=false;state.draft=list(current.annotations?.subjects).map(s=>({id:s.id,kind:s.kind,bbox:[...s.bbox],origin:s.origin}));
    $('direct-annotation-editor').hidden=!!current.model_result;
    const original=current.original_frame||current.frames[0];$('direct-editor-image').src=original?.available!==false?original?.url||'':'';
    $('direct-annotation-notice').textContent=(current.annotations?.notice||'Detect subjects locally or draw missing boxes. A person box does not establish a rider role.')+(current.annotations?.detector?.truncated?' The detector returned more than 12 proposals; only its 12 highest-confidence proposals were retained. Review the image for missing subjects. At most 12 boxes can be submitted: delete or replace boxes to include missed subjects.':'');
    drawEditor();
    $('direct-source').textContent=`Source: ${current.source?.name||'Saved source'} · Image review: ${current.review_id}`;
    $('direct-source-warning').hidden=current.source_readiness?.available!==false;
    $('direct-source-warning').textContent=current.source_readiness?.available===false?'Historical assessment and your decision remain available. Current original frames cannot be verified. '+(current.source_readiness.reason||'Source unavailable.') : '';
    $('direct-provenance').hidden=!current.synthetic;
    $('direct-provenance').textContent=current.synthetic?'Synthetic source. This image review does not demonstrate real-world detection accuracy.':'';
    $('direct-preview').innerHTML=renderPreview(current);
    $('direct-preview-status').textContent='Loading the exact prepared image. Inspect it before consenting to send.';
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
    $('direct-account').textContent=state.available?'Using your ChatGPT plan. This action requests Astra image review.':'Astra image review needs your ChatGPT connection. Use Connect ChatGPT above, then Recheck connection. '+(llm?.message||'');
    if(!state.config)message('Could not connect to the local server. Check that it is running, then choose Recheck connection.',true);actions();
  }
  function upload(file){return new Promise((resolve,reject)=>{
    const xhr=new root.XMLHttpRequest();state.upload=xhr;$('direct-upload-progress').hidden=false;$('direct-upload-progress').value=0;actions();
    xhr.open('POST','/api/upload?name='+encodeURIComponent(file.name));xhr.setRequestHeader('X-Traffic-Token',token);xhr.setRequestHeader('Content-Type','application/octet-stream');
    xhr.upload.onprogress=event=>{if(event.lengthComputable){$('direct-upload-progress').value=Math.round(event.loaded/event.total*100);message('Uploading source: '+$('direct-upload-progress').value+'%.');}};
    xhr.onload=()=>{let body;try{body=JSON.parse(xhr.responseText);}catch{return reject(Error('The upload response could not be read. Retry upload.'));}if(xhr.status<200||xhr.status>=300)return reject(Error(body.error||'Upload failed. Retry upload.'));resolve(body);};
    xhr.onerror=()=>reject(Error('Upload failed. Check the local server and retry.'));xhr.onabort=()=>reject(Error('Upload cancelled. You can choose another source or retry.'));xhr.send(file);
  });}
  async function prepare(){
    if($('direct-prepare').disabled||state.busy)return;
    const file=$('direct-file').files?.[0];let time;try{time=selectedTime($('direct-time').value,sourceKind(file?.name));}catch(error){message(error.message,true);return;}
    const generation=++state.generation;state.busy=true;actions();message('Preparing your local preview. No image is being sent to ChatGPT.');
    try{
      if(!state.media||state.file!==file){const media=await upload(file);if(generation!==state.generation)return;if(!media.id||!media.source_revision)throw Error('Upload identity is incomplete. Retry upload.');state.media=media;state.file=file;}
      state.upload=null;$('direct-upload-progress').hidden=true;actions();
      const view=await api('/api/direct-prepare',{media:state.media.id,source_revision:state.media.source_revision,timestamp:time});if(generation!==state.generation)return;draw(view);message('Preview prepared locally. Detect and number subjects, then check the boxes before sending.');loadLibrary();
    }catch(error){if(generation===state.generation)message(error.message,true);}
    finally{if(generation===state.generation){state.upload=null;state.busy=false;$('direct-upload-progress').hidden=true;actions();}}
  }
  async function open(id){if(state.busy)return;const generation=++state.generation;state.busy=true;clearView();actions();message('Opening saved image review. No new model request is made.');try{const view=await api('/api/direct-review?id='+encodeURIComponent(id));if(generation!==state.generation)return;if(view.review_id!==id)throw Error('The returned review does not match the selected review.');draw(view);message(view.model_result?'Saved Astra assessment restored. No new model request was made.':'Prepared review restored. Inspect its image before choosing to send.');}catch(error){if(generation===state.generation)message(error.message,true);}finally{if(generation===state.generation){state.busy=false;actions();}}}
  async function mutate(action){
    const view=state.view;if(!view||state.busy)return;
    if(action==='analyze'&&(state.dirty||!canSend(view,state.loaded,$('direct-consent').checked,state.available,false)))return;
    if(action!=='analyze'&&(!view.model_result||state.dirty))return;
    const generation=state.generation;state.busy=true;actions();message(action==='analyze'?'Astra is reviewing the selected image. Keep this page open while the request completes.':'Saving your local review.');
    try{
      const data={review_id:view.review_id,fingerprint:view.fingerprint};
      if(action==='analyze')data.consent=true;else if(action!=='export'){data.decision=action;data.note=$('direct-note').value;}
      const result=await api('/api/direct-'+(action==='analyze'?'analyze':action==='export'?'export':'decision'),data);
      if(generation!==state.generation||state.view!==view)return;
      if(action==='export'){const url=localURL(result.url,state.config?.evidence_origin);if(!url)throw Error('The export download address is invalid.');const link=root.document.createElement('a');link.href=url;link.download=result.filename||'astra-image-review.json';root.document.body.append(link);link.click();link.remove();message('Review JSON exported with the model assessment and separate human disposition.');}
      else{if(result.review_id!==view.review_id||result.fingerprint!==view.fingerprint)throw Error('The returned review identity changed. Reopen the saved review before continuing.');draw(result);message(action==='analyze'?'Astra assessment saved. Read the uncertainty and add your review decision.':'Your disposition was saved separately from the model assessment.');loadLibrary();}
    }catch(error){if(generation===state.generation)message(error.message+(action==='analyze'?' No alternate model or paid API fallback will be used. Reopen the saved review before retrying if the result is uncertain.':''),true);}
    finally{if(generation===state.generation){state.busy=false;actions();}}
  }
  async function loadLibrary(){if(state.libraryBusy)return;state.libraryBusy=true;$('direct-library-refresh').disabled=true;try{const response=await api('/api/direct-library');const reviews=Array.isArray(response)?response:list(response.reviews);$('direct-library').innerHTML=reviews.map(item=>`<div class="saved-run"><div><p>${esc(item.source?.name||'Saved image review')}</p><small>${esc(item.review_id)}${item.synthetic?' · Synthetic':''}</small></div><button type="button" class="secondary" data-direct-open="${esc(item.review_id)}">Open image review</button></div>`).join('');$('direct-library-message').textContent=reviews.length?'Reopening shows saved output without a new model request.':'No saved image reviews yet.';}catch(error){$('direct-library-message').textContent='Could not load image reviews: '+error.message;}finally{state.libraryBusy=false;$('direct-library-refresh').disabled=false;actions();}}
  $('direct-file').addEventListener('change',()=>{if(state.busy)return;state.generation++;state.media=null;state.file=null;clearView();const file=$('direct-file').files?.[0],kind=sourceKind(file?.name);$('direct-filename').textContent=file?.name||'No source selected.';$('direct-time-wrap').hidden=kind!=='video';$('direct-time').value='0';$('direct-rights').checked=false;message(kind?'Confirm permission, then upload and preview.':'Choose a JPG, PNG, MP4, MOV or AVI file.',!kind);actions();});
  $('direct-time').addEventListener('input',()=>{if(state.busy)return;state.generation++;clearView();message('Timestamp changed. Prepare the selected frame before sending.');actions();});
  for(const id of ['direct-rights','direct-consent'])$(id).addEventListener('change',actions);
  $('direct-prepare').addEventListener('click',prepare);$('direct-cancel').addEventListener('click',()=>state.upload?.abort());$('direct-send').addEventListener('click',()=>mutate('analyze'));
  $('direct-edit-boxes').addEventListener('click',()=>{if(state.busy||!state.view?.model_result||state.view.source_readiness?.available===false)return;state.editingResult=true;$('direct-consent').checked=false;$('direct-annotation-editor').hidden=false;message('Editing boxes. The current assessment remains saved until you save corrections. Saving changes clears that assessment and your review decision.');actions();});
  $('direct-cancel-box-edit').addEventListener('click',()=>{if(state.busy||!state.editingResult)return;draw(state.view);message('Box edits discarded. The saved assessment and decision are unchanged.');});
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
  (async()=>{await connection();await loadLibrary();let id;try{id=root.localStorage.getItem(storageKey);}catch{}if(id)await open(id);actions();})();
})(typeof window!=='undefined'?window:globalThis);
