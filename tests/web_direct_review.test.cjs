const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const direct=require('../web/direct_review.js');
const common=require('../web/violation_lens.js');
const frame={id:'image-1',timestamp:0,url:'/prepared/image.jpg',available:true,sha256:'abc',width:640,height:480};
const assessment={state:'reviewed',model:'gpt-6-astra',observations:[{kind:'rider_visible',text:'A rider is visible.',evidence_ids:['image-1'],uncertainties:['visibility'],uncertainty_text:['The head is partially obscured.']}],suspected_issues:[],limitations:['A single image cannot establish speed.']};
const view=(overrides={})=>({review_id:'review-1',fingerprint:'fingerprint-1',source:{name:'permitted.png',kind:'image'},frames:[{...frame}],model_result:null,review:{decision:null,note:''},source_readiness:{available:true},local_evidence:{version:1,state:'complete',subjects:[],model:{name:'Helmet specialist'}},annotations:{image_width:640,image_height:480,subjects:[{id:'P1',kind:'person',bbox:[10,20,100,200],origin:'detector',evidence_ids:['image-1']}]},...overrides});

function harness(respond=()=>undefined,saved=null){
  const elements=new Map(),requests=[],uploads=[],downloads=[],savedIDs=[];let latestView=null;
  function node(id){
    const item={id,hidden:false,disabled:false,checked:false,value:id==='direct-time'?'0':'',files:[],textContent:'',listeners:{},dataset:{},images:[],label:{hidden:false},classList:{toggle(){}},addEventListener(name,fn){this.listeners[name]=fn;},closest(){return this.label;},querySelectorAll(selector){return selector.startsWith('img')?this.images:[];},click(){downloads.push(this.href);},remove(){}};
    Object.defineProperty(item,'innerHTML',{configurable:true,get(){return this.html||'';},set(html){this.html=html;this.images=Array.from(html.matchAll(/data-direct-frame="([^"]+)"/g),match=>({...node('image'),dataset:{directFrame:match[1]},naturalWidth:0,complete:false}));}});
    return item;
  }
  function element(id){if(!elements.has(id))elements.set(id,node(id));return elements.get(id);}
  class Upload{constructor(){this.upload={};this.headers={};uploads.push(this);}open(method,url){this.method=method;this.url=url;}setRequestHeader(name,value){this.headers[name]=value;}send(body){this.body=body;}abort(){this.onabort();}finish(body={id:'media-1',source_revision:'revision-1',kind:'image'},status=200){this.status=status;this.responseText=JSON.stringify(body);this.onload();}}
  const window={ViolationLens:common,XMLHttpRequest:Upload,document:{getElementById:element,querySelector(){return {content:'csrf-direct'};},createElement(){return node('link');},body:{append(){}}},localStorage:{getItem(){return saved;},setItem(key,id){savedIDs.push({key,id});}},fetch:async(path,options={})=>{requests.push({path,options});let data=await respond(path,options);if(data===undefined&&path==='/api/direct-plates')data={...(latestView||view()),plate_evidence:{state:'complete',reason:'Local plate check complete.',subjects:[],source:{width:640,height:480}}};if(data===undefined&&path==='/api/direct-vehicle-types')data={...(latestView||view()),vehicle_type_evidence:{version:1,state:'complete',reason:'Local subtype check complete.',subjects:[],unassigned_detections:[],candidate_counts:{total:0,by_type:{}},limitations:[]}};if(data===undefined&&path==='/api/direct-local-models')data={...(latestView||view()),local_evidence:{version:1,state:'complete',subjects:[],model:{name:'Helmet specialist'}}};if(data===undefined)data=path==='/api/config'?{evidence_origin:'http://127.0.0.1:5000',violation_readiness:{ready:false}}:path==='/api/violation-llm'?{available:true}:path==='/api/direct-library'?{reviews:[]}:view();if(data.review_id)latestView=data;return {ok:!data.error,json:async()=>data};}};
  vm.runInContext(fs.readFileSync(require.resolve('../web/direct_review.js'),'utf8'),vm.createContext({window,URL}));
  const settle=()=>new Promise(resolve=>setImmediate(resolve));
  const click=async id=>{await element(id).listeners.click();await settle();};
  const loaded=()=>{for(const image of element('direct-preview').images){image.naturalWidth=640;image.listeners.load();}};
  return {element,requests,uploads,downloads,savedIDs,settle,click,loaded};
}

test('direct inputs accept only supported media and explicit finite video timestamps',()=>{
  assert.equal(direct.sourceKind('helmet.JPG'),'image');assert.equal(direct.sourceKind('clip.avi'),'video');assert.equal(direct.sourceKind('evil.svg'),null);
  assert.equal(direct.selectedTime('','image'),0);assert.equal(direct.selectedTime('2.5','video'),2.5);
  for(const input of ['',-1,'Infinity','garbage'])assert.throws(()=>direct.selectedTime(input,'video'));
});
test('direct prepared views enforce evidence identity and trusted file origins',()=>{
  assert.throws(()=>direct.validateView(view({fingerprint:''})));
  assert.throws(()=>direct.validateView(view({frames:[frame,frame]})));
  assert.throws(()=>direct.validateView(view({frames:[{...frame,sha256:null}]})));
  const checked=direct.validateView(view({frames:[{...frame,url:'https://evil.test/image.jpg'}]}),'http://127.0.0.1:5000');
  assert.equal(checked.frames[0].url,'');assert.equal(direct.canSend(checked,new Set(['image-1']),true,true,false),false);
  assert.equal(direct.validateView(view({frames:[{...frame,url:'http://127.0.0.1:5000/token/image.jpg'}]}),'http://127.0.0.1:5000').frames[0].url,'http://127.0.0.1:5000/token/image.jpg');
});
test('direct output escapes model text and shows evidence references and human uncertainty',()=>{
  const result={...assessment,observations:[{...assessment.observations[0],text:'<img src=x onerror=bad()>',kind:'<script>bad()</script>'}]};
  const html=direct.renderResult(result,[frame]);assert.doesNotMatch(html,/<script>|<img src=x/);assert.match(html,/&lt;img/);assert.match(html,/Evidence: image-1/);assert.match(html,/The head is partially obscured/);assert.match(html,/does not establish that the scene is free/);assert.match(html,/A single image cannot establish speed/);
  const mismatched=direct.renderResult({...assessment,observations:[{text:'unbound claim',evidence_ids:['other-image']}]},[frame]);assert.doesNotMatch(mismatched,/unbound claim/);assert.match(mismatched,/do not match this review/);
});
test('direct sending requires every loaded exact image, consent, connection, and an unchanged available source',()=>{
  const current=view({frames:[frame,{...frame,id:'image-2'}]});const loaded=new Set(['image-1','image-2']);
  assert.equal(direct.canSend(current,loaded,true,true,false),true);
  for(const args of [[current,new Set(['image-1']),true,true,false],[current,loaded,false,true,false],[current,loaded,true,false,false],[current,loaded,true,true,true],[{...current,model_result:assessment},loaded,true,true,false],[{...current,source_readiness:{available:false}},loaded,true,true,false]])assert.equal(direct.canSend(...args),false);
});
test('direct categories have readable headings without changing stored IDs or uncertainty',()=>{
  const labels={person_on_two_wheeler:'Rider visible',bare_head_of_rider_visible:'Uncovered head visible',helmet_on_rider_visible:'Helmet visible on rider',crowded_rider_load_visible:'Crowded rider arrangement',rider_position_appears_unstable:'Possibly unstable rider position',cargo_appears_unsecured:'Possibly unsecured cargo',rider_association_unclear:'Unmatched rider',head_protection_unclear:'Head protection needs review',no_relevant_road_user_visible:'No relevant road user clearly visible',helmet_possible:'Possible missing helmet',unsafe_rider_load_possible:'Possible unsafe rider arrangement',other_visible_concern:'Possible cargo safety concern'};
  for(const [kind,label] of Object.entries(labels)){
    const issue=['helmet_possible','unsafe_rider_load_possible','other_visible_concern'].includes(kind);
    const result={...assessment,observations:[],suspected_issues:[],[issue?'suspected_issues':'observations']:[{kind,text:'Recorded canonical statement.',evidence_ids:['image-1'],uncertainty_text:['Model interpretation needs review.']}]};
    const before=JSON.stringify(result),html=direct.renderResult(result,[frame]);
    assert.ok(html.includes('<strong>'+label+'</strong>'));assert.ok(!html.includes(kind));assert.match(html,/Evidence: image-1/);assert.match(html,/Model interpretation needs review/);assert.equal(JSON.stringify(result),before);
  }
  assert.match(direct.renderResult({...assessment,observations:[{kind:'toString',text:'Saved observation.',evidence_ids:['image-1']}]},[frame]),/<strong>Visual assessment<\/strong>/);
});
test('saved direct assessments reopen without inference or a current connection and retain historical decisions',async()=>{
  const h=harness(path=>path==='/api/violation-llm'?{available:false}:path.startsWith('/api/direct-review')?view({model_result:assessment,review:{decision:'reject',note:'Retain my note'},source_readiness:{available:false,reason:'Original file changed.'}}):undefined,'review-1');
  await h.settle();assert.match(h.element('direct-result').innerHTML,/A rider is visible/);assert.match(h.element('direct-message').textContent,/No new model request/);assert.match(h.element('direct-disposition').textContent,/rejected/);assert.equal(h.element('direct-note').value,'Retain my note');assert.equal(h.element('direct-source-warning').hidden,false);assert.match(h.element('direct-source-warning').textContent,/Original file changed/);assert.equal(h.element('direct-send').hidden,true);assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
});
test('loaded preview and explicit consent trigger one bound Astra request and prevent accidental repeat sends',async()=>{
  let release;const h=harness(path=>path==='/api/direct-analyze'?new Promise(resolve=>release=resolve):undefined,'review-1');await h.settle();
  h.element('direct-consent').checked=true;h.element('direct-consent').listeners.change();await h.click('direct-send');assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
  h.loaded();assert.equal(h.element('direct-send').disabled,false);
  const pending=h.element('direct-send').listeners.click();await h.settle();assert.equal(h.element('direct-send').disabled,true);await h.click('direct-send');
  const calls=h.requests.filter(r=>r.path==='/api/direct-analyze');assert.equal(calls.length,1);assert.deepEqual(JSON.parse(calls[0].options.body),{review_id:'review-1',fingerprint:'fingerprint-1',consent:true});assert.equal(calls[0].options.headers['X-Traffic-Token'],'csrf-direct');
  release(view({model_result:assessment}));await pending;await h.settle();assert.equal(h.element('direct-send').hidden,true);await h.click('direct-send');assert.equal(h.requests.filter(r=>r.path==='/api/direct-analyze').length,1);
});
test('a failed preview cannot be sent and timestamp changes invalidate the former consent and preview',async()=>{
  const h=harness(undefined,'review-1');await h.settle();h.loaded();h.element('direct-consent').checked=true;h.element('direct-consent').listeners.change();
  h.element('direct-preview').images[0].listeners.error();assert.equal(h.element('direct-send').disabled,true);assert.match(h.element('direct-preview-status').textContent,/could not load/);
  h.loaded();assert.equal(h.element('direct-send').disabled,false);h.element('direct-time').value='3';h.element('direct-time').listeners.input();assert.equal(h.element('direct-send').disabled,true);assert.equal(h.element('direct-consent').checked,false);assert.equal(h.element('direct-evidence').hidden,true);await h.click('direct-send');assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
});
test('raw source upload requires permission, works without detector models, and prepares locally without a model call',async()=>{
  const h=harness();await h.settle();const file={name:'street photo.png'};h.element('direct-file').files=[file];h.element('direct-file').listeners.change();assert.equal(h.element('direct-prepare').disabled,true);await h.click('direct-prepare');assert.equal(h.uploads.length,0);
  h.element('direct-rights').checked=true;const pending=h.element('direct-rights').listeners.change();assert.equal(h.element('direct-prepare').disabled,true);assert.equal(h.uploads.length,1);assert.equal(h.uploads[0].body,file);assert.equal(h.uploads[0].headers['X-Traffic-Token'],'csrf-direct');assert.equal(h.uploads[0].url,'/api/upload?name=street%20photo.png');h.uploads[0].finish();await pending;await h.settle();
  const prep=h.requests.find(r=>r.path==='/api/direct-prepare');assert.deepEqual(JSON.parse(prep.options.body),{media:'media-1',source_revision:'revision-1',timestamp:0});assert.equal(h.requests.some(r=>r.path.includes('analyze')||r.path==='/api/job'),false);assert.equal(h.element('direct-evidence').hidden,false);
});
test('cancelled direct upload releases controls without creating a preparation or model request',async()=>{
  const h=harness();await h.settle();h.element('direct-file').files=[{name:'video.mp4'}];h.element('direct-file').listeners.change();h.element('direct-rights').checked=true;h.element('direct-rights').listeners.change();const pending=h.element('direct-prepare').listeners.click();await h.click('direct-cancel');await pending;assert.equal(h.element('direct-prepare').disabled,false);assert.match(h.element('direct-message').textContent,/cancelled/);assert.equal(h.requests.some(r=>r.path==='/api/direct-prepare'),false);
});
test('direct decisions preserve model assessment and use bound identities, exports use the trusted local server',async()=>{
  const h=harness((path,options)=>path.startsWith('/api/direct-review')?view({model_result:assessment}):path==='/api/direct-decision'?view({model_result:assessment,review:{decision:JSON.parse(options.body).decision,note:JSON.parse(options.body).note}}):path==='/api/direct-export'?{url:'http://127.0.0.1:5000/token/review.json',filename:'review.json'}:undefined,'review-1');await h.settle();h.element('direct-note').value='Human note';await h.click('direct-accept');assert.match(h.element('direct-disposition').textContent,/accepted/);assert.match(h.element('direct-result').innerHTML,/A rider is visible/);assert.equal(h.element('direct-note').value,'Human note');const decision=h.requests.find(r=>r.path==='/api/direct-decision');assert.deepEqual(JSON.parse(decision.options.body),{review_id:'review-1',fingerprint:'fingerprint-1',decision:'accept',note:'Human note'});await h.click('direct-export');assert.deepEqual(h.downloads,['http://127.0.0.1:5000/token/review.json']);assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
});
test('direct analysis errors remain actionable and never invoke an alternate route or model',async()=>{
  const h=harness(path=>path==='/api/direct-analyze'?{error:'Astra is unavailable.'}:undefined,'review-1');await h.settle();h.loaded();h.element('direct-consent').checked=true;h.element('direct-consent').listeners.change();await h.click('direct-send');assert.match(h.element('direct-message').textContent,/Astra is unavailable/);assert.match(h.element('direct-message').textContent,/No alternate model or paid API fallback/);assert.equal(h.element('direct-evidence').hidden,false);assert.equal(h.requests.filter(r=>r.options.method==='POST').length,1);
});


test('unannotated and empty subject previews cannot be sent',()=>{
  for(const annotations of [null,{subjects:[]}])assert.equal(direct.canSend(view({annotations}),new Set(['image-1']),true,true,false),false);
});
test('corrections reset consent and require rebuilt exact previews',async()=>{
  const h=harness(path=>path==='/api/direct-annotate'?view({fingerprint:'corrected',frames:[{...frame,id:'context'},{...frame,id:'P1-crop'}]}):undefined,'review-1');await h.settle();h.loaded();h.element('direct-consent').checked=true;h.element('direct-consent').listeners.change();
  await h.click('direct-add-box');assert.equal(h.element('direct-consent').checked,false);assert.equal(h.element('direct-send').disabled,true);h.element('direct-consent').checked=true;await h.click('direct-send');assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
  await h.click('direct-save-boxes');const req=h.requests.find(r=>r.path==='/api/direct-annotate'),data=JSON.parse(req.options.body);assert.equal(data.subjects[0].id,'P1');assert.equal(data.subjects[1].id,undefined);assert.equal(data.subjects[1].origin,'manual');assert.equal(h.element('direct-consent').checked,false);assert.equal(h.element('direct-send').disabled,true);
  h.loaded();h.element('direct-consent').checked=true;h.element('direct-consent').listeners.change();assert.equal(h.element('direct-send').disabled,false);
});
test('per-ID assessments preserve unclear versus no visible violation and escape canonical text',()=>{
  const result={model:'Astra',subject_results:[{id:'P1',status:'unclear',role:'unclear',observations:[{text:'<script>unsafe()</script>',evidence_ids:['image-1']}],suspected_issues:[]},{id:'V1',status:'no_visible_violation',role:'vehicle',observations:[],suspected_issues:[]}],limitations:['<b>One frame</b>']};
  const html=direct.renderResult(result,[frame]);assert.match(html,/P1/);assert.match(html,/V1/);assert.match(html,/Needs review/);assert.match(html,/No visible violation for this subject/);assert.match(html,/&lt;script&gt;/);assert.doesNotMatch(html,/<script>|<b>One frame/);
});
test('detector failure preserves local manual editing without a model call',async()=>{
  const h=harness(path=>path==='/api/direct-annotate'?{error:'Local detector unavailable.'}:path.startsWith('/api/direct-review')?view({annotations:null}):undefined,'review-1');await h.settle();await h.click('direct-annotate');assert.match(h.element('direct-message').textContent,/Local detector unavailable/);assert.match(h.element('direct-message').textContent,/manually/);assert.equal(h.element('direct-add-box').disabled,false);assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
});
test('late image events from previous annotations cannot unlock sending',async()=>{
  const h=harness(path=>path.startsWith('/api/direct-review')?view({annotations:null}):undefined,'review-1');await h.settle();const old=h.element('direct-preview').images[0];await h.click('direct-annotate');h.element('direct-consent').checked=true;old.naturalWidth=640;old.listeners.load();assert.equal(h.element('direct-send').disabled,true);h.loaded();h.element('direct-consent').listeners.change();assert.equal(h.element('direct-send').disabled,false);
});


test('cross-category corrections request a new ID while same-family edits keep their identity',async()=>{
  const h=harness(undefined,'review-1');await h.settle();h.element('direct-boxes').listeners.change({target:{dataset:{boxIndex:'0',boxField:'kind'},value:'motorcycle'}});await h.click('direct-save-boxes');const data=JSON.parse(h.requests.find(r=>r.path==='/api/direct-annotate').options.body);assert.equal(data.subjects[0].id,undefined);assert.equal(data.subjects[0].kind,'motorcycle');assert.equal(data.subjects[0].origin,'manual');
});
test('annotation responses for another review cannot replace current evidence',async()=>{
  const h=harness(path=>path==='/api/direct-annotate'?view({review_id:'wrong-review',fingerprint:'wrong',frames:[{...frame,id:'wrong-frame'}]}):path.startsWith('/api/direct-review')?view({annotations:null}):undefined,'review-1');await h.settle();await h.click('direct-annotate');assert.match(h.element('direct-message').textContent,/another review/);assert.doesNotMatch(h.element('direct-preview').innerHTML,/wrong-frame/);assert.match(h.element('direct-source').textContent,/review-1/);assert.equal(h.element('direct-consent').checked,false);
});
test('per-subject issue cards show canonical uncertainty and associated IDs',()=>{
  const html=direct.renderResult({subject_results:[{id:'P1',role:'rider',associated_vehicle_id:'V1',status:'suspected_issue',observations:[],suspected_issues:[{text:'Possible helmet concern.',evidence_ids:['image-1'],related_subject_ids:['P1','V1']}],uncertainty_text:['Head partly obscured.']}],limitations:[]},[frame],view().annotations);
  assert.match(html,/Person P1/);assert.match(html,/Suspected issue - human review required/);assert.match(html,/Associated vehicle: V1/);assert.match(html,/Related subjects: P1, V1/);assert.match(html,/Head partly obscured/);
});


test('existing annotations disable redetection but keep correction controls available',async()=>{
  const h=harness(undefined,'review-1');await h.settle();assert.equal(h.element('direct-annotate').disabled,true);assert.equal(h.element('direct-add-box').disabled,false);await h.click('direct-annotate');assert.equal(h.requests.some(r=>r.path==='/api/direct-annotate'),false);await h.click('direct-add-box');assert.equal(h.element('direct-save-boxes').disabled,false);
});
test('truncated proposals disclose confidence selection and allow replacement within twelve boxes',async()=>{
  const subjects=Array.from({length:12},(_,i)=>({id:'P'+(i+1),kind:'person',bbox:[i,0,i+10,50],origin:'detector',evidence_ids:['image-1']}));
  const h=harness(path=>path.startsWith('/api/direct-review')?view({annotations:{subjects,image_width:640,image_height:480,detector:{truncated:true}}}):undefined,'review-1');await h.settle();assert.match(h.element('direct-annotation-notice').textContent,/12 highest-confidence proposals/);assert.match(h.element('direct-annotation-notice').textContent,/delete or replace boxes/);assert.equal(h.element('direct-add-box').disabled,true);await h.click('direct-add-box');assert.equal(h.element('direct-save-boxes').disabled,true);
  h.element('direct-boxes').listeners.click({target:{closest:()=>({dataset:{boxDelete:'0'}})}});assert.equal(h.element('direct-add-box').disabled,false);await h.click('direct-add-box');assert.equal(h.element('direct-add-box').disabled,true);
});


test('saved assessments can edit boxes and saving clears old decisions with fresh preview consent',async()=>{
  let saved=view({model_result:assessment,review:{decision:'accept',note:'Original note'}});
  const h=harness((path,options)=>path.startsWith('/api/direct-review')?saved:path==='/api/direct-annotate'?(saved=view({fingerprint:'revision-2',frames:[{...frame,id:'new-context'},{...frame,id:'new-crop'}]})):undefined,'review-1');await h.settle();
  assert.equal(h.element('direct-edit-boxes').hidden,false);assert.equal(h.element('direct-annotation-editor').hidden,true);await h.click('direct-edit-boxes');assert.equal(h.element('direct-annotation-editor').hidden,false);assert.match(h.element('direct-message').textContent,/Saving changes clears/);assert.match(h.element('direct-result').innerHTML,/A rider is visible/);
  h.element('direct-consent').checked=true;await h.click('direct-add-box');assert.equal(h.element('direct-consent').checked,false);for(const id of ['direct-accept','direct-reject','direct-reset','direct-export'])assert.equal(h.element(id).disabled,true);await h.click('direct-accept');await h.click('direct-export');assert.equal(h.requests.some(r=>r.path==='/api/direct-decision'||r.path==='/api/direct-export'),false);
  const old=h.element('direct-preview').images[0];await h.click('direct-save-boxes');assert.equal(h.element('direct-result').hidden,true);assert.equal(h.element('direct-decision').hidden,true);assert.equal(h.element('direct-note').value,'');assert.equal(h.element('direct-consent').checked,false);h.element('direct-consent').checked=true;old.naturalWidth=640;old.listeners.load();assert.equal(h.element('direct-send').disabled,true);h.loaded();h.element('direct-consent').listeners.change();assert.equal(h.element('direct-send').disabled,false);
  const reopened=harness(path=>path.startsWith('/api/direct-review')?saved:undefined,'review-1');await reopened.settle();assert.equal(reopened.element('direct-result').hidden,true);assert.equal(reopened.element('direct-consent').checked,false);assert.equal(reopened.element('direct-send').disabled,true);
});
test('discarding saved-result box edits preserves assessment and decision without a write',async()=>{
  const h=harness(path=>path.startsWith('/api/direct-review')?view({model_result:assessment,review:{decision:'reject',note:'Keep'}}):undefined,'review-1');await h.settle();await h.click('direct-edit-boxes');await h.click('direct-add-box');await h.click('direct-cancel-box-edit');assert.equal(h.element('direct-annotation-editor').hidden,true);assert.match(h.element('direct-result').innerHTML,/A rider is visible/);assert.match(h.element('direct-disposition').textContent,/rejected/);assert.equal(h.element('direct-note').value,'Keep');assert.equal(h.element('direct-accept').disabled,false);assert.equal(h.requests.some(r=>r.options.method==='POST'),false);
});
test('failed saved-result corrections retain old result and block stale disposition until discard',async()=>{
  const h=harness(path=>path.startsWith('/api/direct-review')?view({model_result:assessment}):path==='/api/direct-annotate'?{error:'Invalid box.'}:undefined,'review-1');await h.settle();await h.click('direct-edit-boxes');await h.click('direct-add-box');await h.click('direct-save-boxes');assert.match(h.element('direct-message').textContent,/Invalid box/);assert.match(h.element('direct-result').innerHTML,/A rider is visible/);assert.equal(h.element('direct-accept').disabled,true);assert.equal(h.element('direct-save-boxes').disabled,false);
});


test('numbered evidence uses compact labels while exact IDs remain in titles and data',()=>{
  const contextID='direct-0123456789abcdef0123456789abcdef-context',cropID='direct-0123456789abcdef0123456789abcdef-crop-P3';
  const frames=[{...frame,id:contextID,role:'context',subject_ids:['P3']},{...frame,id:cropID,role:'subject_crop',subject_ids:['P3']}];
  const result={subject_results:[{id:'P3',role:'person_unclear',status:'unclear',observations:[{text:'Subject unclear.',evidence_ids:[contextID,cropID]}],suspected_issues:[]}],limitations:[]};
  const before=JSON.stringify({frames,result}),preview=direct.renderPreview(view({frames})),html=direct.renderResult(result,frames);
  for(const markup of [preview,html]){assert.match(markup,/>Context</);assert.match(markup,/>P3 crop</);assert.ok(markup.includes('title="'+contextID+'"'));assert.ok(markup.includes('title="'+cropID+'"'));assert.doesNotMatch(markup.replace(/<[^>]*>/g,''),/direct-012345/);}
  assert.equal(JSON.stringify({frames,result}),before);assert.ok(preview.includes('data-direct-frame="'+cropID+'"'));
});
test('crop labels use bound annotations when subject metadata is absent and escape ID titles',()=>{
  const crop={...frame,id:'crop-"<unsafe>',role:'subject_crop'};const annotations={subjects:[{id:'P3',evidence_ids:['context',crop.id]}]};
  const html=direct.renderPreview(view({frames:[crop],annotations}));assert.match(html,/>P3 crop</);assert.match(html,/title="crop-&quot;&lt;unsafe&gt;"/);assert.doesNotMatch(html,/<unsafe>/);
});


test('typing a coordinate invalidates saved-result actions immediately without replacing the input table',async()=>{
  const h=harness(path=>path.startsWith('/api/direct-review')?view({model_result:assessment}):undefined,'review-1');await h.settle();await h.click('direct-edit-boxes');h.element('direct-consent').checked=true;
  const table=h.element('direct-boxes'),before=table.innerHTML;let replacements=0;Object.defineProperty(table,'innerHTML',{get:()=>before,set:()=>{replacements++;},configurable:true});
  const input={dataset:{boxIndex:'0',boxField:'0'},value:'17'};table.listeners.input({target:input});assert.equal(replacements,0);assert.equal(input.value,'17');assert.equal(h.element('direct-consent').checked,false);assert.equal(h.element('direct-save-boxes').disabled,false);for(const action of ['accept','reject','reset','export'])assert.equal(h.element('direct-'+action).disabled,true);assert.match(h.element('direct-preview-status').textContent,/Boxes changed/);
  await h.click('direct-export');assert.equal(h.requests.some(r=>r.path==='/api/direct-export'),false);await h.click('direct-save-boxes');const sent=JSON.parse(h.requests.find(r=>r.path==='/api/direct-annotate').options.body);assert.equal(sent.subjects[0].bbox[0],17);
});

test('primary local cards preserve raw scores and distinguish absent, weak and ambiguous helmet evidence',()=>{
  const subjects=['P1','P2','P3','P4','P5'].map((id,i)=>({id,kind:'person',origin:'detector',bbox:[i,0,i+5,10]}));
  subjects.push({id:'V1',kind:'motorcycle',origin:'detector',bbox:[0,0,50,50]},{id:'V2',kind:'motorcycle',origin:'manual',bbox:[50,0,100,50]});
  const evidence={state:'complete',model:{name:'Specialist <safe>'},subjects:[{id:'P1',helmet:{status:'helmet',detections:[{label:'helmet',model_score:0.83}],reason:'Visible head.'},vehicle_candidates:[{id:'V1',relationship:'geometric_unconfirmed'}]},{id:'P2',helmet:{status:'not_observed',detections:[],reason:'No matching head detection.'}},{id:'P3',helmet:{status:'low_confidence',detections:[{label:'no_helmet',model_score:0.31}]}},{id:'P4',helmet:{status:'unclear',detections:[{label:'helmet',model_score:0.61},{label:'no_helmet',model_score:0.59}]}},{id:'P5',helmet:{status:'unavailable',reason:'Crop unavailable.'}}],vehicle_counts:{truncated:true},limitations:['<script>bad()</script>']};
  const current=view({annotations:{subjects},local_evidence:evidence}),before=JSON.stringify(current),html=direct.renderLocalEvidence(current);
  assert.match(html,/>0\.830</);assert.match(html,/title="Exact raw model score: 0\.83"/);assert.doesNotMatch(html,/83%/);assert.match(html,/raw model score/);assert.match(html,/Not observed/);assert.match(html,/Low-confidence detection/);assert.match(html,/Conflicting predictions/);assert.match(html,/Unavailable/);assert.match(html,/Geometry only; rider association unconfirmed/);assert.match(html,/motorcycle<\/th><td>2<\/td><td>1<\/td><td>1/);assert.match(html,/not a unique vehicle count across the clip/);assert.match(html,/truncated to 12/);assert.doesNotMatch(html,/<script>|<safe>/);assert.equal(JSON.stringify(current),before);
});

test('Astra requires a completed or explicitly unavailable local check, including legacy reviews',()=>{
  for(const evidence of [null,{state:'running'},{state:'failed'}])assert.equal(direct.canSend(view({local_evidence:evidence}),new Set(['image-1']),true,true,false),false);
  assert.equal(direct.canSend(view({local_evidence:{state:'unavailable'}}),new Set(['image-1']),true,true,false),true);
  assert.equal(direct.localReady(view({local_evidence:null,annotations:{local_evidence:{state:'complete'}}})),true);
});

test('numbering automatically runs bound local specialist endpoint and waits before fresh Astra consent',async()=>{
  let release;
  const h=harness(path=>path.startsWith('/api/direct-review')?view({annotations:null,local_evidence:null}):path==='/api/direct-annotate'?view({fingerprint:'numbered',local_evidence:null}):path==='/api/direct-local-models'?new Promise(resolve=>release=resolve):undefined,'review-1');
  await h.settle();const pending=h.element('direct-annotate').listeners.click();await h.settle();
  const call=h.requests.find(r=>r.path==='/api/direct-local-models');assert.deepEqual(JSON.parse(call.options.body),{review_id:'review-1',fingerprint:'numbered'});assert.equal(call.options.headers['X-Traffic-Token'],'csrf-direct');assert.match(h.element('direct-message').textContent,/Checking helmets locally/);assert.equal(h.element('direct-consent').disabled,true);assert.equal(h.element('direct-send').disabled,true);assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
  release(view({fingerprint:'locally-checked'}));await pending;h.loaded();assert.equal(h.element('direct-consent').checked,false);assert.equal(h.element('direct-send').disabled,true);h.element('direct-consent').checked=true;h.element('direct-consent').listeners.change();assert.equal(h.element('direct-send').disabled,false);
});

test('local endpoint failure offers retry and cannot silently unlock Astra',async()=>{
  let attempts=0;
  const h=harness(path=>path.startsWith('/api/direct-review')?view({local_evidence:null}):path==='/api/direct-local-models'?(++attempts===1?{error:'Local model runtime failed.'}:view({local_evidence:{state:'unavailable',reason:'Weights not installed.',subjects:[]}})):undefined,'review-1');
  await h.settle();h.loaded();await h.click('direct-local-run');assert.match(h.element('direct-message').textContent,/Retry with Run specialist models/);assert.equal(h.element('direct-local-run').disabled,false);assert.equal(h.element('direct-send').disabled,true);
  await h.click('direct-local-run');assert.match(h.element('direct-local-output').innerHTML,/Weights not installed/);h.loaded();h.element('direct-consent').checked=true;h.element('direct-consent').listeners.change();assert.equal(h.element('direct-send').disabled,false);assert.equal(h.requests.filter(r=>r.path==='/api/direct-local-models').length,2);assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
});

test('local scores reopen without rerunning and remain separate from saved Astra interpretation',async()=>{
  const evidence={state:'complete',subjects:[{id:'P1',helmet:{status:'helmet',detections:[{label:'helmet',model_score:0.83}]}}]};
  const h=harness(path=>path.startsWith('/api/direct-review')?view({local_evidence:evidence,model_result:assessment}):undefined,'review-1');await h.settle();assert.match(h.element('direct-local-output').innerHTML,/0\.83/);assert.match(h.element('direct-result').innerHTML,/Optional LLM interpretation/);assert.equal(h.requests.some(r=>r.path==='/api/direct-local-models'||r.path==='/api/direct-analyze'),false);
  await h.click('direct-edit-boxes');await h.click('direct-add-box');assert.match(h.element('direct-local-output').innerHTML,/Boxes changed/);assert.equal(h.element('direct-result').hidden,true);await h.click('direct-cancel-box-edit');assert.match(h.element('direct-local-output').innerHTML,/0\.83/);assert.equal(h.element('direct-result').hidden,false);
});

test('local evidence exports without requesting an Astra assessment',async()=>{
  const h=harness(path=>path==='/api/direct-export'?{url:'/prepared/review.json',filename:'review.json'}:undefined,'review-1');await h.settle();assert.equal(h.element('direct-export').disabled,false);await h.click('direct-export');const call=h.requests.find(r=>r.path==='/api/direct-export');assert.deepEqual(JSON.parse(call.options.body),{review_id:'review-1',fingerprint:'fingerprint-1'});assert.deepEqual(h.downloads,['/prepared/review.json']);assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
});

test('legacy saved Astra reviews can explicitly run specialists and clear downstream results',async()=>{
  const h=harness(path=>path.startsWith('/api/direct-review')?view({local_evidence:null,model_result:assessment,review:{decision:'accept',note:'Old decision'}}):path==='/api/direct-local-models'?view({fingerprint:'local-refresh'}):undefined,'review-1');await h.settle();
  assert.equal(h.element('direct-local-run').disabled,false);assert.match(h.element('direct-local-run').textContent,/clears LLM review/);assert.equal(h.element('direct-local-rerun-notice').hidden,false);assert.match(h.element('direct-local-output').innerHTML,/Run specialist models/);
  await h.click('direct-local-run');assert.equal(h.element('direct-result').hidden,true);assert.equal(h.element('direct-decision').hidden,true);assert.equal(h.element('direct-note').value,'');assert.equal(h.element('direct-consent').checked,false);assert.equal(h.element('direct-local-rerun-notice').hidden,true);h.loaded();assert.equal(h.element('direct-send').disabled,true);assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
});

test('unavailable evidence with a saved Astra assessment can retry while a failure preserves the saved review',async()=>{
  let attempts=0;
  const h=harness(path=>path.startsWith('/api/direct-review')?view({local_evidence:{state:'unavailable',reason:'Local runtime unavailable.',subjects:[]},model_result:assessment,review:{decision:'reject',note:'Keep until saved'}}):path==='/api/direct-local-models'?(++attempts===1?{error:'Local service disconnected.'}:view({fingerprint:'retry-saved'})):undefined,'review-1');await h.settle();assert.equal(h.element('direct-local-run').disabled,false);await h.click('direct-local-run');assert.equal(h.element('direct-result').hidden,false);assert.match(h.element('direct-result').innerHTML,/A rider is visible/);assert.equal(h.element('direct-note').value,'Keep until saved');assert.equal(h.element('direct-local-run').disabled,false);
  await h.click('direct-local-run');assert.equal(h.element('direct-result').hidden,true);assert.equal(h.element('direct-note').value,'');assert.equal(h.requests.filter(r=>r.path==='/api/direct-local-models').length,2);
});


test('local score display uses three decimals while exact values remain available and unchanged',()=>{
  const evidence={state:'complete',subjects:[{id:'P1',helmet:{status:'helmet',detections:[{label:'helmet',model_score:0.8032053112983704}]}}]};
  const before=JSON.stringify(evidence),html=direct.renderLocalEvidence(view({local_evidence:evidence}));
  assert.match(html,/>0\.803</);assert.match(html,/title="Exact raw model score: 0\.8032053112983704"/);assert.equal(JSON.stringify(evidence),before);assert.doesNotMatch(html,/[^\x00-\x7F]/);
});

test('dropping an image queues it until permission then runs each local stage once without hosted sends',async()=>{
  const h=harness(path=>path==='/api/direct-prepare'?view({annotations:null,local_evidence:null}):path==='/api/direct-annotate'?view({fingerprint:'numbered',local_evidence:null}):undefined);await h.settle();
  const file={name:'dropped.jpg'};const event={preventDefault(){},dataTransfer:{files:[file]}};await h.element('direct-drop-zone').listeners.drop(event);assert.equal(h.uploads.length,0);assert.match(h.element('direct-message').textContent,/Image queued/);
  h.element('direct-rights').checked=true;const pending=h.element('direct-rights').listeners.change();assert.equal(h.uploads.length,1);assert.equal(h.uploads[0].body,file);assert.equal(h.element('direct-feature-localization').dataset.status,'running');h.uploads[0].finish();await pending;await h.settle();
  for(const path of ['/api/direct-prepare','/api/direct-annotate','/api/direct-local-models'])assert.equal(h.requests.filter(r=>r.path===path).length,1);
  assert.equal(h.element('direct-feature-localization').dataset.status,'complete');assert.equal(h.element('direct-feature-helmet').dataset.status,'complete');assert.equal(h.element('direct-feature-vehicles').dataset.status,'complete');assert.equal(h.element('direct-report').hidden,false);assert.equal(h.element('direct-consent').checked,false);
  h.element('direct-rights').checked=false;await h.element('direct-rights').listeners.change();h.element('direct-rights').checked=true;await h.element('direct-rights').listeners.change();await h.element('direct-drop-zone').listeners.drop(event);assert.equal(h.uploads.length,1);assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
});

test('permitted image replacement aborts previous upload and ignores its late completion',async()=>{
  const h=harness(path=>path==='/api/direct-prepare'?view({annotations:null,local_evidence:null}):undefined);await h.settle();h.element('direct-rights').checked=true;
  const first=h.element('direct-drop-zone').listeners.drop({preventDefault(){},dataTransfer:{files:[{name:'old.jpg'}]}});assert.equal(h.uploads.length,1);
  const second=h.element('direct-drop-zone').listeners.drop({preventDefault(){},dataTransfer:{files:[{name:'new.jpg'}]}});assert.equal(h.uploads.length,2);h.uploads[0].finish({id:'old',source_revision:'old-revision'});h.uploads[1].finish({id:'new',source_revision:'new-revision'});await Promise.all([first,second]);
  const calls=h.requests.filter(r=>r.path==='/api/direct-prepare');assert.equal(calls.length,1);assert.equal(JSON.parse(calls[0].options.body).media,'new');assert.equal(h.element('direct-filename').textContent,'new.jpg');
});

test('replacement waits for active inference before uploading and cannot overwrite the newer source report',async()=>{
  let release,preparations=0;
  const h=harness(path=>path==='/api/direct-prepare'?view({review_id:++preparations===1?'old-review':'new-review',source:{name:preparations===1?'old.jpg':'new.jpg'},annotations:null,local_evidence:null}):path==='/api/direct-annotate'&&preparations===1?new Promise(resolve=>release=resolve):path==='/api/direct-annotate'?view({review_id:'new-review',source:{name:'new.jpg'}}):undefined);await h.settle();h.element('direct-rights').checked=true;
  const first=h.element('direct-drop-zone').listeners.drop({preventDefault(){},dataTransfer:{files:[{name:'old.jpg'}]}});h.uploads[0].finish();await h.settle();
  const second=h.element('direct-drop-zone').listeners.drop({preventDefault(){},dataTransfer:{files:[{name:'new.jpg'}]}});assert.equal(h.uploads.length,1);assert.match(h.element('direct-feature-localization-detail').textContent,/Waiting for the previous local check/);release(view({review_id:'old-review',source:{name:'old.jpg'}}));await first;await h.settle();assert.equal(h.uploads.length,2);h.uploads[1].finish();await second;assert.match(h.element('direct-source').textContent,/new.jpg/);assert.doesNotMatch(h.element('direct-source').textContent,/old.jpg/);assert.equal(h.requests.filter(r=>r.path==='/api/direct-local-models').length,1);
});

test('video drops keep timestamp selection explicit and reject multiple files',async()=>{
  const h=harness();await h.settle();h.element('direct-rights').checked=true;await h.element('direct-drop-zone').listeners.drop({preventDefault(){},dataTransfer:{files:[{name:'video.mp4'}]}});assert.equal(h.uploads.length,0);assert.equal(h.element('direct-time-wrap').hidden,false);assert.equal(h.element('direct-prepare').disabled,false);await h.element('direct-drop-zone').listeners.drop({preventDefault(){},dataTransfer:{files:[{name:'a.jpg'},{name:'b.jpg'}]}});assert.match(h.element('direct-message').textContent,/one image or video/);assert.equal(h.uploads.length,0);
});

test('outdated backend is explicit and preserves other feature successes without futile retry',async()=>{
  const h=harness(path=>path==='/api/direct-local-models'?{error:'Unknown operation.'}:undefined,'review-1');await h.settle();await h.click('direct-local-run');assert.match(h.element('direct-message').textContent,/running server needs the updated backend loaded/);assert.match(h.element('direct-message').textContent,/Retrying cannot fix this/);assert.equal(h.element('direct-local-run').disabled,true);assert.equal(h.element('direct-feature-helmet').dataset.status,'unavailable');assert.equal(h.element('direct-feature-localization').dataset.status,'complete');assert.equal(h.element('direct-feature-vehicles').dataset.status,'complete');assert.match(h.element('direct-report-summary').innerHTML,/unavailable/);assert.match(h.element('direct-local-output').innerHTML,/request error is shown here only/);
});

test('vehicle subtype rendering separates per-ID candidates from unmatched proposals and threshold counts',()=>{
  const evidence={state:'complete',reason:'Local check complete.',subjects:[{id:'V1',status:'low_confidence',candidates:[{label:'Three-wheeler',model_score:.31234567}]},{id:'V2',status:'unclear',candidates:[{label:'Sedan',model_score:.62},{label:'SUV',model_score:.61}]}],unassigned_detections:[{label:'Truck <unsafe>',model_score:.83,reason:'no_numbered_vehicle_match'}],candidate_counts:{total:3,by_type:{Sedan:1,SUV:1,Truck:1}},limitations:['Small objects can be missed.']};const before=JSON.stringify(evidence),html=direct.renderVehicleEvidence(view({vehicle_type_evidence:evidence}));
  assert.match(html,/V1/);assert.match(html,/Low-confidence detection/);assert.match(html,/Needs review/);assert.match(html,/>0\.312</);assert.match(html,/Exact raw model score: 0\.31234567/);assert.match(html,/Unmatched proposals/);assert.match(html,/No matching numbered vehicle box; association unconfirmed/);assert.match(html,/score at least 0\.5/);assert.match(html,/Lower-score candidates above are excluded/);assert.match(html,/not verified objects/);assert.doesNotMatch(html,/<unsafe>/);assert.equal(JSON.stringify(evidence),before);
  assert.doesNotMatch(direct.renderVehicleEvidence(view({vehicle_type_evidence:{state:'unavailable',reason:'Unavailable model.'}})),/proposal counts: 0/);
});

test('automatic vehicle subtype request uses the new helmet fingerprint and is independent of helmet failure',async()=>{
  for(const failed of [false,true]){
    const h=harness(path=>path.startsWith('/api/direct-review')?view({annotations:null,local_evidence:null}):path==='/api/direct-annotate'?view({fingerprint:'numbered',local_evidence:null}):path==='/api/direct-local-models'?(failed?{error:'Helmet runtime request failed.'}:view({fingerprint:'helmet-saved'})):undefined,'review-1');await h.settle();await h.click('direct-annotate');
    const request=h.requests.find(r=>r.path==='/api/direct-vehicle-types');assert.ok(request);assert.deepEqual(JSON.parse(request.options.body),{review_id:'review-1',fingerprint:failed?'numbered':'helmet-saved'});assert.equal(h.element('direct-feature-types').dataset.status,'complete');assert.equal(h.element('direct-feature-helmet').dataset.status,failed?'error':'complete');assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
  }
});

test('vehicle subtype failures preserve successful helmet evidence and offer independent retry',async()=>{
  const h=harness(path=>path==='/api/direct-vehicle-types'?{error:'Subtype service failed.'}:undefined,'review-1');await h.settle();await h.click('direct-vehicle-run');assert.equal(h.element('direct-feature-types').dataset.status,'error');assert.equal(h.element('direct-feature-helmet').dataset.status,'complete');assert.match(h.element('direct-vehicle-output').innerHTML,/Other completed features remain available/);assert.equal(h.element('direct-vehicle-run').disabled,false);assert.equal(h.requests.some(r=>r.path==='/api/direct-local-models'),false);
});

test('unavailable subject localization still runs specialists and keeps manual correction controls',async()=>{
  const empty={subjects:[],image_width:640,image_height:480,detector:{state:'unavailable',reason:'Subject detector unavailable. Add boxes manually.'}};
  const h=harness(path=>path.startsWith('/api/direct-review')?view({annotations:null,local_evidence:null}):path==='/api/direct-annotate'?view({annotations:empty,local_evidence:null}):undefined,'review-1');await h.settle();await h.click('direct-annotate');assert.equal(h.element('direct-feature-localization').dataset.status,'unavailable');assert.equal(h.element('direct-feature-helmet').dataset.status,'unsupported');assert.equal(h.element('direct-feature-types').dataset.status,'complete');assert.equal(h.requests.filter(r=>r.path==='/api/direct-vehicle-types').length,1);assert.equal(h.element('direct-add-box').disabled,false);assert.equal(h.element('direct-send').disabled,true);
});

test('reopening subtype evidence does not rerun any specialist and saved corrections rerun both',async()=>{
  const types={state:'complete',reason:'Saved subtype result.',subjects:[],unassigned_detections:[],candidate_counts:{total:0,by_type:{}},limitations:[]};
  const h=harness(path=>path.startsWith('/api/direct-review')?view({vehicle_type_evidence:types}):undefined,'review-1');await h.settle();assert.match(h.element('direct-vehicle-output').innerHTML,/Saved subtype result/);assert.equal(h.requests.some(r=>r.path==='/api/direct-vehicle-types'||r.path==='/api/direct-local-models'),false);await h.click('direct-add-box');await h.click('direct-save-boxes');assert.equal(h.requests.filter(r=>r.path==='/api/direct-vehicle-types').length,1);assert.equal(h.requests.filter(r=>r.path==='/api/direct-local-models').length,1);
});

test('queued image waits for initial pending library read before uploading or mutating',async()=>{
  let release,count=0;
  const h=harness(path=>path==='/api/direct-library'&&++count===1?new Promise(resolve=>release=resolve):path==='/api/direct-prepare'?view({annotations:null,local_evidence:null}):undefined);await h.settle();h.element('direct-rights').checked=true;
  await h.element('direct-drop-zone').listeners.drop({preventDefault(){},dataTransfer:{files:[{name:'queued.jpg'}]}});assert.equal(h.uploads.length,0);assert.equal(h.requests.some(r=>r.options.method==='POST'),false);
  release({reviews:[]});await h.settle();assert.equal(h.uploads.length,1);h.uploads[0].finish();await h.settle();assert.equal(h.requests.filter(r=>r.path==='/api/direct-annotate').length,1);assert.equal(h.requests.filter(r=>r.path==='/api/direct-vehicle-types').length,1);
});

test('automatic pipeline refreshes library only after all bound mutations finish',async()=>{
  let release,libraryCalls=0;
  const h=harness(path=>path==='/api/direct-library'?(++libraryCalls===1?{reviews:[]}:new Promise(resolve=>release=resolve)):path==='/api/direct-prepare'?view({annotations:null,local_evidence:null}):undefined);await h.settle();h.element('direct-rights').checked=true;
  const pending=h.element('direct-drop-zone').listeners.drop({preventDefault(){},dataTransfer:{files:[{name:'serial.jpg'}]}});h.uploads[0].finish();await h.settle();const relevant=h.requests.filter(r=>r.path.startsWith('/api/direct-')&&r.path!=='/api/direct-review').map(r=>r.path);
  assert.deepEqual(relevant,['/api/direct-library','/api/direct-prepare','/api/direct-annotate','/api/direct-local-models','/api/direct-vehicle-types','/api/direct-plates','/api/direct-library']);assert.equal(libraryCalls,2);release({reviews:[]});await pending;
});

test('manual specialist mutation waits for a pending library refresh and never retries its POST',async()=>{
  let release,count=0;
  const h=harness(path=>path==='/api/direct-library'&&++count===2?new Promise(resolve=>release=resolve):undefined,'review-1');await h.settle();const reading=h.element('direct-library-refresh').listeners.click();await h.settle();const mutating=h.element('direct-local-run').listeners.click();await h.settle();assert.equal(h.requests.some(r=>r.path==='/api/direct-local-models'),false);release({reviews:[]});await reading;await mutating;assert.equal(h.requests.filter(r=>r.path==='/api/direct-local-models').length,1);
});

test('only the latest replacement starts after an active inference fails; no old downstream request runs',async()=>{
  let release,preparations=0;
  const h=harness(path=>path==='/api/direct-prepare'?view({annotations:null,local_evidence:null,source:{name:++preparations===1?'old.jpg':'latest.jpg'}}):path==='/api/direct-annotate'&&preparations===1?new Promise(resolve=>release=resolve):undefined);await h.settle();h.element('direct-rights').checked=true;
  const drop=name=>h.element('direct-drop-zone').listeners.drop({preventDefault(){},dataTransfer:{files:[{name}]}});
  const first=drop('old.jpg');h.uploads[0].finish();await h.settle();const skipped=drop('skip.jpg'),latest=drop('latest.jpg');assert.equal(h.uploads.length,1);release({error:'Previous detector failed.'});await first;await skipped;await h.settle();assert.equal(h.uploads.length,2);assert.equal(h.uploads[1].body.name,'latest.jpg');h.uploads[1].finish();await latest;assert.equal(h.requests.filter(r=>r.path==='/api/direct-local-models').length,1);assert.equal(h.requests.filter(r=>r.path==='/api/direct-vehicle-types').length,1);assert.equal(h.element('direct-filename').textContent,'latest.jpg');assert.doesNotMatch(h.element('direct-message').textContent,/Previous detector failed/);
});


test('restored reviews name their saved source without populating the file input or rerunning inference',async()=>{
  const h=harness(path=>path.startsWith('/api/direct-review')?view({source:{name:'indian_riders.png',kind:'image'}}):undefined,'review-1');await h.settle();
  assert.equal(h.element('direct-filename').textContent,'Saved review source: indian_riders.png. Choose a file to start a new review.');assert.equal(h.element('direct-file').files.length,0);assert.equal(h.uploads.length,0);assert.equal(h.requests.some(r=>r.options.method==='POST'),false);
});


test('final interpretation labels use LLM without exposing model branding or changing stored model ID',()=>{
  const before=JSON.stringify(assessment),html=direct.renderResult(assessment,[frame]);
  assert.match(html,/Optional LLM interpretation/);assert.doesNotMatch(html,/Astra|ChatGPT|gpt-6-astra/i);assert.equal(JSON.stringify(assessment),before);
});
test('plate checks are bound, independent and saved reopening never triggers OCR',async()=>{
  const saved=view({plate_evidence:{state:'complete',reason:'Saved OCR.',source:{width:640,height:480},subjects:[{id:'P1',status:'not_applicable',candidates:[]}]}});
  const h=harness(path=>path.startsWith('/api/direct-review')?saved:undefined,'review-1');await h.settle();
  assert.equal(h.element('direct-feature-plates-status').textContent,'Complete');assert.equal(h.requests.some(r=>r.path==='/api/direct-plates'),false);
  await h.click('direct-plate-run');const request=h.requests.find(r=>r.path==='/api/direct-plates');assert.deepEqual(JSON.parse(request.options.body),{review_id:'review-1',fingerprint:'fingerprint-1'});assert.equal(h.requests.some(r=>r.path==='/api/direct-analyze'),false);
});
test('plate failure retains other evidence and gives an independent retry',async()=>{
  const h=harness(path=>path==='/api/direct-plates'?{error:'OCR unavailable'}:undefined,'review-1');await h.settle();await h.click('direct-plate-run');
  assert.equal(h.element('direct-feature-plates-status').textContent,'Failed');assert.match(h.element('direct-feature-plates-detail').textContent,/Run plate check/);assert.equal(h.element('direct-feature-helmet-status').textContent,'Complete');assert.equal(h.element('direct-plate-run').disabled,false);
});
test('needs-input counts remain waiting when source localization is unavailable',async()=>{
  const h=harness(path=>path.startsWith('/api/direct-review')?view({annotations:{subjects:[],detector:{state:'unavailable'}},feature_report:{features:{vehicle_counts:{state:'needs_input',summary:'Add subject boxes before counting.'}}}}):undefined,'review-1');await h.settle();
  assert.equal(h.element('direct-feature-vehicles-status').textContent,'Waiting');assert.match(h.element('direct-feature-vehicles-detail').textContent,/Add subject boxes/);
});

test('direct sharing consent explicitly names OpenAI and ChatGPT while ordinary review labels use LLM',()=>{
 const html=fs.readFileSync(require.resolve('../web/violation_lens.html'),'utf8');
 assert.match(html,/send these images, their subject identifiers, and local model evidence to OpenAI for LLM review through my ChatGPT plan/);
 assert.match(html,/Ask LLM to explain/);assert.doesNotMatch(html,/Astra/);
 assert.doesNotMatch(direct.renderResult(assessment,[frame]),/ChatGPT|Astra/i);
});

test('unassigned plate proposals remain visible with uncertain OCR, reason, bounds and raw scores',()=>{
 const evidence={state:'complete',source:{width:640,height:480},subjects:[],unassigned_detections:[{plate_id:'plate-1',box:[20,30,80,50],detector_score:0.687,ocr_score:0.027,text:'DED',reason:'ambiguous_vehicle_overlap'},{plate_id:'plate-2',box:[90,100,120,130],detector_score:0.440,ocr_score:0.045,text:'<E>',reason:'no_numbered_vehicle_match'}]};
 const html=direct.renderPlates(view({plate_evidence:evidence}));
 assert.match(html,/Unassigned plate proposals/);assert.match(html,/Ambiguous vehicle overlap/);assert.match(html,/No numbered vehicle match/);assert.match(html,/OCR text is unverified/);assert.match(html,/0.687/);assert.match(html,/0.027/);assert.match(html,/20, 30, 80, 50/);assert.match(html,/Unassigned plate<\/text>/);assert.match(html,/&lt;E&gt;/);assert.doesNotMatch(html,/data-plate-save|Apparent vehicle/);
});

test('actionable review labels preserve saved enums and detailed uncertainty',()=>{
 const saved={subject_results:[{subject_id:'P1',status:'unclear',role:'person_unclear',observations:[{text:'The head is obscured; protection is unclear.',evidence_ids:['image-1']}],uncertainty_text:['Not enough evidence for rider assignment.']}],limitations:[]};
 const before=JSON.stringify(saved),html=direct.renderResult(saved,[frame]);
 assert.doesNotMatch(html,/unclear|not enough evidence/i);assert.match(html,/head is obscured/);assert.match(html,/Needs review|needs review/);assert.equal(JSON.stringify(saved),before);
 assert.equal(common.reviewText('rider_association_unclear'),'rider_association_unclear');
});
test('every feature documents its actual model and inference limits',()=>{
 const html=fs.readFileSync(require.resolve('../web/violation_lens.html'),'utf8');
 assert.equal((html.match(/Model used \/ How it works/g)||[]).length,5);
 for(const model of ['yolo11s.pt','iam-tsr/yolov8n-helmet-detection','iisc-aim/UVH-26','morsetechlab/yolov11-license-plate-detection','EasyOCR 1.7.2'])assert.ok(html.includes(model));
 assert.match(html,/No additional model/);assert.match(html,/training\/test contamination/);
});

test('evidence table rounds displayed scores while retaining exact values and separate observations',()=>{
 const {renderEvidenceTable}=require('../web/evidence_table.js');
 const record={evidence_table:{summary:{suspected_issue_count:0},rows:[{subject_id:'V1',kind:'car',assessment:'observed',observations:['Vehicle visible.'],suspected_issues:[],raw_model_scores:[{source:'vehicle',label:'car',model_score:0.687123456789}],evidence_ids:[],review_status:'pending'}]}};
 const before=JSON.stringify(record),html=renderEvidenceTable(record);
 assert.match(html,/>0\.687<\/code>/);assert.match(html,/title="Exact raw model score: 0\.687123456789"/);assert.match(html,/0 distinct suspected issues/);assert.match(html,/No supported suspicion listed/);assert.match(html,/<summary>Observations<\/summary>/);assert.match(html,/Vehicle visible/);assert.equal(JSON.stringify(record),before);
});

test('ordinary connected summary and usage label use LLM while sign-in keeps provider disclosure',()=>{
 const html=fs.readFileSync(require.resolve('../web/violation_lens.html'),'utf8'),script=fs.readFileSync(require.resolve('../web/violation_lens.js'),'utf8');
 assert.match(html,/Manage LLM usage/);assert.doesNotMatch(html,/Manage ChatGPT usage/);assert.match(html,/https:\/\/chatgpt.com\/settings\/usage/);assert.match(html,/Approve and open ChatGPT sign-in/);
 assert.match(script,/LLM connection ready\. Account eligibility is checked when requesting a model\./);
});
