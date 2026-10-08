const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const direct=require('../web/direct_review.js');
const common=require('../web/violation_lens.js');
const frame={id:'image-1',timestamp:0,url:'/prepared/image.jpg',available:true,sha256:'abc',width:640,height:480};
const assessment={state:'reviewed',model:'gpt-6-astra',observations:[{kind:'rider_visible',text:'A rider is visible.',evidence_ids:['image-1'],uncertainties:['visibility'],uncertainty_text:['The head is partially obscured.']}],suspected_issues:[],limitations:['A single image cannot establish speed.']};
const view=(overrides={})=>({review_id:'review-1',fingerprint:'fingerprint-1',source:{name:'permitted.png',kind:'image'},frames:[{...frame}],model_result:null,review:{decision:null,note:''},source_readiness:{available:true},annotations:{image_width:640,image_height:480,subjects:[{id:'P1',kind:'person',bbox:[10,20,100,200],origin:'detector',evidence_ids:['image-1']}]},...overrides});

function harness(respond=()=>undefined,saved=null){
  const elements=new Map(),requests=[],uploads=[],downloads=[],savedIDs=[];
  function node(id){
    const item={id,hidden:false,disabled:false,checked:false,value:id==='direct-time'?'0':'',files:[],textContent:'',listeners:{},dataset:{},images:[],label:{hidden:false},classList:{toggle(){}},addEventListener(name,fn){this.listeners[name]=fn;},closest(){return this.label;},querySelectorAll(selector){return selector.startsWith('img')?this.images:[];},click(){downloads.push(this.href);},remove(){}};
    Object.defineProperty(item,'innerHTML',{configurable:true,get(){return this.html||'';},set(html){this.html=html;this.images=Array.from(html.matchAll(/data-direct-frame="([^"]+)"/g),match=>({...node('image'),dataset:{directFrame:match[1]},naturalWidth:0,complete:false}));}});
    return item;
  }
  function element(id){if(!elements.has(id))elements.set(id,node(id));return elements.get(id);}
  class Upload{constructor(){this.upload={};this.headers={};uploads.push(this);}open(method,url){this.method=method;this.url=url;}setRequestHeader(name,value){this.headers[name]=value;}send(body){this.body=body;}abort(){this.onabort();}finish(body={id:'media-1',source_revision:'revision-1',kind:'image'},status=200){this.status=status;this.responseText=JSON.stringify(body);this.onload();}}
  const window={ViolationLens:common,XMLHttpRequest:Upload,document:{getElementById:element,querySelector(){return {content:'csrf-direct'};},createElement(){return node('link');},body:{append(){}}},localStorage:{getItem(){return saved;},setItem(key,id){savedIDs.push({key,id});}},fetch:async(path,options={})=>{requests.push({path,options});let data=await respond(path,options);if(data===undefined)data=path==='/api/config'?{evidence_origin:'http://127.0.0.1:5000',violation_readiness:{ready:false}}:path==='/api/violation-llm'?{available:true}:path==='/api/direct-library'?{reviews:[]}:view();return {ok:!data.error,json:async()=>data};}};
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
  const labels={person_on_two_wheeler:'Rider visible',bare_head_of_rider_visible:'Uncovered head visible',helmet_on_rider_visible:'Helmet visible on rider',crowded_rider_load_visible:'Crowded rider arrangement',rider_position_appears_unstable:'Possibly unstable rider position',cargo_appears_unsecured:'Possibly unsecured cargo',rider_association_unclear:'Rider association unclear',head_protection_unclear:'Head protection unclear',no_relevant_road_user_visible:'No relevant road user clearly visible',helmet_possible:'Possible missing helmet',unsafe_rider_load_possible:'Possible unsafe rider arrangement',other_visible_concern:'Possible cargo safety concern'};
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
  h.element('direct-rights').checked=true;h.element('direct-rights').listeners.change();assert.equal(h.element('direct-prepare').disabled,false);const pending=h.element('direct-prepare').listeners.click();assert.equal(h.uploads.length,1);assert.equal(h.uploads[0].body,file);assert.equal(h.uploads[0].headers['X-Traffic-Token'],'csrf-direct');assert.equal(h.uploads[0].url,'/api/upload?name=street%20photo.png');h.uploads[0].finish();await pending;await h.settle();
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
  const html=direct.renderResult(result,[frame]);assert.match(html,/P1/);assert.match(html,/V1/);assert.match(html,/Unclear — evidence is insufficient/);assert.match(html,/No visible violation for this subject/);assert.match(html,/&lt;script&gt;/);assert.doesNotMatch(html,/<script>|<b>One frame/);
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
  assert.match(html,/Person P1/);assert.match(html,/Suspected issue — human review required/);assert.match(html,/Associated vehicle: V1/);assert.match(html,/Related subjects: P1, V1/);assert.match(html,/Head partly obscured/);
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
