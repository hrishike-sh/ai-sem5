const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const direct=require('../web/direct_review.js');
const common=require('../web/violation_lens.js');
const frame={id:'image-1',timestamp:0,url:'/prepared/image.jpg',available:true,sha256:'abc',width:640,height:480};
const assessment={state:'reviewed',model:'gpt-6-astra',observations:[{kind:'rider_visible',text:'A rider is visible.',evidence_ids:['image-1'],uncertainties:['visibility'],uncertainty_text:['The head is partially obscured.']}],suspected_issues:[],limitations:['A single image cannot establish speed.']};
const view=(overrides={})=>({review_id:'review-1',fingerprint:'fingerprint-1',source:{name:'permitted.png',kind:'image'},frames:[{...frame}],model_result:null,review:{decision:null,note:''},source_readiness:{available:true},...overrides});

function harness(respond=()=>undefined,saved=null){
  const elements=new Map(),requests=[],uploads=[],downloads=[],savedIDs=[];
  function node(id){
    const item={id,hidden:false,disabled:false,checked:false,value:id==='direct-time'?'0':'',files:[],textContent:'',listeners:{},dataset:{},images:[],label:{hidden:false},classList:{toggle(){}},addEventListener(name,fn){this.listeners[name]=fn;},closest(){return this.label;},querySelectorAll(selector){return selector.startsWith('img')?this.images:[];},click(){downloads.push(this.href);},remove(){}};
    Object.defineProperty(item,'innerHTML',{get(){return this.html||'';},set(html){this.html=html;this.images=Array.from(html.matchAll(/data-direct-frame="([^"]+)"/g),match=>({...node('image'),dataset:{directFrame:match[1]},naturalWidth:0,complete:false}));}});
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
