const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

test('speed outcome feedback preserves accepted/withheld counts, escapes reasons and leaves legacy totals unknown',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const context=vm.createContext({escapeHTML:s=>String(s).replaceAll('<','&lt;').replaceAll('>','&gt;')});
  vm.runInContext(source.slice(source.indexOf('function speedOutcomeSummary('),source.indexOf('function showMotionResult(')),context);
  const speed={observed_track_samples:95,accepted_track_samples:49,withheld_track_samples:46,status_counts:{estimated:49,'collecting stable track':24,'outside calibrated road':21,'<unsafe>':1}};
  const html=context.speedOutcomeSummary(speed);
  assert.match(html,/49 accepted · 46 withheld/);assert.match(html,/95 observed track samples/);
  assert.match(html,/collecting stable track/);assert.match(html,/outside calibrated road/);
  assert.match(html,/&lt;unsafe&gt;/);assert.doesNotMatch(html,/<unsafe>/);
  assert.match(html,/not vehicles or an accuracy score/);
  assert.match(context.speedOutcomeSummary({accepted_track_samples:49}),/totals are unavailable/);
  assert.match(context.speedOutcomeSummary({...speed,observed_track_samples:90}),/totals are unavailable/);
  assert.match(context.speedOutcomeSummary({...speed,status_counts:{estimated:-1}}),/totals are unavailable/);
  assert.match(context.speedOutcomeSummary({observed_track_samples:0,accepted_track_samples:0,withheld_track_samples:0,status_counts:{}}),/No track observations/);
});

test('saved calibrated synthetic speed keeps its virtual label with incident review off',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const context=vm.createContext({escapeHTML:s=>String(s),incidentSummary:()=>''});
  vm.runInContext(source.slice(source.indexOf('function motionSummary('),source.indexOf('function showMotionResult(')),context);
  const motion={timeline:[],line:{axis:'horizontal'},crossings:{positive:0,negative:0},peak_activity_percent:0,
    measured_seconds:0,suppressed_frames:0,limitations:[],speed:{calibrated:true,sample_mean:36,unit:'km/h',accepted_track_samples:18}};
  for(const provenance of [{synthetic:true},{source_provenance:{synthetic:true}}]){
    const html=context.motionSummary({...motion,...provenance});
    assert.match(html,/Synthetic speed · declared virtual scale/);
    assert.match(html,/not a measured real-world speed/);
    assert.doesNotMatch(html,/Uses your measured road patch|Approximate road speed/);
  }
});

test('switching source discards synthetic calibration and incident confirmations before preview',async()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const elements=new Map();
  const get=id=>{if(!elements.has(id))elements.set(id,{value:'',checked:false,hidden:false,classList:{remove(){}},setAttribute(){}});return elements.get(id);};
  const state={revision:0,media:{id:'synthetic'},preview:{},region:[.1,.1,.9,.9],regionTool:true};
  const context=vm.createContext({state,$:get,paintSceneReview(){},drawRegion(){},clearDirection(){},clearZebra(){},clearTarget(){},clearSurvey(){},actions(){},
    drawCalibration(){},paintIncidents(){},renderPreview:async()=>{assert.equal(get('speed-enabled').checked,false);assert.equal(get('incident-calibration-confirmed').checked,false);}});
  vm.runInContext(source.slice(source.indexOf('let calibrationPoints='),source.indexOf('function drawCalibration(')),context);
  vm.runInContext(source.slice(source.indexOf('function clearIncidents('),source.indexOf('function incidentConfig(')),context);
  vm.runInContext(source.slice(source.indexOf('async function openMedia('),source.indexOf('async function sample(')),context);
  vm.runInContext("calibrationPoints=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]; calibrationSource={media_id:'synthetic'};",context);
  for(const id of ['speed-enabled','incident-enabled','incident-calibration-confirmed','incident-real-time','incident-synthetic'])get(id).checked=true;
  get('speed-width').value='16';get('speed-length').value='40';
  get('incident-limit').value='30';get('synthetic-review').checked=true;
  await context.openMedia({id:'real',name:'eufy-low-t1p1.mp4',width:2304,height:1296,kind:'video',duration:65.868,fps:15,size_bytes:123,example:{synthetic:false,calibration_status:'Physical speed withheld'}});
  assert.equal(vm.runInContext('calibrationPoints.length',context),0);
  assert.equal(vm.runInContext('calibrationSource',context),null);
  assert.equal(get('speed-width').value,'');assert.equal(get('speed-length').value,'');
  assert.equal(get('incident-synthetic').checked,false);assert.equal(state.region,null);
  assert.equal(get('incident-limit').value,'');assert.equal(get('synthetic-review').checked,false);
  assert.match(get('example-note').textContent,/withheld/);
});

test('temporary unknown status continues polling until completion', () => {
  const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
  const start = source.indexOf('function showJob(');
  const end = source.indexOf('async function pollJob(', start);
  const elements = new Map();
  const pending = new Map();
  const saved = new Map();
  let serial = 0;
  const context = vm.createContext({
    state: {}, jobTimer: null,
    $: id => { if (!elements.has(id)) elements.set(id, {}); return elements.get(id); },
    showMotionResult: () => {}, links: () => '', pollJob: () => {},
    jobMessage: run => run.status.message || '',
    localStorage: {setItem: (k,v) => saved.set(k,v), getItem:k => saved.get(k), removeItem:k => saved.delete(k)},
    clearTimeout: id => pending.delete(id),
    setTimeout: fn => { const id = ++serial; pending.set(id, fn); return id; },
  });
  vm.runInContext(source.slice(start, end), context);
  for (const state of ['running', 'unknown', 'encoding', 'complete']) {
    context.showJob({id:'example', source:'traffic.mp4', status:{state}, downloads:[]});
    assert.equal(pending.size, state === 'complete' ? 0 : 1, state);
    assert.equal(saved.get('traffic-lab-active-job'), state === 'complete' ? undefined : 'example');
  }
  for (const state of ['cancelled', 'failed', 'interrupted']) {
    context.showJob({id:'example', source:'traffic.mp4', status:{state}, downloads:[]});
    assert.equal(pending.size, 0, state);
  }
});

test('saving a frame leaves the active video polling and identity intact', () => {
  const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
  const start = source.indexOf('function showFrameExport(');
  const end = source.indexOf('async function pollJob(', start);
  const card = {};
  const context = vm.createContext({state:{job:'video-in-progress'}, jobTimer:123,
    $: () => card, escapeHTML: s => s, links: () => 'frame download'});
  vm.runInContext(source.slice(start,end),context);
  context.showFrameExport({source:'traffic.mp4',status:{message:'Frame ready'},downloads:[]});
  assert.equal(context.state.job,'video-in-progress');
  assert.equal(context.jobTimer,123);
  assert.equal(card.hidden,false);
  assert.match(card.innerHTML,/frame download/);
});

test('progress reports measured frames and processing rate, not an early ETA', () => {
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const context=vm.createContext({});
  vm.runInContext(source.slice(source.indexOf('function jobMessage('),source.indexOf('function showJob(')),context);
  assert.equal(context.jobMessage({status:{message:'Starting',frames:1,fps:.1}}),'Starting · 1 frame processed');
  assert.equal(context.jobMessage({status:{message:'Processing',frames:120,fps:8.25},report:{info:{frame_count:600}}}),
    'Processing · 120 / 600 frames processed · 8.3 processing frames/s');
  assert.equal(context.jobMessage({status:{message:'Waiting'}}),'Waiting');
});

test('saved unfinished motion runs open status without treating settings as measurements', async () => {
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  for (const status of ['queued','running','encoding','unknown','cancelled','failed','interrupted','complete']) {
    const elements=new Map(), errors=[], polled=[], rendered=[];
    let modalOpened=false;
    const run={id:'long-video',source:'workload.mp4',status:{state:status,message:'Job status'},downloads:[],
      report:{motion:status==='complete'?{timeline:[],line:{axis:'horizontal'}}:{axis:'horizontal',tracking_mode:'vehicles'}}};
    const context=vm.createContext({
      runOpenRevision:0,
      api:async()=>run, escapeHTML:s=>s, links:()=>'',
      $:id=>{
        if(id==='reuse-settings')return null;
        if(!elements.has(id))elements.set(id,{showModal:()=>{modalOpened=true;},querySelector:()=>null});
        return elements.get(id);
      },
      motionSummary:motion=>{if(!motion.line)throw Error('Request settings are not a completed summary');rendered.push(motion);return 'Completed measurements';},
      evidencePanel:()=>'',bindEvidencePanel:()=>{},
      showMotionResult:()=>{}, showJob:r=>polled.push(r.id), notify:e=>errors.push(e),
    });
    vm.runInContext(source.slice(source.indexOf('async function openRun('),source.indexOf('async function startReview(')),context);
    await context.openRun(run.id);
    assert.deepEqual(errors,[],status);
    assert.equal(modalOpened,true,status);
    assert.equal(rendered.length,status==='complete'?1:0,status);
    assert.equal(polled.length,['queued','running','encoding','unknown'].includes(status)?1:0,status);
  }
});

test('incident setup is opt-in and sends explicit conditions, scale flags and finite geometry',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const values={'incident-start':'15','incident-end':'85','motion-position':'65','motion-axis':'horizontal',
    'incident-direction':'positive','incident-condition':'observe','incident-intervals':'0, 5, red\n5, 12, green',
    'incident-zone-rect':'15, 65, 85, 78','incident-approach':'15, 45, 85, 65','incident-limit':'30'};
  const elements=Object.fromEntries(Object.entries(values).map(([key,value])=>[key,{value}]));
  for(const name of ['enabled','line','zone','restricted','calibration-confirmed','real-time','synthetic'])elements['incident-'+name]={checked:false};
  const context=vm.createContext({$:id=>elements[id]});
  vm.runInContext(source.slice(source.indexOf('function incidentConfig('),source.indexOf('function paintIncidents(')),context);
  assert.equal(context.incidentConfig(),null);
  elements['incident-enabled'].checked=true;elements['incident-line'].checked=true;
  const config=JSON.parse(JSON.stringify(context.incidentConfig()));
  assert.deepEqual(config.line,[[.15,.65],[.85,.65]]);
  assert.equal(config.condition,'observe');assert.equal(config.zone_role,'observe');
  assert.equal(config.calibration_confirmed,false);assert.equal(config.real_time_confirmed,false);
  assert.equal(config.synthetic,false);assert.deepEqual(config.condition_intervals[1],{start:5,end:12,state:'green'});
  elements['incident-end'].value='10';assert.throws(()=>context.incidentConfig(),/line end/);
});

test('incident results distinguish candidates, neutral observations, withheld and synthetic evidence',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const context=vm.createContext({escapeHTML:s=>String(s)});
  vm.runInContext(source.slice(source.indexOf('function incidentSummary('),source.indexOf('function motionSummary(')),context);
  assert.equal(context.incidentSummary(undefined),'');
  const html=context.incidentSummary({enabled:true,configuration:{synthetic:true},counts:{candidate:1,observation:2,withheld:3},
    events:[],event_count:6,thumbnail_policy:'First 40'});
  assert.match(html,/Synthetic incident review/);assert.match(html,/1 review candidates/);
  assert.match(html,/2 neutral observations/);assert.match(html,/3 withheld assessments/);
  assert.match(html,/Zero candidates does not establish/);
});

test('completed-stop evidence displays observed interval and measured positions without inventing missing proof',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const context=vm.createContext({escapeHTML:s=>String(s)});
  vm.runInContext(source.slice(source.indexOf('function stopEvidenceSummary('),source.indexOf('function motionSummary(')),context);
  const proof={start_seconds:3.533333,end_seconds:4.6,duration_seconds:1.066667,
    observed_samples:17,position_diameter_bound_m:.422689,fitted_speed_kmh:.140754};
  const result=context.incidentSummary({enabled:true,configuration:{synthetic:true},counts:{observation:1},event_count:1,
    thumbnail_policy:'First 40',events:[{type:'line_crossing',status:'observation',event_id:'example',track_id:1,
      timestamp_seconds:5.61,synthetic:true,reason:'Observed stop',measurements:{speed:18, speed_unit:'km/h',stop_evidence:proof}}]});
  assert.match(result,/Observed low-motion interval/);assert.match(result,/3\.53–4\.60 s/);
  assert.match(result,/17 observed frames/);assert.match(result,/envelope ≈ 0\.42 m/);
  assert.match(result,/fitted speed 0\.14 km\/h during this interval/);
  assert.match(result,/SYNTHETIC/);assert.match(result,/18\.00 km\/h/);
  assert.equal(context.stopEvidenceSummary(null),'');
  assert.equal(context.stopEvidenceSummary({...proof,observed_samples:7}),'');
  assert.equal(context.stopEvidenceSummary({...proof,position_diameter_bound_m:'<img src=x onerror=alert(1)>'}),'');
  assert.equal(context.stopEvidenceSummary({...proof,duration_seconds:NaN}),'');
});

test('saved evidence review is explicit and preserves withheld, synthetic, unknown and pixel-speed facts',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const context=vm.createContext({escapeHTML:esc,links:()=>'<a>Full JSON</a>'});
  vm.runInContext(source.slice(source.indexOf('function evidencePanel('),source.indexOf('function bindEvidencePanel(')),context);
  const run={id:'completed',status:{state:'complete'},report:{motion:{incidents:{enabled:true}}}};
  const panel=context.evidencePanel(run);
  assert.match(panel,/Build evidence review/);assert.match(panel,/type="checkbox"/);assert.doesNotMatch(panel,/checked/);
  assert.equal(context.evidencePanel({...run,status:{state:'running'}}),'');
  const html=context.evidenceReviewSummary({model_states:{unavailable:1},source_integrity:{state:'changed',message:'Source changed'},downloads:[],
    review:{summary:{event_count:1,statuses:{withheld:1},synthetic_events:1,unknown_provenance_events:0},indexed_events:1,
      warnings:[],limitations:['Review only'],source:{files:[]},events:[{event:{event_id:'test',status:'withheld',type:'line_crossing',timestamp_seconds:2,synthetic:true,track_id:1},
        facts:[{id:'condition',text:'Unknown condition <script>unsafe()</script>'},{id:'speed',text:'Pixel speed is px/s'},
               {id:'source',text:'Recorded source revision fixture'}],
        warnings:[{text:'No calibrated road speed'}],review_questions:[{id:'scene',text:'Verify applicability'}],
        evidence_ref:{file:'motion-incidents.jsonl',line:1},language_model:{state:'unavailable'}}]}});
  assert.match(html,/withheld/);assert.match(html,/SYNTHETIC/);assert.match(html,/px\/s/);
  assert.match(html,/Local model unavailable/);assert.match(html,/Source changed/);
  assert.match(html,/&lt;script&gt;unsafe/);assert.doesNotMatch(html,/<script>/);
  assert.match(html,/Supporting facts and provenance \(1\)/);
  assert.match(html,/Recorded source revision fixture/);
  assert.doesNotMatch(html,/probability|confirmed violation/);
});

test('polling optional model focus preserves evidence details until the artifact changes',async()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const elements={'.evidence-build':{},'.evidence-use-model':{checked:true},'.evidence-status':{}};
  let renders=0,detailsOpen=false;
  elements['.evidence-content']={set innerHTML(value){renders++;detailsOpen=false;}};
  const panel={isConnected:true,dataset:{evidenceRun:'completed'},querySelector:key=>elements[key]};
  const timers=[];
  let current={id:'review',state:'running',phase:'local_model',message:'Offline ready',
    source_integrity:{state:'verified'},review:{generated_at:'first',artifact:'offline'}};
  const context=vm.createContext({
    api:async(path,data)=>data?current:path.includes('?source=')?{state:'none'}:current,
    evidenceReviewSummary:r=>r.review.artifact,
    clearTimeout:()=>{},setTimeout:fn=>{timers.push(fn);return timers.length;}
  });
  vm.runInContext(source.slice(source.indexOf('function bindEvidencePanel('),source.indexOf('function stopEvidenceSummary(')),context);
  context.bindEvidencePanel(panel);
  await Promise.resolve();await Promise.resolve();
  await elements['.evidence-build'].onclick();
  assert.equal(renders,1);detailsOpen=true;
  await timers.shift()();
  assert.equal(renders,1);assert.equal(detailsOpen,true);
  current={...current,state:'complete',model_states:{selected:1},review:{generated_at:'second',artifact:'focused'}};
  await timers.shift()();
  assert.equal(renders,2);assert.equal(elements['.evidence-build'].disabled,false);
});

test('a late saved-review restore cannot replace a newly requested review',async()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const elements={'.evidence-build':{},'.evidence-use-model':{checked:false},'.evidence-status':{},'.evidence-content':{}};
  const panel={isConnected:true,dataset:{evidenceRun:'completed'},querySelector:key=>elements[key]};
  let restore;
  const oldResponse=new Promise(resolve=>{restore=resolve;});
  const current={id:'new',state:'running',message:'New review running',review:{generated_at:'new',artifact:'new'}};
  const context=vm.createContext({api:async(path,data)=>data?current:oldResponse,
    evidenceReviewSummary:r=>r.review.artifact,clearTimeout:()=>{},setTimeout:()=>1});
  vm.runInContext(source.slice(source.indexOf('function bindEvidencePanel('),source.indexOf('function stopEvidenceSummary(')),context);
  context.bindEvidencePanel(panel);
  await elements['.evidence-build'].onclick();
  restore({id:'old',state:'complete',message:'Old saved review',review:{generated_at:'old',artifact:'old'}});
  await Promise.resolve();await Promise.resolve();await Promise.resolve();
  assert.equal(elements['.evidence-content'].innerHTML,'new');
  assert.equal(elements['.evidence-status'].textContent,'New review running');
  assert.equal(elements['.evidence-build'].disabled,true);
});

test('a changed review artifact clears previously displayed facts and offers a rebuild',async()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const elements={'.evidence-build':{},'.evidence-use-model':{checked:false},'.evidence-status':{},'.evidence-content':{},'.evidence-integrity':{}};
  const panel={isConnected:true,dataset:{evidenceRun:'completed'},querySelector:key=>elements[key]};
  const ready={id:'saved',state:'complete',review:{artifact:'offline',generated_at:'first'},artifact_integrity:{state:'verified',message:'Files intact'}};
  const changed={id:'saved',state:'complete',artifact_integrity:{state:'changed',message:'Files changed; rebuild'}};
  const context=vm.createContext({api:async(path,data)=>data?changed:ready,
    evidenceReviewSummary:()=>'<p>Original facts</p>',clearTimeout:()=>{},setTimeout:()=>1});
  vm.runInContext(source.slice(source.indexOf('function bindEvidencePanel('),source.indexOf('function stopEvidenceSummary(')),context);
  context.bindEvidencePanel(panel);await Promise.resolve();await Promise.resolve();
  assert.match(elements['.evidence-content'].innerHTML,/Original facts/);
  await elements['.evidence-build'].onclick();
  assert.equal(elements['.evidence-content'].innerHTML,'');
  assert.match(elements['.evidence-integrity'].textContent,/Files changed/);
  assert.equal(elements['.evidence-build'].disabled,false);
});

test('saved-run library appends older metadata pages using the server cursor',async()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const elements={'run-search':{value:''},'run-filter':{value:''},'library-more':{},'library-summary':{},'runs':{querySelectorAll:()=>[]}};
  const state={runs:[]},calls=[];
  const pages=[{runs:[{id:'new',source:'new.mp4',status:{state:'complete'}}],has_more:true,next_before:'new'},
    {runs:[{id:'old',source:'india-city.mp4',status:{state:'complete'}}],has_more:false,next_before:null}];
  const context=vm.createContext({state,$:id=>elements[id],escapeHTML:String,api:async path=>{calls.push(path);return pages.shift();},notify:e=>{throw Error(e);},setTimeout:()=>1,clearTimeout:()=>{}});
  vm.runInContext(source.slice(source.indexOf('let libraryRevision='),source.indexOf('async function openRun(')),context);
  await context.loadRuns();assert.equal(elements['library-more'].hidden,false);
  await context.loadRuns(true);
  assert.deepEqual(Array.from(state.runs,r=>r.id),['new','old']);
  assert.match(calls[1],/before=new/);assert.equal(elements['library-more'].hidden,true);
  assert.match(elements['runs'].innerHTML,/india-city/);assert.match(elements['library-summary'].textContent,/2 saved runs/);
});

test('a slow old library search cannot replace a newer query or its paging state',async()=>{
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const elements={'run-search':{value:'old'},'run-filter':{value:''},'library-more':{},'library-summary':{},'runs':{querySelectorAll:()=>[]}};
  let release;const slow=new Promise(resolve=>{release=resolve;});const state={runs:[]};
  const context=vm.createContext({state,$:id=>elements[id],escapeHTML:String,
    api:async path=>path.includes('q=old')?slow:{runs:[{id:'correct',source:'city.mp4',status:{state:'complete'}}],has_more:false,next_before:null},
    notify:e=>{throw Error(e);},setTimeout:()=>1,clearTimeout:()=>{}});
  vm.runInContext(source.slice(source.indexOf('let libraryRevision='),source.indexOf('async function openRun(')),context);
  const pending=context.loadRuns();elements['run-search'].value='city';await context.loadRuns();
  release({runs:[{id:'stale',source:'old.mp4',status:{state:'complete'}}],has_more:true,next_before:'stale'});await pending;
  assert.deepEqual(Array.from(state.runs,r=>r.id),['correct']);assert.equal(elements['library-more'].hidden,true);
  assert.equal(elements['library-more'].disabled,false);assert.doesNotMatch(elements['runs'].innerHTML,/stale/);
});

function sceneHarness(provider){
 const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8'),elements=new Map(),timers=[],storage=new Map();
 const state={media:{id:'source',source_revision:'revision'},review:null};
 const context=vm.createContext({state,$:id=>{if(!elements.has(id))elements.set(id,{value:'0',textContent:'',append:()=>{}});return elements.get(id);},
  api:provider,payload:()=>({media:state.media.id,timestamp:0}),actions:()=>{},notify:()=>{},downloadJSON:()=>{},
  document:{createElement:()=>({})},localStorage:{setItem:(k,v)=>storage.set(k,v),getItem:k=>storage.get(k)},
  setTimeout:(fn)=>{timers.push(fn);return timers.length;},clearTimeout:()=>{}});
 vm.runInContext(source.slice(source.indexOf('let sceneStarting='),source.indexOf('async function init(')),context);
 return {context,state,elements,timers,storage};
}
const sceneRecord={timestamp:0,total_seconds:40,observation:{summary:'Unverified categories: car.',uncertainties:['No spatial or rule decisions.']}};

test('a failed overview retry retains the previous successful source result',async()=>{
 for(const failure of ['missing-cache','invalid-saved-record']){
  let calls=0;
  const h=sceneHarness(async(path,data)=>{
   if(data)return {id:'attempt'+(++calls)};
   if(calls===1)return {state:'complete',review:sceneRecord};
   if(failure==='invalid-saved-record'){const error=Error('Saved overview contents are unreadable.');error.status=400;throw error;}
   return {state:'unavailable',message:'Cached model missing; no download attempted.'};
  });
  await h.context.startReview();await h.context.pollReview();
  assert.equal(h.elements.get('review-output').textContent,sceneRecord.observation.summary);
  await h.context.startReview();
  assert.equal(h.elements.get('review-output').textContent,sceneRecord.observation.summary);
  await h.context.pollReview();
  assert.equal(h.elements.get('review-output').textContent,sceneRecord.observation.summary);
  assert.match(h.elements.get('review-status').textContent,failure==='missing-cache'?/Cached model missing/:/unreadable/);
  assert.equal(h.state.review,null);
 }
});

test('overview cancellation targets its request and discards a late cancelled response',async()=>{
 const calls=[];
 const h=sceneHarness(async(path,data)=>{calls.push([path,data]);return data?.id?{state:'cancelling'}:data?{id:'owned'}:{state:'cancelled',message:'Overview cancelled; previous results retained.'};});
 await h.context.startReview();await h.context.cancelReview();
 assert.equal(calls[1][0],'/api/review/cancel');assert.equal(calls[1][1].id,'owned');
 assert.equal(h.state.review,null);assert.match(h.elements.get('review-status').textContent,/cancelled/);
 assert.equal(h.elements.get('review-output'),undefined);
});

test('a completed overview never overwrites the image selected while it was running',async()=>{
 const h=sceneHarness(async(path,data)=>data?{id:'original'}:{state:'complete',review:sceneRecord});
 await h.context.startReview();h.state.media={id:'next',source_revision:'next-revision'};
 await h.context.pollReview();
 assert.equal(h.elements.get('review-output').textContent,'');
 assert.match(h.elements.get('review-status').textContent,/previous source/);
 h.state.media={id:'source',source_revision:'revision'};h.context.paintSceneReview();
 assert.equal(h.elements.get('review-output').textContent,sceneRecord.observation.summary);
});

test('reload restores a cancellable request and a completed source-bound overview',async()=>{
 const h=sceneHarness(async()=>({id:'saved',state:'complete',media_id:'source',source_revision:'revision',review:sceneRecord}));
 h.storage.set('traffic-lab-scene-results',JSON.stringify([['source:revision','saved']]));
 h.storage.set('traffic-lab-scene-active',JSON.stringify({id:'active',source:'source:revision',stamp:0}));
 await h.context.restoreSceneReviews();
 assert.equal(h.state.review.id,'active');assert.equal(h.elements.get('review-output').textContent,sceneRecord.observation.summary);
 assert.match(h.elements.get('review-status').textContent,/cancel/);assert.equal(h.timers.length,1);
});

test('a missing saved attempt ends polling while a connection failure remains retryable',async()=>{
 let missing=false;
 const h=sceneHarness(async(path,data)=>{if(data)return {id:'owned'};const error=Error('Unavailable');if(missing)error.status=400;throw error;});
 await h.context.startReview();await h.context.pollReview();
 assert.equal(h.state.review.id,'owned');assert.match(h.elements.get('review-status').textContent,/retrying/);
 missing=true;await h.context.pollReview();assert.equal(h.state.review,null);
 assert.match(h.elements.get('review-status').textContent,/Previous successful results/);
});

test('source controls wait for restoration and retain missing-demo availability',()=>{
 const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8'),elements=new Map();
 const state={config:{samples:[{name:'street.jpg'}]}};
 const context=vm.createContext({state,workspaceRestoring:true,$:id=>{if(!elements.has(id))elements.set(id,{});return elements.get(id);}});
 vm.runInContext(source.slice(source.indexOf('function sourceActions('),source.indexOf('const groups =')),context);
 context.sourceActions();assert.equal(elements.get('open-footage').disabled,true);assert.equal(elements.get('upload').disabled,true);
 context.workspaceRestoring=false;context.sourceActions();assert.equal(elements.get('open-footage').disabled,false);
 assert.equal(elements.get('sample').disabled,false);assert.equal(elements.get('motion-demo').disabled,true);
 state.config.samples.push({name:'india-city.mp4'});context.sourceActions();assert.equal(elements.get('motion-demo').disabled,false);
});

test('uploads wait behind an in-flight API request, including after its failure',async()=>{
 const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
 for(const rejectPrevious of [false,true]){
  let finish,xhr,opened=null;
  const previous=new Promise((resolve,reject)=>{finish=()=>rejectPrevious?reject(Error('Earlier request failed')):resolve({});});
  const context=vm.createContext({workspaceRestoring:false,token:'test',encodeURIComponent,
   performApi:()=>previous,XMLHttpRequest:class{constructor(){this.upload={};xhr=this;}open(){}setRequestHeader(){}send(){this.sent=true;}},
   $:()=>({}),busy:()=>{},notify:()=>{},openMedia:async value=>{opened=value;}});
  vm.runInContext(source.slice(source.indexOf('let apiQueue='),source.indexOf('function payload(')),context);
  vm.runInContext(source.slice(source.indexOf('async function upload('),source.indexOf('const concepts=')),context);
  const pending=context.api('/slow').catch(()=>{}),upload=context.upload({name:'next.png',size:100});
  await Promise.resolve();assert.equal(xhr,undefined);finish();await pending;await Promise.resolve();await Promise.resolve();
  assert.equal(xhr.sent,true);xhr.status=200;xhr.responseText='{"id":"new-source"}';xhr.onload();await upload;
  assert.equal(opened.id,'new-source');
 }
});

test('dropping a file during restoration leaves the saved source untouched',async()=>{
 const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');let warning='';
 const context=vm.createContext({workspaceRestoring:true,notify:value=>{warning=value;},queueApi:()=>{throw Error('Must not upload while restoring');}});
 vm.runInContext(source.slice(source.indexOf('async function upload('),source.indexOf('const concepts=')),context);
 await context.upload({name:'next.png',size:100});assert.match(warning,/restoration/);
});

test('an interrupted saved-evidence review keeps its artifact and enables a fresh request',async()=>{
 const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
 const elements={'.evidence-build':{},'.evidence-use-model':{checked:false},'.evidence-status':{},'.evidence-content':{},'.evidence-integrity':{}};
 const panel={isConnected:true,dataset:{evidenceRun:'real-city'},querySelector:key=>elements[key]},timers=[];
 const response={id:'interrupted',state:'interrupted',message:'Review stopped before final status; offline retained.',
  source_integrity:{state:'verified'},artifact_integrity:{state:'verified'},review:{artifact:'offline',generated_at:'saved'}};
 const context=vm.createContext({api:async()=>response,evidenceReviewSummary:()=>'<p>Completed offline facts</p>',
  clearTimeout:()=>{},setTimeout:fn=>{timers.push(fn);return timers.length;}});
 vm.runInContext(source.slice(source.indexOf('function bindEvidencePanel('),source.indexOf('function stopEvidenceSummary(')),context);
 context.bindEvidencePanel(panel);await Promise.resolve();await Promise.resolve();
 assert.equal(elements['.evidence-content'].innerHTML,'<p>Completed offline facts</p>');
 assert.equal(elements['.evidence-build'].disabled,false);assert.equal(elements['.evidence-use-model'].disabled,false);
 assert.match(elements['.evidence-status'].textContent,/offline retained/);assert.equal(timers.length,0);
});

test('fact references preserve missing, null and false values and escape source text',()=>{
 const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
 const context=vm.createContext({escapeHTML:value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')});
 vm.runInContext(source.slice(source.indexOf('function evidencePanel('),source.indexOf('function bindEvidencePanel(')),context);
 const html=context.evidenceCitations({pointer_base:'event',citations:[
  {pointer:'/calibration',recorded:true,value:null},{pointer:'/configuration/calibration_confirmed',recorded:true,value:false},
  {pointer:'/configuration/real_time_confirmed',recorded:false},{pointer:'/reason<script>',recorded:true,value:'<script>bad()</script>'}]});
 assert.match(html,/<code>event:\/calibration<\/code> · null/);
 assert.match(html,/<code>event:\/configuration\/calibration_confirmed<\/code> · false/);
 assert.match(html,/Unknown — not recorded/);assert.match(html,/&lt;script&gt;/);assert.doesNotMatch(html,/<script>/);
 assert.equal(context.evidenceCitations({pointer:'/legacy',value:false}),'');
});

test('grounding context describes a recorded source check without claiming measurement truth',()=>{
 const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
 const context=vm.createContext({escapeHTML:String,links:()=>''});
 vm.runInContext(source.slice(source.indexOf('function evidencePanel('),source.indexOf('function bindEvidencePanel(')),context);
 const review={summary:{event_count:164,statuses:{observation:17,withheld:147},synthetic_events:0,unknown_provenance_events:0},
  warnings:[],events:[],indexed_events:0,limitations:[],source:{files:[]}};
 assert.doesNotMatch(context.evidenceReviewSummary({review}),/Recorded source-reference check/);
 review.grounding={state:'source_citations_checked',event_count:164,fact_count:1752,citation_count:2838,unrecorded_field_citations:54,
  limitation:'Checks source correspondence, not measurement accuracy or a legal finding.'};
 const html=context.evidenceReviewSummary({review});assert.match(html,/At publication: 164 source assessments, 1752 facts and 2838 references checked/);
 assert.match(html,/54 references point to unrecorded fields/);assert.match(html,/not measurement accuracy or a legal finding/);
});
