const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

function harness(){
 const elements=new Map(),calls={opened:[],draws:0,scheduled:0,modeChanges:0};
 const get=id=>{if(!elements.has(id))elements.set(id,{value:'',checked:false,disabled:false,textContent:''});return elements.get(id);};
 const setup={kind:'fixed_translation_fixture_v1',points:[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],width_m:16,length_m:40,reference_frame:0,threshold_is_legal_limit:false};
 const state={media:{id:'known',kind:'video',name:'speed_above_36.mp4',source_revision:'revision-1',example:{synthetic:true,virtual_setup:setup}},config:{vehicle_tracking_ready:true},pending:false,rendering:false};
 const context=vm.createContext({state,$:get,workspaceRestoring:false,clearTarget(){},drawCalibration(){calls.draws++;},drawRegion(){},paintMotionLine(){},paintIncidents(){},motionActions(){},motionModeChanged(){calls.modeChanges++;},schedule(){calls.scheduled++;},sample:async name=>calls.opened.push(name),api(){throw Error('No API/job inference during setup');},startMotion(){throw Error('No video job during setup');}});
 vm.runInContext(source.slice(source.indexOf('let calibrationPoints='),source.indexOf('function drawCalibration(')),context);
 vm.runInContext(source.slice(source.indexOf('function speedCalibration('),source.indexOf('function clearIncidents(')),context);
 vm.runInContext(source.slice(source.indexOf('function clearIncidents('),source.indexOf('function paintIncidents(')),context);
 get('synthetic-threshold').value='30';get('synthetic-choice').value='synthetic-speed-v1/videos/speed_above_36.mp4';get('timestamp').value='0';get('motion-mode').value='ivp';
 for(const [id,value] of Object.entries({'motion-position':'65','motion-axis':'horizontal','incident-start':'15','incident-end':'85','incident-direction':'positive','incident-condition':'observe','incident-zone-rect':'15,65,85,78','incident-approach':'15,45,85,65'}))get(id).value=value;
 return {context,state,get,calls,setup};
}

test('opening a demo only selects its existing sample and never applies calibration or starts a job',async()=>{
 const h=harness();h.context.syntheticActions();assert.equal(h.get('speed-enabled').checked,false);assert.deepEqual(h.calls.opened,[]);
 await h.context.openSyntheticExample();assert.deepEqual(h.calls.opened,['synthetic-speed-v1/videos/speed_above_36.mp4']);assert.equal(h.get('sample').value,'');assert.equal(h.get('speed-enabled').checked,false);assert.equal(h.calls.draws,0);
});

test('explicit virtual setup binds current source, copies points, preserves IVP mode and leaves review off',()=>{
 const h=harness();h.context.applySyntheticSetup();
 const calibration=JSON.parse(JSON.stringify(h.context.speedCalibration()));
 assert.equal(calibration.media_id,'known');assert.equal(calibration.source_revision,'revision-1');assert.equal(calibration.reference_frame,0);
 assert.deepEqual(calibration.points,h.setup.points);assert.equal(calibration.width_m,16);assert.equal(calibration.length_m,40);
 assert.equal(h.get('motion-mode').value,'ivp');assert.equal(h.context.incidentConfig(),null);assert.equal(h.get('incident-limit').value,'30');
 assert.match(h.get('synthetic-status').textContent,/not a legal speed limit/);assert.match(h.get('synthetic-status').textContent,/No video job or inference/);
 vm.runInContext('calibrationPoints[0][0]=.2',h.context);assert.equal(h.setup.points[0][0],.1);assert.equal(h.calls.scheduled,0);
});

test('explicit optional review uses the configurable TEST threshold without a signal or legal restriction',()=>{
 const h=harness();h.get('synthetic-threshold').value='45';h.get('synthetic-review').checked=true;h.context.applySyntheticSetup();
 const review=JSON.parse(JSON.stringify(h.context.incidentConfig()));
 assert.equal(review.enabled,true);assert.equal(review.synthetic,true);assert.equal(review.speed_limit_kmh,45);
 assert.equal(review.condition,'observe');assert.equal(review.line,null);assert.equal(review.zone,null);
 assert.equal(h.get('synthetic-review').checked,true);assert.equal(h.get('motion-mode').value,'vehicles');
 assert.equal(h.calls.modeChanges,1);
 h.get('synthetic-threshold').value='55';h.context.applySyntheticSetup();assert.equal(h.get('synthetic-review').checked,true);assert.equal(h.context.incidentConfig().enabled,true);assert.equal(h.context.incidentConfig().speed_limit_kmh,55);
});

test('invalid thresholds and unavailable vehicle review fail before mutating existing setup',()=>{
 for(const value of ['', '0','161','NaN','Infinity']){
  const h=harness();h.get('synthetic-threshold').value=value;h.get('speed-width').value='keep';assert.throws(()=>h.context.applySyntheticSetup(),/TEST threshold/);assert.equal(h.get('speed-width').value,'keep');assert.equal(h.get('speed-enabled').checked,false);
 }
 const h=harness();h.state.config.vehicle_tracking_ready=false;h.get('synthetic-review').checked=true;assert.throws(()=>h.context.applySyntheticSetup(),/Vehicle mode is unavailable/);assert.equal(h.get('speed-enabled').checked,false);
});

test('real, unknown and missing-identity sources cannot receive virtual geometry',()=>{
 for(const change of [h=>{h.state.media.example.synthetic=false;},h=>{delete h.state.media.example.virtual_setup;},h=>{delete h.state.media.source_revision;}]){
  const h=harness();change(h);h.context.syntheticActions();assert.throws(()=>h.context.applySyntheticSetup(),/known synthetic examples/);assert.equal(h.get('speed-enabled').checked,false);assert.equal(h.calls.draws,0);
 }
});

test('all four Open / Apply / Run paths submit source-bound IVP calibration without implicit threshold review',async()=>{
 const html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
 const choices=[...html.match(/<select id="synthetic-choice">([\s\S]*?)<\/select>/)[1].matchAll(/value="([^"]+)"/g)].map(m=>m[1]);
 assert.equal(choices.length,4);
 for(const [index,name] of choices.entries()){
  const h=harness(),requests=[];
  h.get('synthetic-choice').value=name;
  h.get('motion-view').value='tracks';h.get('active-job').scrollIntoView=()=>{};
  h.context.sample=async selected=>{h.calls.opened.push(selected);h.state.media={...h.state.media,id:'source-'+index,name:path.basename(selected),source_revision:'revision-'+index};};
  h.context.startingMotion=false;h.context.targetRectangle=null;
  h.context.api=async(route,data)=>{requests.push({route,data});return {id:'mock-run-'+index};};
  h.context.showJob=()=>{};h.context.notify=()=>{};
  vm.runInContext(source.slice(source.indexOf('async function startMotion('),source.indexOf('let selectingTarget=')),h.context);
  await h.context.openSyntheticExample();assert.equal(requests.length,0);assert.equal(h.get('speed-enabled').checked,false);
  h.context.applySyntheticSetup();assert.equal(requests.length,0);
  await h.context.startMotion();assert.equal(requests.length,1);
  const request=JSON.parse(JSON.stringify(requests[0]));
  assert.equal(request.route,'/api/motion');assert.equal(request.data.media,'source-'+index);
  assert.equal(request.data.tracking_mode,'ivp');assert.equal(request.data.incidents,null);
  assert.match(h.get('synthetic-status').textContent,/Run submitted for/);
  assert.match(h.get('synthetic-status').textContent,/declared virtual geometry, not measured real-world speed/);
  assert.doesNotMatch(h.get('synthetic-status').textContent,/No video job or inference has started/);
  assert.equal(request.data.calibration.media_id,request.data.media);
  assert.equal(request.data.calibration.source_revision,'revision-'+index);
  assert.deepEqual(request.data.calibration.points,h.setup.points);
  assert.equal(request.data.calibration.reference_frame,0);assert.equal(request.data.calibration.width_m,16);assert.equal(request.data.calibration.length_m,40);
 }
});

test('loading a demo updates its source picker before rendering and a late preview cannot overwrite it',async()=>{
 const elements=new Map(),state={media:{id:'old'}},get=id=>{if(!elements.has(id))elements.set(id,{value:'old.mp4'});return elements.get(id);};
 let finishFirst;
 const firstWait=new Promise(resolve=>finishFirst=resolve);
 const context=vm.createContext({state,$:get,busy(){},notify(){},api:async(route,data)=>({id:data.name}),
  openMedia:async media=>{assert.equal(get('sample').value,media.id);state.media=media;if(media.id==='first.mp4')await firstWait;}});
 vm.runInContext(source.slice(source.indexOf('async function sample('),source.indexOf('async function upload(')),context);
 const first=context.sample('first.mp4');await Promise.resolve();
 await context.sample('second.mp4');assert.equal(get('sample').value,'second.mp4');
 finishFirst();await first;assert.equal(state.media.id,'second.mp4');assert.equal(get('sample').value,'second.mp4');
});

test('submitting a synthetic clip without applying setup keeps the explicit px/s warning',async()=>{
 const h=harness();let request;
 h.context.startingMotion=false;h.context.targetRectangle=null;h.get('active-job').scrollIntoView=()=>{};
 h.context.api=async(route,data)=>{request=data;return {id:'uncalibrated'};};h.context.showJob=()=>{};h.context.notify=()=>{};
 vm.runInContext(source.slice(source.indexOf('async function startMotion('),source.indexOf('let selectingTarget=')),h.context);
 await h.context.startMotion();assert.equal(request.calibration,null);
 assert.match(h.get('synthetic-status').textContent,/No virtual calibration applied: speed stays in px\/s/);
 assert.doesNotMatch(h.get('synthetic-status').textContent,/Uses declared virtual geometry/);
});

test('a submitted run cannot replace setup feedback for a different source or revision opened while awaiting the response',async()=>{
 for(const change of [media=>({...media,id:'next'}),media=>({...media,source_revision:'revised'})]){
  const h=harness();h.context.applySyntheticSetup();
  h.context.startingMotion=false;h.context.targetRectangle=null;h.get('active-job').scrollIntoView=()=>{};
  h.context.api=async()=>{h.state.media=change(h.state.media);h.get('synthetic-status').textContent='Feedback for the newly opened source';return {id:'previous-source-run'};};
  h.context.showJob=()=>{};h.context.notify=()=>{};
  vm.runInContext(source.slice(source.indexOf('async function startMotion('),source.indexOf('let selectingTarget=')),h.context);
  await h.context.startMotion();assert.equal(h.get('synthetic-status').textContent,'Feedback for the newly opened source');
 }
});
