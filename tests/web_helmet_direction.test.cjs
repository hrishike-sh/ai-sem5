const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
function harness(){
 const elements=new Map();const $=id=>{if(!elements.has(id))elements.set(id,{style:{},classList:{remove(){}},setAttribute(){},scrollIntoView(){},addEventListener(){}});return elements.get(id);};
 const state={config:{helmet:{ready:true,message:'local'}},media:{id:'m',source_revision:'r',kind:'video'},preview:{timestamp:0},pending:false,rendering:false};
 const requests=[],notices=[];const c=vm.createContext({state,$,workspaceRestoring:false,copy:v=>JSON.parse(JSON.stringify(v)),escapeHTML:String,
   selectStage(){},schedule(){},showJob(){},notify:(...x)=>notices.push(x),calibrating:false,selectingTarget:false,zebraDrawing:false,
   api:async(route,data)=>{requests.push({route,data});return {id:'fixture'};}});
 vm.runInContext(source.slice(source.indexOf('let helmetStarting=')),c);c.setupDirection();
 return {c,state,$,requests,notices};
}
test('helmet request pins current source and decoded preview timestamp without a review/LLM',async()=>{
 const h=harness();h.state.preview.timestamp=2.4;await h.c.startHelmet();
 assert.equal(h.requests.length,1);assert.equal(h.requests[0].route,'/api/helmet');
 assert.deepEqual(JSON.parse(JSON.stringify(h.requests[0].data)),{media:'m',source_revision:'r',timestamp:2.4});
});
test('unavailable helmet models do not dispatch',async()=>{
 const h=harness();h.state.config.helmet={ready:false,message:'missing'};await h.c.startHelmet();
 assert.equal(h.requests.length,0);assert.match(h.notices[0][0],/missing/);
});
test('direction needs six marked points plus an explicit frame-zero review',async()=>{
 const h=harness();h.$('direction-mark').onclick();
 for(const p of [[.1,.1],[.9,.1],[.9,.9],[.1,.9],[.5,.8],[.5,.2]])h.c.addDirectionPoint(p);
 await h.c.startDirection();assert.equal(h.requests.length,0);
 h.$('direction-review').onclick();await h.c.startDirection();const d=h.requests[0].data;
 assert.equal(d.direction.reference_seconds,0);assert.equal(d.calibration,null);assert.equal(d.incidents,null);
 assert.equal(d.direction.type,'configured_direction');assert.equal(d.direction.reviewed,true);
});
test('changed source, revision or preview cannot reuse configured direction',async()=>{
 const h=harness();h.$('direction-mark').onclick();
 for(const p of [[.1,.1],[.9,.1],[.9,.9],[.1,.9],[.5,.8],[.5,.2]])h.c.addDirectionPoint(p);
 h.$('direction-review').onclick();h.state.media.source_revision='new';await h.c.startDirection();
 h.state.media.source_revision='r';h.state.preview.timestamp=.1;await h.c.startDirection();
 h.state.preview.timestamp=0;h.state.media.id='other';await h.c.startDirection();assert.equal(h.requests.length,0);
});
test('focused IVP permits an explicit helmet dispatch without enabling broad scene inference',async()=>{
 let called=false;const c=vm.createContext({ivpPage:()=>true,fetch:async()=>{called=true;return {ok:true,json:async()=>({})};},token:'x'});
 vm.runInContext(source.slice(source.indexOf('async function performApi('),source.indexOf('let apiQueue')),c);
 await c.performApi('/api/helmet',{media:'m',source_revision:'r',timestamp:0});assert.equal(called,true);
});

test('immediate motion summary includes direction counts and invalidation notice',()=>{
 const c=vm.createContext({escapeHTML:String,incidentSummary:()=>'',speedOutcomeSummary:()=>''});
 vm.runInContext(source.slice(source.indexOf('function directionSummary(')),c);
 vm.runInContext(source.slice(source.indexOf('function motionSummary('),source.indexOf('function showMotionResult(')),c);
 const motion={direction:{notice:'Configured assumption, never legal finding',episodes:1,sample_counts:{withheld:2},review_invalidated:true,method:'Existing evaluator'},timeline:[],line:{axis:'horizontal'},crossings:{positive:0,negative:0},peak_activity_percent:0,measured_seconds:0,suppressed_frames:0,limitations:[]};
 const html=c.motionSummary(motion);assert.match(html,/Configured-direction review/);assert.match(html,/1 opposite-direction episodes/);assert.match(html,/Review invalidated/);assert.match(html,/never legal finding/);
});

test('saved motion renders the shared direction report once and failed runs expose none',async()=>{
 const direction={notice:'configured only',episodes:1,sample_counts:{withheld:2},review_invalidated:true,method:'existing evaluator'};
 const motion={direction,timeline:[],line:{axis:'horizontal'},crossings:{positive:0,negative:0},peak_activity_percent:0,measured_seconds:0,suppressed_frames:0,limitations:[]};
 for(const status of ['complete','failed']){
  const elements=new Map(),errors=[];const run={id:'run',source:'fixture.mp4',status:{state:status},downloads:[],report:{motion}};
  const c=vm.createContext({runOpenRevision:0,api:async()=>run,escapeHTML:String,links:()=>'',incidentSummary:()=>'',speedOutcomeSummary:()=>'',
   $:id=>{if(id==='reuse-settings')return null;if(!elements.has(id))elements.set(id,{showModal(){},querySelector:()=>null});return elements.get(id);},
   evidencePanel:()=>'',bindEvidencePanel(){},showMotionResult(){},showJob(){},notify:e=>errors.push(e)});
  vm.runInContext(source.slice(source.indexOf('function directionSummary(')),c);
  vm.runInContext(source.slice(source.indexOf('function motionSummary('),source.indexOf('function showMotionResult(')),c);
  vm.runInContext(source.slice(source.indexOf('async function openRun('),source.indexOf('async function startReview(')),c);
  await c.openRun('run');assert.deepEqual(errors,[]);
  const html=elements.get('result-body').innerHTML;
  assert.equal((html.match(/<h3>Configured-direction review/g)||[]).length,status==='complete'?1:0);
 }
});
