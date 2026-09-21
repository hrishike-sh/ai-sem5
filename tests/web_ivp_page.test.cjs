const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

function harness(pathname='/',persisted={}){
 const saved=new Map(Object.entries(persisted)),elements=new Map(),events={},requests=[];
 const calls={sceneRestore:0,library:0,pushes:0,closed:0};
 const get=id=>{if(!elements.has(id))elements.set(id,{value:'0',hidden:false,dataset:{},setAttribute(){},removeAttribute(){},focus(){},insertAdjacentHTML(){},querySelector(){return {pause(){}};},close(){calls.closed++;},scrollIntoView(){}});return elements.get(id);};
 const body={dataset:{appPage:pathname.startsWith('/ivp')?'ivp':'full'},classList:{remove(){},toggle(){}}};
 const document={body,title:'',querySelectorAll(){return [];}};
 const window={location:{pathname},history:{pushState(_s,_t,p){window.location.pathname=p;calls.pushes++;}},addEventListener(name,fn){events[name]=fn;}};
 const state={media:{id:'kept',source_revision:'revision',kind:'video'},preview:{timestamp:2.9},settings:{gamma:1.2},region:[.1,.2,.8,.9],job:'existing-job',review:{id:'active-review',cancelling:true},view:'stages',area:'ivp'};
 const calibration=[[.1,.2],[.8,.2],[.9,.9],[.1,.9]],before=JSON.stringify({media:state.media,preview:state.preview,settings:state.settings,region:state.region,job:state.job,review:state.review,calibration});
 const context=vm.createContext({state,document,window,calibrationPoints:calibration,$:get,workspaceRestoring:false,
  localStorage:{getItem:k=>saved.get(k)||null,setItem(k,v){saved.set(k,v);}},
  setArea(area){state.area=area;get('workspace').hidden=false;get('library').hidden=true;},paint(){},paintWorkbench(){},actions(){},
  loadRuns(){calls.library++;},restoreSceneReviews:async()=>{calls.sceneRestore++;},
  api(){throw Error('Navigation must not start processing');},
  fetch:async(route,options)=>{requests.push({route,options});return {ok:true,json:async()=>({ok:true})};},token:'test-token'});
 vm.runInContext(source.slice(source.indexOf('const workbenchTasks='),source.indexOf('function paintWorkbench(')),context);
 return {context,state,get,body,document,window,events,saved,requests,calls,unchanged(){assert.equal(JSON.stringify({media:state.media,preview:state.preview,settings:state.settings,region:state.region,job:state.job,review:state.review,calibration}),before);}};
}

test('direct IVP route ignores full-app Tracking/AI preferences and clamps an incompatible IVP preference',()=>{
 for(const prior of ['tracking','detection','overview','library']){
  const h=harness('/ivp',{'traffic-lab-tool':prior,'traffic-lab-ivp-tool':'detection'});
  h.context.applyPageRoute();h.unchanged();
  assert.equal(h.body.dataset.appPage,'ivp');assert.equal(h.body.dataset.workbench,'lab');
  assert.equal(h.saved.get('traffic-lab-tool'),prior);assert.equal(h.saved.get('traffic-lab-ivp-tool'),'lab');
  assert.match(h.document.title,/IVP/);
 }
});

test('IVP permits classical and explicit bounded traffic tools while fencing broad workspaces without discarding work',()=>{
 const h=harness('/ivp/');h.context.applyPageRoute();
 for(const task of ['markings','library','lab','tracking','zebra','helmet','direction','red_line','restricted_red','speed_review','required_stop','detection','overview','constructor','__proto__']){
  h.context.selectWorkbench(task);h.unchanged();
  assert.equal(h.body.dataset.workbench,['markings','library','lab','tracking','zebra','direction','red_line','restricted_red','speed_review','required_stop'].includes(task)?task:'lab');
  assert.equal(h.state.area,'ivp');
 }
 assert.equal(h.calls.library,1);assert.equal(h.calls.sceneRestore,0);
});

test('page links and browser back/forward preserve source, settings, calibration, jobs and separate tool preferences',()=>{
 const h=harness('/',{'traffic-lab-tool':'tracking'});h.context.setupPageNavigation();h.context.applyPageRoute();
 h.context.navigatePage('/ivp');assert.equal(h.body.dataset.workbench,'lab');h.unchanged();
 h.context.selectWorkbench('markings');h.context.navigatePage('/');
 assert.equal(h.body.dataset.workbench,'tracking');h.unchanged();
 h.window.location.pathname='/ivp';h.events.popstate();assert.equal(h.body.dataset.workbench,'markings');h.unchanged();
 h.window.location.pathname='/';h.events.popstate();assert.equal(h.body.dataset.workbench,'tracking');h.unchanged();
 assert.equal(h.saved.get('traffic-lab-tool'),'tracking');assert.equal(h.saved.get('traffic-lab-ivp-tool'),'markings');
});

test('route changes invalidate pending saved opens and close their modal without retargeting the current job',()=>{
 const h=harness('/');h.context.applyPageRoute();
 const before=vm.runInContext('runOpenRevision',h.context);
 h.context.navigatePage('/ivp');
 assert.ok(vm.runInContext('runOpenRevision',h.context)>before);assert.equal(h.calls.closed,1);h.unchanged();
});

test('modifier clicks retain native link behaviour while plain clicks use state-preserving navigation',()=>{
 const h=harness('/');h.context.setupPageNavigation();
 const click=h.get('ivp-page-link').onclick;
 for(const modifier of ['ctrlKey','metaKey','altKey','shiftKey']){
  click({button:0,[modifier]:true,preventDefault(){throw Error('Modified link must remain native');}});
 }
 assert.equal(h.calls.pushes,0);
 let prevented=false;click({button:0,preventDefault(){prevented=true;}});
 assert.equal(prevented,true);assert.equal(h.window.location.pathname,'/ivp');h.unchanged();
});

test('the brand home link also returns to the full app without a destructive document reload',()=>{
 const h=harness('/ivp',{'traffic-lab-tool':'tracking'});h.context.setupPageNavigation();
 let prevented=false;h.get('app-home-link').onclick({button:0,preventDefault(){prevented=true;}});
 assert.equal(prevented,true);assert.equal(h.window.location.pathname,'/');assert.equal(h.body.dataset.workbench,'tracking');h.unchanged();
});

function startup(pathname){
 const h=harness(pathname,{'traffic-lab-tool':'overview'}),noop=()=>{};
 for(const name of ['setupDirection','startHelmet','setupZebra','motionModeChanged','makeControls','clearTarget','clearCalibration','drawCalibration','startMotion','paintMotionLine','openSyntheticExample','paintIncidents','runSurvey','selectSurveyRegion','selectStage','updateWipe','scheduleRuns','cancelReview','startReview','downloadJSON'])h.context[name]=noop;
 Object.assign(h.context,{copy:v=>JSON.parse(JSON.stringify(v)),escapeHTML:String,setupWorkbench:()=>h.context.setupPageNavigation(),setMode:noop,notify:noop,
  api:async route=>{assert.equal(route,'/api/config');return {defaults:{gamma:1},presets:{Original:{gamma:1}},samples:[],vehicle_tracking_ready:false};}});
 vm.runInContext(source.slice(source.indexOf('async function init('),source.indexOf('function setMode(')),h.context);
 return h;
}

test('direct IVP initialization restores no scene overview and lands in classical Lab',async()=>{
 const h=startup('/ivp');await h.context.init();
 assert.equal(h.calls.sceneRestore,0);assert.equal(h.body.dataset.workbench,'lab');assert.equal(h.get('connection').textContent,'Connected');
 assert.equal(h.saved.get('traffic-lab-tool'),'overview');
});

test('navigation during delayed initialization wins over old full-app startup state',async()=>{
 const h=startup('/');let resolve;
 h.context.api=()=>new Promise(ok=>{resolve=ok;});const pending=h.context.init();
 h.context.navigatePage('/ivp');resolve({defaults:{gamma:1},presets:{Original:{gamma:1}},samples:[],vehicle_tracking_ready:false});await pending;
 assert.equal(h.body.dataset.appPage,'ivp');assert.equal(h.body.dataset.workbench,'lab');assert.equal(h.calls.sceneRestore,0);
});

test('full-app initialization keeps its persisted tool and passive overview restoration',async()=>{
 const h=startup('/');await h.context.init();
 assert.equal(h.body.dataset.workbench,'overview');assert.equal(h.calls.sceneRestore,1);
});

test('entering the full application after direct IVP startup restores existing optional work once',async()=>{
 const h=startup('/ivp');await h.context.init();
 h.context.navigatePage('/');await Promise.resolve();await Promise.resolve();
 assert.equal(h.calls.sceneRestore,1);
 h.context.navigatePage('/ivp');h.context.navigatePage('/');await Promise.resolve();
 assert.equal(h.calls.sceneRestore,1);
});

test('IVP selectively permits bounded detector/motion actions but fences broad analyzer and scene/LLM starts',async()=>{
 const h=harness('/ivp');h.context.applyPageRoute();
 vm.runInContext(source.slice(source.indexOf('async function performApi('),source.indexOf('let apiQueue=')),h.context);
 for(const [route,data] of [['/api/review',{}],['/api/evidence-review',{}],['/api/job',{ai:true}]]){
  await assert.rejects(h.context.performApi(route,data),/full application/i);
 }
 assert.equal(h.requests.length,0);
 for(const [route,data] of [['/api/preview',{}],['/api/export',{}],['/api/survey',{}],['/api/job',{ai:false}],['/api/cancel',{id:'existing'}],['/api/review/cancel',{id:'existing'}],['/api/helmet',{media:'m',source_revision:'r',timestamp:0}],['/api/zebra',{media:'m',source_revision:'r'}],['/api/motion',{media:'m',source_revision:'r',tracking_mode:'ivp'}],['/api/motion',{media:'m',source_revision:'r',tracking_mode:'vehicles',incidents:{enabled:true}}]])await h.context.performApi(route,data);
 assert.equal(h.requests.length,10);
 h.body.dataset.appPage='full';await h.context.performApi('/api/job',{ai:true});assert.equal(h.requests.length,11);
});

test('an AI request queued on the main page cannot begin after navigation to IVP',async()=>{
 const h=harness('/');let release;const blocker=new Promise(ok=>{release=ok;});
 h.context.fetch=async(route)=>{h.requests.push(route);if(route==='/api/preview')await blocker;return {ok:true,json:async()=>({})};};
 vm.runInContext(source.slice(source.indexOf('async function performApi('),source.indexOf('function payload(')),h.context);
 const first=h.context.api('/api/preview',{});await Promise.resolve();
 const queued=h.context.api('/api/job',{ai:true});h.context.navigatePage('/ivp');release();await first;
 await assert.rejects(queued,/full application/i);assert.deepEqual(h.requests,['/api/preview']);h.unchanged();
});


test('fresh direct IVP startup defaults tracking to classical without starting a job',async()=>{
 const h=startup('/ivp');h.get('motion-mode').value='vehicles';
 h.context.api=async route=>{assert.equal(route,'/api/config');return {defaults:{},presets:{Original:{}},samples:[],vehicle_tracking_ready:true};};
 await h.context.init();assert.equal(h.get('motion-mode').value,'ivp');assert.equal(h.requests.length,0);assert.equal(h.body.dataset.workbench,'lab');
});

test('bounded requests queued across IVP navigation retain explicit source and geometry',async()=>{
 for(const route of ['/api/helmet','/api/zebra','/api/motion']){
  const h=harness('/');let release;const blocker=new Promise(ok=>{release=ok;});
  h.context.fetch=async(path,options)=>{h.requests.push({path,options});if(path==='/api/preview')await blocker;return {ok:true,json:async()=>({})};};
  vm.runInContext(source.slice(source.indexOf('async function performApi('),source.indexOf('function payload(')),h.context);
  const first=h.context.api('/api/preview',{});await Promise.resolve();
  const payload={media:'captured',source_revision:'captured-revision',timestamp:0,region:{reviewed:true,points:[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]}};
  const queued=h.context.api(route,payload);h.context.navigatePage('/ivp');release();await first;await queued;
  assert.equal(h.requests.length,2);assert.equal(h.requests[1].path,route);assert.deepEqual(JSON.parse(h.requests[1].options.body),payload);
  assert.equal(h.requests[1].options.headers['X-Traffic-Token'],'test-token');h.unchanged();
 }
});
