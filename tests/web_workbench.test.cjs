const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

function navigation(){
 const elements=new Map(),saved=new Map(),calls={library:0,pauses:0,paint:0};
 const body={dataset:{},classList:{remove(){}}};
 const get=id=>{if(!elements.has(id))elements.set(id,{hidden:false,value:'',setAttribute(){},focus(){},querySelector(){return {pause(){calls.pauses++;}};}});return elements.get(id);};
 const state={media:{id:'original',source_revision:'sha',kind:'video'},preview:{timestamp:2.9},settings:{gamma:1.2},region:[.1,.2,.8,.9],job:'running-job',review:{id:'live-overview',cancelling:true},view:'stages',area:'ivp'};
 const calibration=[['measured',5,12]],original=JSON.stringify({media:state.media,preview:state.preview,settings:state.settings,region:state.region,job:state.job,review:state.review,calibration});
 const context=vm.createContext({state,calibrationPoints:calibration,$:get,document:{body,querySelectorAll(){return [];}},
  localStorage:{setItem(k,v){saved.set(k,v);}},setArea(area){state.area=area;get('workspace').hidden=false;get('library').hidden=true;},
  paint(){calls.paint++;},paintWorkbench(){},loadRuns(){calls.library++;},
  api(){throw Error('Navigation must not start inference or mutate exports');}});
 vm.runInContext(source.slice(source.indexOf('const workbenchTasks='),source.indexOf('function paintWorkbench(')),context);
 return {context,state,get,saved,calls,body,unchanged(){assert.equal(JSON.stringify({media:state.media,preview:state.preview,settings:state.settings,region:state.region,job:state.job,review:state.review,calibration}),original);}};
}

test('all six feature routes preserve source, calibration, settings and active job/review state without inference',()=>{
 const h=navigation();
 for(const task of ['lab','tracking','markings','detection','overview','library']){
  h.context.selectWorkbench(task);h.unchanged();assert.equal(h.body.dataset.workbench,task);assert.equal(h.saved.get('traffic-lab-tool'),task);
  assert.equal(h.get('workspace').hidden,task==='library');assert.equal(h.get('library').hidden,task!=='library');
 }
 assert.equal(h.calls.library,1);
});

test('AI and IVP navigation returns to the requested task without discarding physical calibration or cancelling work',()=>{
 const h=navigation();h.context.selectWorkbench('tracking');assert.equal(h.state.view,'processed');
 h.context.selectWorkbench('overview');assert.equal(h.state.area,'ai');
 h.context.selectWorkbench('lab');assert.equal(h.state.view,'stages');assert.equal(h.state.area,'ivp');h.unchanged();
});

test('invalid persisted routes fail back to the image laboratory',()=>{
 const h=navigation();for(const value of ['__proto__','constructor','missing',null]){h.context.selectWorkbench(value);assert.equal(h.body.dataset.workbench,'lab');h.unchanged();}
 assert.equal(h.calls.library,0);
});

test('leaving tracking pauses its hidden preview while leaving job identity and cancellation state intact',()=>{
 const h=navigation();h.context.selectWorkbench('tracking');assert.equal(h.calls.pauses,0);
 h.context.selectWorkbench('library');assert.equal(h.calls.pauses,1);h.unchanged();
});

test('tracking hides an incompatible still source without losing IVP work or starting processing',()=>{
 const h=navigation();
 h.state.media.kind='image';
 const before=JSON.stringify(h.state);
 vm.runInContext("workbenchTask='tracking';paintTrackingSource();",h.context);
 assert.equal(h.body.dataset.videoSource,'false');
 assert.equal(h.get('tracking-empty').hidden,false);
 assert.equal(h.get('upload').accept,'.mp4,.mov,.avi');
 assert.equal(JSON.stringify(h.state),before);
 vm.runInContext("workbenchTask='lab';paintTrackingSource();",h.context);
 assert.equal(h.get('tracking-empty').hidden,true);
 assert.match(h.get('upload').accept,/\.png/);
 assert.equal(JSON.stringify(h.state),before);
});

test('tracking switches between its video prompt and calibration preview as sources change',()=>{
 const h=navigation();
 vm.runInContext("workbenchTask='tracking';paintTrackingSource();",h.context);
 assert.equal(h.body.dataset.videoSource,'true');
 assert.equal(h.get('tracking-empty').hidden,true);
 h.state.media=null;
 vm.runInContext('paintTrackingSource();',h.context);
 assert.equal(h.body.dataset.videoSource,'false');
 assert.equal(h.get('tracking-empty').hidden,false);
 h.state.media={id:'new-video',kind:'video'};
 vm.runInContext('paintTrackingSource();',h.context);
 assert.equal(h.body.dataset.videoSource,'true');
 assert.equal(h.get('tracking-empty').hidden,true);
 assert.equal(h.state.job,'running-job');
 assert.equal(h.state.review.cancelling,true);
});

function resultTabs(){
 const names=['overview','evidence','downloads','record'],media={pauses:0,pause(){this.pauses++;}},panels=new Map(names.map(n=>['saved-'+n,{hidden:false,originalContent:{source:'immutable-'+n},querySelectorAll(){return n==='overview'?[media]:[];}}]));
 const buttons=names.map(name=>({dataset:{resultTab:name},attrs:{},tabIndex:0,setAttribute(k,v){this.attrs[k]=v;},focus(){this.focused=true;}}));
 const context=vm.createContext({document:{querySelectorAll(){return buttons;}},$:id=>panels.get(id)});
 vm.runInContext(source.slice(source.indexOf('function showSavedResultTab('),source.indexOf('function organizeSavedResult(')),context);
 return {context,buttons,panels,media};
}

test('saved result tabs retain original evidence nodes and provide one keyboard tab stop',()=>{
 const h=resultTabs(),evidence=h.panels.get('saved-evidence').originalContent;
 h.context.showSavedResultTab('evidence',true);
 assert.equal(h.buttons.filter(b=>b.tabIndex===0).length,1);assert.equal(h.buttons[1].attrs['aria-selected'],'true');assert.equal(h.buttons[1].focused,true);
 assert.equal(h.panels.get('saved-overview').hidden,true);assert.equal(h.panels.get('saved-evidence').hidden,false);
 h.context.showSavedResultTab('downloads');assert.equal(h.panels.get('saved-evidence').originalContent,evidence);
});

test('an invalid result section opens overview and keeps the full record available',()=>{
 const h=resultTabs();h.context.showSavedResultTab('invalid');assert.equal(h.panels.get('saved-overview').hidden,false);assert.equal(h.panels.get('saved-record').originalContent.source,'immutable-record');
});

test('evidence and download navigation pauses hidden video without removing its evidence',()=>{
 const h=resultTabs();h.context.showSavedResultTab('overview');assert.equal(h.media.pauses,0);
 h.context.showSavedResultTab('evidence');assert.equal(h.media.pauses,1);
 assert.equal(h.panels.get('saved-overview').originalContent.source,'immutable-overview');
});

test('unknown or active work cannot be dismissed as finished, while terminal results retain downloads',()=>{
 const elements=new Map(),saved=new Map();
 const context=vm.createContext({state:{},jobTimer:null,$:id=>{if(!elements.has(id))elements.set(id,{});return elements.get(id);},showMotionResult(){},links(){return 'original links';},jobMessage(){return 'original status';},pollJob(){},clearTimeout(){},setTimeout(){return 1;},localStorage:{setItem(k,v){saved.set(k,v);},getItem(k){return saved.get(k);},removeItem(k){saved.delete(k);}}});
 vm.runInContext(source.slice(source.indexOf('function showJob('),source.indexOf('function showFrameExport(')),context);
 for(const state of ['queued','running','encoding','unknown','complete','cancelled']){
  context.showJob({id:'owned',source:'original.mp4',status:{state},downloads:state==='complete'?[{name:'processed.mp4'}]:[]});
  assert.equal(elements.get('dismiss-job').hidden,['queued','running','encoding','unknown'].includes(state));
  assert.equal(elements.get('job-output-options').hidden,state!=='complete');
 }
});
