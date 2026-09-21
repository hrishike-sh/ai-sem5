const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

test('completed jobs expose an explicit same-page result action without auto-opening',()=>{
 const elements=new Map(),opened=[];const $=id=>{if(!elements.has(id))elements.set(id,{});return elements.get(id);};
 const c=vm.createContext({state:{},$,showMotionResult(){},jobMessage:()=>'',links:()=>'',localStorage:{setItem(){},getItem(){},removeItem(){}},
  setTimeout:()=>1,clearTimeout(){},pollJob(){},jobTimer:null,openRun:id=>opened.push(id)});
 vm.runInContext(source.slice(source.indexOf('function showJob('),source.indexOf('async function pollJob(')),c);
 c.showJob({id:'complete',source:'frame.png',status:{state:'complete'},downloads:[]});
 assert.equal($('view-job-result').hidden,false);assert.deepEqual(opened,[]);$('view-job-result').onclick();assert.deepEqual(opened,['complete']);
 c.showJob({id:'partial',source:'frame.png',status:{state:'running'},downloads:[]});assert.equal($('view-job-result').hidden,true);
});


const features=fs.readFileSync(path.join(__dirname,'../web/ivp_features.js'),'utf8');
function scoped(feature){
 const values={'motion-axis':{value:'horizontal'},'motion-position':{value:'65'},'motion-use-roi':{checked:false},'motion-mode':{value:'ivp'},'motion-view':{value:'stages'},'incident-enabled':{checked:true},'feature-context-confirmed':{checked:true},'incident-start':{value:'15'},'incident-end':{value:'85'},'incident-direction':{value:'positive'},'incident-condition':{value:feature==='required_stop'?'stop_required':'red'},'incident-intervals':{value:''},'incident-line':{checked:true},'incident-zone':{checked:true},'incident-zone-rect':{value:'15,65,85,78'},'incident-restricted':{checked:true},'incident-approach':{value:'15,45,85,65'},'incident-limit':{value:'10'},'incident-calibration-confirmed':{checked:true},'incident-real-time':{checked:true},'incident-synthetic':{checked:true}};
 const c=vm.createContext({workbenchTask:feature,state:{preview:{timestamp:0},region:null},$:id=>values[id],speedCalibration:()=>({source_revision:'r'}),targetRectangle:null});
 vm.runInContext(features,c);return {c,values};
}
test('each incident page serializes only its own geometry and context despite hidden stale fields',()=>{
 for(const f of ['red_line','restricted_red','speed_review','required_stop']){
  const h=scoped(f),d=h.c.scopedTrafficRequest({id:'m',source_revision:'r'}),i=d.incidents;assert.equal(d.ui_feature,f);assert.equal(d.tracking_mode,'vehicles');assert.equal(d.target_rectangle,null);
  assert.equal(!!i.line,['red_line','required_stop'].includes(f));assert.equal(!!i.zone,f==='restricted_red');assert.equal(!!i.approach_zone,f==='required_stop');assert.equal(i.speed_limit_kmh,f==='speed_review'?10:null);
  assert.equal(!!d.calibration,['speed_review','required_stop'].includes(f));assert.equal('zebra' in d,false);assert.equal('direction' in d,false);
 }
});
test('motion page ignores stale enabled incident settings without loading a detector',()=>{
 const h=scoped('tracking'),d=h.c.scopedTrafficRequest({id:'m',source_revision:'r'});assert.equal(d.incidents,null);assert.equal(d.tracking_mode,'ivp');
});
test('wrong feature conditions and unconfirmed context reject before dispatch',()=>{
 const h=scoped('required_stop');h.values['incident-condition'].value='red';assert.throws(()=>h.c.scopedTrafficRequest({id:'m',source_revision:'r'}),/only its own/);
 h.values['incident-condition'].value='stop_required';h.values['feature-context-confirmed'].checked=false;assert.throws(()=>h.c.scopedTrafficRequest({id:'m',source_revision:'r'}),/Confirm/);
});
test('course units follow contributing operations and retain titles/parameters/noops',()=>{
 const h=scoped('lab');const run={status:{state:'complete'},report:{ivp_demo:{steps:[{id:'gray',title:'Black & white grayscale',parameters:{}},{id:'denoised',title:'Gaussian filtering',parameters:{kernel:3}},{id:'stretch',title:'Percentile stretch',parameters:{percentile:1}},{id:'opening',title:'Morphological opening',parameters:{kernel:3}}],omitted_steps:[{title:'Disabled sharpening'}]}}};
 const e=h.c.unitsForRun(run,'lab');assert.deepEqual(Array.from(e.units),[1,2,4]);assert.equal(e.groups[2][0].parameters.kernel,3);assert.equal(e.omitted.length,1);
 const image={status:{state:'complete'},report:{helmet:{source_binding:{timestamp_seconds:3}}}};assert.deepEqual(Array.from(h.c.unitsForRun(image,'helmet').units),[]);
 const video={status:{state:'complete'},report:{motion:{frames:180,uses_ai:true,speed:{calibrated:true}}}};assert.deepEqual(Array.from(h.c.unitsForRun(video,'speed_review').units),[8]);assert.equal(h.c.unitsForRun(video,'speed_review').units.includes(3),false);
});
test('required-stop identity is explicit even when its base condition is unknown',()=>{
 const h=scoped('required_stop');assert.equal(h.c.motionRunFeature({report:{source_info:{ui_feature:'required_stop'},motion:{incidents:{configuration:{enabled:true,condition:'unknown',line:[[0,0],[1,1]]}}}}}),'required_stop');
 assert.equal(h.c.motionRunFeature({report:{motion:{incidents:{configuration:{enabled:true,condition:'unknown',line:[[0,0],[1,1]]}}}}}),null);
});

test('footer evidence is source/revision/timestamp/config bound and late results cannot claim changed settings',()=>{
 const footer={innerHTML:''},time={value:'0'};const state={media:{id:'m',source_revision:'r'},preview:{timestamp:0},settings:{gamma:1.5}};
 const c=vm.createContext({workbenchTask:'lab',state,$:id=>id==='course-units'?footer:id==='timestamp'?time:undefined,escapeHTML:String});vm.runInContext(features,c);
 const run={id:'run',source:'fixture',status:{state:'complete'},report:{source_binding:{media_id:'m',source_revision:'r',timestamp_seconds:0},ivp_demo:{steps:[{id:'gamma',title:'Gamma correction',parameters:{gamma:1.5}}]}}};
 const captured=c.captureFeatureSubmission();state.settings.gamma=2;c.bindFeatureSubmission(run,captured);c.rememberFeatureRun(run);assert.doesNotMatch(footer.innerHTML,/Unit 2 —/);
 state.settings.gamma=1.5;c.rememberFeatureRun(run);assert.match(footer.innerHTML,/Unit 2 — Image enhancement/);assert.match(footer.innerHTML,/Gamma correction/);
 time.value='1';c.updateFeatureUnits();assert.doesNotMatch(footer.innerHTML,/Unit 2 —/);time.value='0';state.media.source_revision='new';c.updateFeatureUnits();assert.doesNotMatch(footer.innerHTML,/Unit 2 —/);
 state.media.source_revision='r';c.rememberFeatureRun({id:'pending',status:{state:'failed'},report:{info:{ui_feature:'lab'}}});assert.doesNotMatch(footer.innerHTML,/Unit 2 —/);
});
test('feature links retain separate IVP query views without starting processing',()=>{
 const calls=[],c=vm.createContext({ivpPage:()=>true,window:{location:{search:'?feature=helmet'},history:{pushState(_a,_b,url){calls.push(url);}}},selectWorkbench:(task,focus)=>calls.push([task,focus])});vm.runInContext(features,c);
 assert.equal(c.featureFromQuery(),'helmet');c.navigateTool('required_stop');assert.deepEqual(calls,['/ivp?feature=required_stop',['required_stop',true]]);
});

test('a delayed frame export cannot replace output after source or settings changes',async()=>{
 for(const change of ['source','settings']){
  const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,{value:id==='ivp-goal'?'regions':'0',innerHTML:'',hidden:false});return elements.get(id);};
  get('frame-export-result').innerHTML='previous output';const state={media:{id:'m',source_revision:'r'},preview:{timestamp:0},settings:{gamma:1.5}};let resolve;
  const wait=new Promise(ok=>{resolve=ok;});const c=vm.createContext({state,workbenchTask:'lab',$:get,escapeHTML:String,links:()=>'',notify(){},actions(){},masks:()=>[],payload:()=>({media:'m',source_revision:'r',settings:{gamma:1.5}}),api:()=>wait});
  vm.runInContext(features,c);vm.runInContext(source.slice(source.indexOf('function showFrameExport('),source.indexOf('async function pollJob(')),c);
  const assignment=source.slice(source.indexOf("$('export-frame').onclick=async()=>"),source.indexOf("$('export-video').onclick"));vm.runInContext(assignment,c);
  const pending=get('export-frame').onclick();if(change==='source')state.media.source_revision='new';else state.settings.gamma=2;
  resolve({id:'old-export',source:'old.png',status:{state:'complete'},downloads:[],report:{source_binding:{media_id:'m',source_revision:'r',timestamp_seconds:0},settings:{gamma:1.5}}});await pending;
  assert.equal(get('frame-export-result').innerHTML,'previous output',change);
 }
});
