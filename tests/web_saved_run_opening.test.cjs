const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

function browser(){
 const elements=new Map(),pending=new Map(),notices=[],shown=[],organized=[],jobs=[];
 const state={media:{id:'current-footage',source_revision:'current-revision',kind:'video'},
  preview:{timestamp:2.4},settings:{gamma:1.2},region:[.1,.2,.8,.9],job:'current-worker',review:{id:'current-review'}};
 const calibration=[[.1,.2],[.8,.2],[.8,.9],[.1,.9]];
 const preserved=()=>JSON.stringify({media:state.media,preview:state.preview,settings:state.settings,
  region:state.region,job:state.job,review:state.review,calibration});
 const original=preserved();
 const get=id=>{
  if(id==='reuse-settings')return null;
  if(!elements.has(id))elements.set(id,{textContent:'',innerHTML:'',hidden:false,value:'',setAttribute(){},focus(){},
   querySelector(){return null;},showModal(){this.open=true;shown.push(get('result-title').textContent);}});
  return elements.get(id);
 };
 const context=vm.createContext({state,calibrationPoints:calibration,$:get,
  document:{body:{dataset:{},classList:{remove(){}}},querySelectorAll(){return [];}},
  localStorage:{setItem(){}},paint(){},paintWorkbench(){},loadRuns(){},setArea(area){state.area=area;},
  api(url){const id=new URL(url,'http://local').searchParams.get('id');return new Promise((resolve,reject)=>pending.set(id,{resolve,reject}));},
  escapeHTML:value=>String(value),links:items=>items.map(item=>item.url).join(' '),
  motionSummary:motion=>'motion:'+motion.marker,evidencePanel:()=>'',surveySummary:()=>'',aiAssessmentPanel:()=>'',
  bindEvidencePanel(){},bindAiAssessmentPanel(){},organizeSavedResult:run=>organized.push(run.id),
  showMotionResult:run=>shown.push('motion:'+run.id),showJob:run=>{jobs.push(run.id);state.job=run.id;},
  notify:message=>notices.push(message)});
 vm.runInContext(source.slice(source.indexOf('const workbenchTasks='),source.indexOf('function paintWorkbench(')),context);
 vm.runInContext(source.slice(source.indexOf('async function openRun('),source.indexOf('function aiAssessmentPanel(')),context);
 context.selectWorkbench('library');
 return {context,pending,get,notices,shown,organized,jobs,state,unchanged(){assert.equal(preserved(),original);}};
}

function saved(id,state='complete'){
 return {id,source:id+'.mp4',status:{state,message:id+' saved'},
  downloads:[{url:'/download/'+id+'/motion-evidence.zip'}],preview:null,report:{motion:{marker:id}}};
}

test('saved results keep the latest selection when responses arrive out of order',async()=>{
 const h=browser(),old=h.context.openRun('old'),latest=h.context.openRun('latest');
 h.pending.get('latest').resolve(saved('latest'));await latest;
 h.pending.get('old').resolve(saved('old','running'));await old;
 assert.equal(h.get('result-title').textContent,'latest.mp4');
 assert.match(h.get('result-body').innerHTML,/\/download\/latest\/motion-evidence.zip/);
 assert.doesNotMatch(h.get('result-body').innerHTML,/\/download\/old\//);
 assert.deepEqual(h.organized,['latest']);assert.deepEqual(h.jobs,[]);h.unchanged();
});

test('an obsolete saved-result error cannot notify over a newer successful selection',async()=>{
 const h=browser(),old=h.context.openRun('old'),latest=h.context.openRun('latest');
 h.pending.get('latest').resolve(saved('latest'));await latest;
 h.pending.get('old').reject(Error('obsolete error'));await old;
 assert.deepEqual(h.notices,[]);assert.deepEqual(h.organized,['latest']);h.unchanged();
});

test('leaving and returning to Saved runs discards a pending result without retargeting jobs',async()=>{
 const h=browser(),request=h.context.openRun('old');
 h.context.selectWorkbench('tracking');h.context.selectWorkbench('library');
 h.pending.get('old').resolve(saved('old','running'));await request;
 assert.deepEqual(h.organized,[]);assert.deepEqual(h.shown,[]);assert.deepEqual(h.jobs,[]);h.unchanged();
});

test('a saved-result error after workspace navigation is discarded',async()=>{
 const h=browser(),request=h.context.openRun('old');h.context.selectWorkbench('tracking');
 h.pending.get('old').reject(Error('obsolete navigation error'));await request;
 assert.deepEqual(h.notices,[]);assert.deepEqual(h.organized,[]);h.unchanged();
});

test('a latest-request error stays visible and an older success cannot replace it',async()=>{
 const h=browser(),old=h.context.openRun('old'),latest=h.context.openRun('latest');
 h.pending.get('latest').reject(Error('Selected result is unavailable'));await latest;
 h.pending.get('old').resolve(saved('old'));await old;
 assert.deepEqual(h.notices,['Selected result is unavailable']);assert.deepEqual(h.organized,[]);h.unchanged();
});

test('a normal saved result opens with its own downloads and preserves current IVP work',async()=>{
 const h=browser(),request=h.context.openRun('latest');
 h.pending.get('latest').resolve(saved('latest'));await request;
 assert.equal(h.get('result-dialog').open,true);assert.equal(h.get('result-title').textContent,'latest.mp4');
 assert.match(h.get('result-body').innerHTML,/\/download\/latest\/motion-evidence.zip/);
 assert.deepEqual(h.organized,['latest']);assert.deepEqual(h.notices,[]);h.unchanged();
});
