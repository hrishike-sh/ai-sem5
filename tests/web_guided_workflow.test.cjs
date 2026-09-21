const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/ivp_features.js'),'utf8');

function guide(){
 const c=vm.createContext({});
 vm.runInContext(source,c);
 return c.guidedPlan;
}
test('first-time guide progresses through footage, setup, run and job status without starting work',()=>{
 const plan=guide(),ready={media:true,kind:'video',preview:true};
 assert.equal(plan('lab',{}).target,'source-picker');
 assert.equal(plan('direction',{...ready,kind:'image',videoRequired:true}).stage,1);
 assert.equal(plan('lab',{...ready,loading:true}).target,'preview-state');
 assert.equal(plan('helmet',{...ready,dependency:'Weights unavailable'}).target,'ivp-workspace');
 assert.equal(plan('direction',{...ready,firstFrame:true}).target,'direction-first');
 assert.equal(plan('lab',ready).target,'ivp-generate');
 assert.equal(plan('lab',{...ready,running:true}).target,'active-job');
});
test('finishing a polygon reveals explicit consent and never supplies it',()=>{
 const plan=guide();
 for(const [feature,review] of [['zebra','zebra-review'],['direction','direction-review']]){
  const done=plan(feature,{media:true,kind:'video',preview:true,drawComplete:true});
  assert.equal(done.target,review);assert.equal(done.activate,false);
 }
});
test('ready unknown and synthetic context remains visible and is not rewritten',()=>{
 const plan=guide(),facts={media:true,kind:'video',preview:true,unknown:true,synthetic:true};
 const before=JSON.stringify(facts),next=plan('red_line',facts);
 assert.match(next.body,/withheld/);assert.match(next.body,/synthetic test/);
 assert.equal(next.activate,false);assert.equal(JSON.stringify(facts),before);
});
test('specific setup targets remain the next step instead of an implicit submission',()=>{
 const plan=guide();
 for(const target of ['incident-limit','incident-intervals','incident-end','incident-zone-rect','synthetic-apply']){
  const next=plan('speed_review',{media:true,kind:'video',preview:true,setupError:'Correct this field',setupTarget:target});
  assert.equal(next.target,target);assert.equal(next.stage,2);assert.equal(next.activate,false);
 }
});
test('showing a review target opens its disclosure but does not check it',()=>{
 const calls=[],details={tagName:'DETAILS',open:false,parentElement:null};
 const control={parentElement:details,tagName:'INPUT',disabled:false,scrollIntoView(){calls.push('scroll');},focus(){calls.push('focus');},click(){throw Error('Consent was clicked');}};
 const c=vm.createContext({$:()=>control});vm.runInContext(source,c);
 c.revealGuidedControl('feature-context-confirmed',false);
 assert.equal(details.open,true);assert.deepEqual(calls,['scroll','focus']);
});
test('completed and in-flight image work guides to the actual result and progress area',()=>{
 const plan=guide(),f={media:true,kind:'image',preview:true};
 assert.equal(plan('lab',{...f,running:true,foreground:true}).target,'ivp-demo-status');
 const result=plan('lab',{...f,completed:true,resultTarget:'ivp-demo-result'});
 assert.equal(result.stage,4);assert.equal(result.target,'ivp-demo-result');assert.equal(result.activate,false);
});
test('task selection survives input before change and navigates exactly once',()=>{
 const nodes=new Map(),listeners={},calls=[];
 const get=id=>{if(!nodes.has(id))nodes.set(id,{value:'',innerHTML:''});return nodes.get(id);};
 const c=vm.createContext({$:get,workbenchTask:'lab',escapeHTML:String,document:{addEventListener(name,fn){listeners[name]=fn;}}});
 vm.runInContext(source,c);
 c.updateGuidedUX=()=>{get('guided-goal').value=c.workbenchTask;};
 c.navigateTool=goal=>{calls.push(goal);c.workbenchTask=goal;c.updateGuidedUX();};
 c.setupGuidedUX();
 const goal=get('guided-goal');goal.value='zebra';goal.oninput();
 listeners.input({target:{id:'guided-goal'}});goal.onchange();listeners.change();
 assert.equal(goal.value,'zebra');assert.equal(c.workbenchTask,'zebra');assert.deepEqual(calls,['zebra']);
});
test('unreadable-image errors lead to source selection instead of unrelated review setup',()=>{
 const c=vm.createContext({});vm.runInContext(source,c);
 for(const message of ['Cannot read this image. Use a valid JPG or PNG.','Could not open uploaded file broken.png: Cannot read this image.']){
  const advice=c.guidedErrorAdvice(message);assert.equal(advice.category,'Could not read this input');assert.equal(advice.target,'source-picker');
 }
});
test('View results opens the current feature result rather than another feature in the global job card',()=>{
 const nodes=new Map(),opened=[];
 const get=id=>{if(!nodes.has(id))nodes.set(id,{setAttribute(){}});return nodes.get(id);};
 const c=vm.createContext({$:get,ivpPage:()=>true,escapeHTML:String,workbenchTask:'helmet',workspaceRestoring:false,state:{media:{name:'same-image.png'},job:'other-feature-run'},openRun:id=>opened.push(id)});
 vm.runInContext(source,c);
 c.guidedFacts=()=>({media:true,kind:'image',preview:true,completed:true,runId:'helmet-result'});
 c.updateGuidedUX();get('guided-next').onclick();get('guided-preview-next').onclick();
 assert.deepEqual(opened,['helmet-result','helmet-result']);
});
