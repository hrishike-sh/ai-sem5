const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
const fragment=source.slice(source.indexOf('function aiAssessmentPanel('),source.indexOf('let sceneStarting='));
const escapeHTML=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
function result(start=1,verdict='cannot_assess',rule='all'){
 return {decisions:[],limits:'Recorded only',provenance_notice:'Unrecorded is unknown',integrity:'Current fingerprints, not authenticity',report_counts:'match',fingerprints:{},
  pagination:{start,end:Math.min(45,start+19),filtered_total:45,total:45,verdict,rule,has_previous:start>1,previous_cursor:start===21?null:'previous',next_cursor:start<41?'page-'+(start+20):null}};
}
function setup(api,render=true){
 const elements=Object.fromEntries(['load','verdict','rule','status','content','nav','prev','next','page'].map(n=>['.ai-assessment-'+n,{}]));
 elements['.ai-assessment-verdict'].value='cannot_assess';elements['.ai-assessment-rule'].value='all';
 const panel={isConnected:true,dataset:{aiRun:'saved'},querySelector:name=>elements[name]};
 const context=vm.createContext({api,escapeHTML});vm.runInContext(fragment,context);
 if(!render)context.aiAssessmentCards=v=>`${v.pagination.verdict}:${v.pagination.rule}:${v.pagination.start}`;
 context.bindAiAssessmentPanel(panel);return {elements,panel,context};
}
test('opening a result does not read all frame decisions; load uses GET and bounded navigation',async()=>{
 const calls=[];
 const {elements:e}=setup(async(url,data)=>{calls.push({url,data});return url.includes('cursor=page-21')?result(21):result();},false);
 assert.equal(calls.length,0);assert.equal(e['.ai-assessment-nav'].hidden,true);
 await e['.ai-assessment-load'].onclick();assert.equal(e['.ai-assessment-content'].innerHTML,'cannot_assess:all:1');
 await e['.ai-assessment-next'].onclick();assert.match(e['.ai-assessment-page'].textContent,/21–40 of 45/);
 await e['.ai-assessment-prev'].onclick();assert.equal(e['.ai-assessment-content'].innerHTML,'cannot_assess:all:1');
 assert.ok(calls.every(c=>c.data===undefined));assert.doesNotMatch(calls.at(-1).url,/cursor=/);
});
test('a late page cannot replace a newly selected verdict or rule',async()=>{
 let release;
 const {elements:e}=setup(async url=>url.includes('cursor=page-21')?new Promise(r=>{release=r;}):url.includes('rule=prohibited_turn')?result(1,'no_violation_observed','prohibited_turn'):result(),false);
 await e['.ai-assessment-load'].onclick();const pending=e['.ai-assessment-next'].onclick();
 e['.ai-assessment-verdict'].value='no_violation_observed';e['.ai-assessment-rule'].value='prohibited_turn';await e['.ai-assessment-rule'].onchange();
 release(result(21));await pending;assert.equal(e['.ai-assessment-content'].innerHTML,'no_violation_observed:prohibited_turn:1');
});
test('network errors retain a valid page but invalid sources clear it',async()=>{
 let invalid=false;
 const {elements:e}=setup(async url=>{if(url.includes('cursor=')){const error=Error('Read failed');if(invalid)error.status=400;throw error;}return result();},false);
 await e['.ai-assessment-load'].onclick();await e['.ai-assessment-next'].onclick();
 assert.equal(e['.ai-assessment-content'].innerHTML,'cannot_assess:all:1');assert.equal(e['.ai-assessment-nav'].hidden,false);
 invalid=true;await e['.ai-assessment-next'].onclick();assert.equal(e['.ai-assessment-content'].innerHTML,'');assert.equal(e['.ai-assessment-nav'].hidden,true);
 assert.equal(e['.ai-assessment-status'].textContent,'Read failed');
});
test('a detached result panel ignores late reads',async()=>{
 let release;const {elements:e,panel}=setup(()=>new Promise(r=>{release=r;}),false);
 const pending=e['.ai-assessment-load'].onclick();panel.isConnected=false;release(result());await pending;
 assert.equal(e['.ai-assessment-content'].innerHTML,undefined);
});
test('recorded reasons, provenance and raw evidence are escaped and missing fields stay unrecorded',()=>{
 const {context}=setup(async()=>result());
 const view={...result(),decisions:[{source_position:21,decision:{rule:'prohibited_turn',verdict:'cannot_assess',timestamp:0,reason:'<img src=x onerror=alert(1)>',evidence:{note:'</pre><script>attack</script>'}},frame_context:{frame:null,synthetic:null,provenance:'<svg onload=attack>',event_confirmation:{pending:[{reason:'</pre><script>trace</script>'}]}},source:{file:'detections.jsonl',line:2,pointer:'/decisions/5',line_sha256:'abc'}}]};
 const html=context.aiAssessmentCards(view);
 assert.doesNotMatch(html,/<img|<script|<svg/);assert.match(html,/&lt;img/);assert.match(html,/source assessment 21/i);
 assert.match(html,/detections.jsonl · line 2 · \/decisions\/5/);assert.match(html,/Recorded frame index: Unrecorded/);
 assert.match(html,/Current fingerprints, not authenticity/);
});
test('incomplete runs have no inspector and all complete AI runs expose it',()=>{
 const {context}=setup(async()=>result());
 assert.equal(context.aiAssessmentPanel({status:{state:'running'}}),'');
 assert.match(context.aiAssessmentPanel({id:'saved',status:{state:'complete'}}),/Inspect recorded AI assessments/);
});
