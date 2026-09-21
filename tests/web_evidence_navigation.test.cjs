const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
const flush=async()=>{for(let i=0;i<5;i++)await Promise.resolve();};
function result(start=1,filter='all'){
 return {id:'saved',state:'complete',message:'Ready',artifact_integrity:{state:'verified'},
  review:{artifact:'offline',generated_at:'fixed',pagination:{start,end:start+19,filtered_total:45,status_filter:filter,
   current_cursor:start===1?null:'page-'+start,next_cursor:start<41?'page-'+(start+20):null,previous_cursor:start>1?'page-'+(start-20):null}}};
}
function controller(api){
 const elements=Object.fromEntries(['build','use-model','status','content','integrity','navigation','filter','prev','next','page-label'].map(name=>['.evidence-'+name,{}]));
 elements['.evidence-use-model'].checked=false;
 const panel={isConnected:true,dataset:{evidenceRun:'source'},querySelector:key=>elements[key]};
 const context=vm.createContext({api,evidenceReviewSummary:r=>`${r.review.pagination.status_filter}:${r.review.pagination.start}`,
  clearTimeout:()=>{},setTimeout:()=>1});
 vm.runInContext(source.slice(source.indexOf('function bindEvidencePanel('),source.indexOf('function stopEvidenceSummary(')),context);
 context.bindEvidencePanel(panel);
 return elements;
}
test('later pages and status filters request saved facts without exporting or using a model',async()=>{
 const calls=[];
 const e=controller(async(url,data)=>{calls.push({url,data});if(url.includes('status=candidate'))return result(1,'candidate');if(url.includes('cursor=page-21'))return result(21);return result();});
 await flush();assert.equal(e['.evidence-prev'].disabled,true);
 await e['.evidence-next'].onclick();assert.equal(e['.evidence-content'].innerHTML,'all:21');
 assert.equal(e['.evidence-page-label'].textContent,'21–40 of 45');
 e['.evidence-filter'].value='candidate';await e['.evidence-filter'].onchange();
 assert.equal(e['.evidence-content'].innerHTML,'candidate:1');
 assert.ok(calls.every(c=>c.data===undefined));
 assert.match(calls.at(-1).url,/status=candidate/);assert.doesNotMatch(calls.at(-1).url,/cursor=/);
});
test('a late old page cannot replace a newly selected status filter',async()=>{
 let release;
 const e=controller(async url=>url.includes('cursor=page-21')?new Promise(r=>{release=r;}):url.includes('status=withheld')?result(1,'withheld'):result());
 await flush();const old=e['.evidence-next'].onclick();
 e['.evidence-filter'].value='withheld';await e['.evidence-filter'].onchange();
 release(result(21));await old;
 assert.equal(e['.evidence-content'].innerHTML,'withheld:1');
 assert.equal(e['.evidence-filter'].value,'withheld');
});
test('an old page cannot replace a newly built review and tampered pages clear facts',async()=>{
 let release;
 const e=controller(async(url,data)=>data?{...result(),id:'new'}:url.includes('cursor=page-21')?new Promise(r=>{release=r;}):result());
 await flush();const old=e['.evidence-next'].onclick();
 await e['.evidence-build'].onclick();release(result(21));await old;
 assert.equal(e['.evidence-content'].innerHTML,'all:1');
 const tampered=controller(async url=>url.includes('cursor=')?{id:'saved',state:'complete',artifact_integrity:{state:'changed',message:'Changed; rebuild'}}:result());
 await flush();await tampered['.evidence-next'].onclick();
 assert.equal(tampered['.evidence-content'].innerHTML,'');assert.equal(tampered['.evidence-navigation'].hidden,true);
 assert.equal(tampered['.evidence-build'].disabled,false);
});
test('an outdated cursor retries the first page once while connection failures retain the valid page',async()=>{
 let calls=0;
 const e=controller(async url=>{calls++;if(url.includes('cursor=')){const error=Error('Outdated page');error.status=400;throw error;}return result();});
 await flush();await e['.evidence-next'].onclick();
 assert.equal(calls,3);assert.equal(e['.evidence-content'].innerHTML,'all:1');
 const offline=controller(async url=>{if(url.includes('status=candidate'))throw Error('Connection lost');return result();});
 await flush();offline['.evidence-filter'].value='candidate';await offline['.evidence-filter'].onchange();
 assert.equal(offline['.evidence-content'].innerHTML,'all:1');assert.equal(offline['.evidence-filter'].value,'all');
 assert.equal(offline['.evidence-status'].textContent,'Connection lost');
});
test('rendered late cards retain source position and disclose missing thumbnails',()=>{
 const context=vm.createContext({escapeHTML:s=>String(s),links:()=>'',stopEvidenceSummary:()=>''});
 vm.runInContext(source.slice(source.indexOf('function evidencePanel('),source.indexOf('function bindEvidencePanel(')),context);
 const html=context.evidenceReviewSummary({downloads:[],review:{summary:{event_count:45,statuses:{candidate:1},synthetic_events:45,unknown_provenance_events:0},warnings:[],limitations:[],source:{files:[]},
  pagination:{start:1,end:1,filtered_total:1},events:[{source_position:25,thumbnail_notice:'No exported thumbnail available',facts:[],warnings:[],review_questions:[],evidence_ref:{file:'events.jsonl',line:25},
   event:{event_id:'synthetic-25',status:'candidate',type:'line_crossing',timestamp_seconds:2.4,synthetic:true,measurements:{}}}]}});
 assert.match(html,/source assessment 25/);assert.match(html,/SYNTHETIC/);assert.match(html,/No exported thumbnail/);assert.match(html,/Showing 1–1 of 1 matching assessments/);
});
test('a completed report without its video still permits saved-evidence inspection',()=>{
 let touched=false;
 const context=vm.createContext({lastMotionRun:'previous-good',localStorage:{setItem:()=>{}},
  $:()=>{touched=true;throw Error('A missing video must not replace the workspace preview');}});
 vm.runInContext(source.slice(source.indexOf('function showMotionResult('),source.indexOf('\ninit();')),context);
 assert.doesNotThrow(()=>context.showMotionResult({id:'saved',status:{state:'complete'},report:{motion:{}},preview:null,downloads:[]}));
 assert.equal(touched,false);assert.equal(context.lastMotionRun,'previous-good');
});
