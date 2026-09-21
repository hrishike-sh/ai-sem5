const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const test=require('node:test');
const assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
function harness(kind='image',stamp=0){
 const els=new Map();const $=id=>{if(!els.has(id))els.set(id,{disabled:false,style:{},classList:{remove(){}},setAttribute(){},scrollIntoView(){}});return els.get(id);};
 const state={media:{id:'m',source_revision:'r',kind},preview:{timestamp:stamp},pending:false,rendering:false};
 const requests=[],notices=[];
 const c=vm.createContext({state,$,workspaceRestoring:false,copy:v=>JSON.parse(JSON.stringify(v)),escapeHTML:String,
   selectingTarget:false,calibrating:false,selectStage(){},schedule(){},showJob(){},notify:(...a)=>notices.push(a),
   api:async(route,data)=>{requests.push({route,data});return {id:'fixture-job'};}});
 vm.runInContext(source.slice(source.indexOf('// A reviewed crossing band')),c);
 c.setupZebra();
 function mark(){ $('zebra-mark').onclick();for(const p of [[.2,.3],[.8,.3],[.8,.6],[.2,.6]])c.addZebraPoint(p); }
 return {c,$,state,requests,notices,mark};
}
test('four corners alone cannot launch until explicit band review',async()=>{
 const h=harness();h.mark();assert.equal(h.$('zebra-frame').disabled,true);
 await h.c.startZebra();assert.equal(h.requests.length,0);
 h.$('zebra-review').onclick();assert.equal(h.$('zebra-frame').disabled,false);
 await h.c.startZebra();assert.equal(h.requests[0].route,'/api/zebra');
 const d=h.requests[0].data;assert.equal(d.region.reviewed,true);assert.equal(d.source_revision,'r');
 assert.equal(d.region.semantic,'zebra_crossing');assert.equal(d.timestamp,0);
 assert.equal('calibration' in d,false);
});
test('changed source revision or displayed frame cannot reuse a reviewed band',async()=>{
 const h=harness('video',2);h.mark();h.$('zebra-review').onclick();
 h.state.media.source_revision='new';await h.c.startZebra();assert.equal(h.requests.length,0);
 h.state.media.source_revision='r';h.state.preview.timestamp=3;await h.c.startZebra();assert.equal(h.requests.length,0);
 h.c.zebraActions();assert.equal(h.$('zebra-overlay').style.display,'none');
});
test('video occupancy requires first frame and never borrows speed/rule geometry',async()=>{
 const h=harness('video',2);h.mark();h.$('zebra-review').onclick();await h.c.startZebra(true);
 assert.equal(h.requests.length,0);h.state.preview.timestamp=0;h.mark();h.$('zebra-review').onclick();
 await h.c.startZebra(true);const d=h.requests[0].data;
 assert.equal(h.requests[0].route,'/api/motion');assert.equal(d.calibration,null);assert.equal(d.incidents,null);
 assert.equal(d.tracking_mode,'vehicles');assert.equal(d.zebra.reference_seconds,0);
});
test('clear review removes geometry and disables both processing buttons',()=>{
 const h=harness();h.mark();h.$('zebra-review').onclick();h.c.clearZebra();
 assert.equal(h.$('zebra-frame').disabled,true);assert.equal(h.$('zebra-video').disabled,true);
 assert.equal(h.$('zebra-overlay').innerHTML,'');
});
test('focused IVP route permits an explicit detector crossing job with auth and source binding',async()=>{
 let fetched=false;const c=vm.createContext({ivpPage:()=>true,fetch:async(_path,options)=>{fetched=options;return {ok:true,json:async()=>({})};},token:'unused'});
 vm.runInContext(source.slice(source.indexOf('async function performApi('),source.indexOf('let apiQueue')),c);
 await c.performApi('/api/zebra',{media:'m',source_revision:'r'});assert.equal(fetched.headers['X-Traffic-Token'],'unused');assert.deepEqual(JSON.parse(fetched.body),{media:'m',source_revision:'r'});
});
