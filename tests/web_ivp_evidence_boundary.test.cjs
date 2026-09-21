const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
function requests(page='ivp'){
 const body={dataset:{appPage:page}},calls=[];
 const context=vm.createContext({document:{body},token:'test',fetch:async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>({saved:true})};}});
 vm.runInContext(source.slice(source.indexOf('function ivpPage('),source.indexOf('function workbenchStorageKey(')),context);
 vm.runInContext(source.slice(source.indexOf('async function performApi('),source.indexOf('function payload(')),context);
 return {context,body,calls};
}

test('IVP saved evidence cannot start either a model-enabled or offline review but keeps saved reads and full-app behaviour',async()=>{
 const h=requests();
 for(const use_model of [true,false])await assert.rejects(h.context.performApi('/api/evidence-review',{source_run:'saved',use_model}),/full application/i);
 assert.equal(h.calls.length,0);
 await h.context.performApi('/api/evidence-review?source=saved');await h.context.performApi('/api/evidence-review?id=saved&cursor=next');
 assert.equal(h.calls.length,2);assert.ok(h.calls.every(c=>!c.options.method));
 h.body.dataset.appPage='full';await h.context.performApi('/api/evidence-review',{source_run:'saved',use_model:true});assert.equal(h.calls.length,3);
});

test('a queued saved-evidence model request is rejected before fetch after the route becomes IVP',async()=>{
 const h=requests('full');let release;const firstResponse=new Promise(ok=>{release=ok;});
 h.context.fetch=async url=>{h.calls.push(url);if(url==='/api/preview')await firstResponse;return {ok:true,json:async()=>({})};};
 const first=h.context.api('/api/preview',{});await Promise.resolve();
 const model=h.context.api('/api/evidence-review',{source_run:'saved',use_model:true});h.body.dataset.appPage='ivp';release();await first;
 await assert.rejects(model,/full application/i);assert.deepEqual(h.calls,['/api/preview']);
});

test('saved evidence renders a read-only IVP notice and marks every new-review/model control as full-app-only',()=>{
 const context=vm.createContext({escapeHTML:String});
 vm.runInContext(source.slice(source.indexOf('function evidencePanel('),source.indexOf('function evidenceCitations(')),context);
 const html=context.evidencePanel({id:'saved',status:{state:'complete'},report:{motion:{incidents:{enabled:true}}}});
 assert.match(html,/class="evidence-model" data-full-app-only/);
 assert.match(html,/class="quiet evidence-build" data-full-app-only/);
 assert.match(html,/class="hint ivp-page-only"[^>]*>Saved evidence is read-only here/);
 assert.match(html,/class="evidence-content"/);assert.match(html,/class="evidence-navigation"/);
});
