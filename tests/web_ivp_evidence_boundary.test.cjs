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

test('AI saved evidence allows explicit review requests and reads',async()=>{
 const h=requests('full');
 for(const use_model of [true,false])await h.context.performApi('/api/evidence-review',{source_run:'saved',use_model});
 await h.context.performApi('/api/evidence-review?source=saved');
 assert.equal(h.calls.length,3);assert.equal(h.calls[2].options.method,undefined);
});
