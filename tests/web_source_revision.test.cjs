const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

test('IVP requests pin the selected media revision and keep settings/timestamp intact',()=>{
 const state={media:{id:'same-path',source_revision:'selected-version'},settings:{gamma:1.2},mode:'automatic'};
 const context=vm.createContext({state,$:()=>({value:'4.25'}),copy:x=>JSON.parse(JSON.stringify(x))});
 vm.runInContext(source.slice(source.indexOf('function payload('),source.indexOf('function masks(')),context);
 const sent=JSON.parse(JSON.stringify(context.payload()));
 assert.deepEqual(sent,{media:'same-path',source_revision:'selected-version',timestamp:4.25,settings:{gamma:1.2},mode:'automatic'});
 assert.equal(state.media.source_revision,'selected-version');
});
