const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
function harness(){
 const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,{value:id==='ivp-goal'?'regions':'',hidden:false});return elements.get(id);};
 const context=vm.createContext({$:get,state:{preview:{},pending:false,rendering:false},workspaceRestoring:false,workbenchTask:'lab',
  ivpPage:()=>true,payload:()=>({media:'source',source_revision:'revision',settings:{gamma:1},timestamp:0}),masks:()=>['Edges'],
  escapeHTML:value=>String(value).replaceAll('<','&lt;'),links:()=>'',notify(){},renderDemonstration(){}});
 vm.runInContext(source.slice(source.indexOf('let demoTicket='),source.indexOf('\ninit();')),context);
 return {context,get};
}
test('selected image goal uses the bound export route without AI or a video job',async()=>{
 const h=harness(),calls=[];h.context.api=async(route,data)=>{calls.push({route,data});return {id:'saved'};};h.context.renderDemonstration=()=>{};
 await h.context.generateDemonstration();assert.equal(calls.length,1);assert.equal(calls[0].route,'/api/export');
 assert.equal(calls[0].data.source_revision,'revision');assert.equal(calls[0].data.ivp_goal,'regions');assert.equal(calls[0].data.ai,undefined);
});
test('superseded source/settings results stay saved without replacing the current demonstration',async()=>{
 const h=harness();let resolve,shown=0;h.context.api=()=>new Promise(ok=>{resolve=ok;});h.context.renderDemonstration=()=>shown++;
 const pending=h.context.generateDemonstration();h.context.clearDemonstration();resolve({id:'earlier'});await pending;
 assert.equal(shown,0);assert.equal(h.get('ivp-demo-result').hidden,true);assert.equal(h.get('ivp-generate').disabled,false);
});
test('an export response cannot open the focused result after navigation away',async()=>{
 const h=harness();let resolve,shown=0;h.context.api=()=>new Promise(ok=>{resolve=ok;});h.context.renderDemonstration=()=>shown++;
 const pending=h.context.generateDemonstration();h.context.ivpPage=()=>false;resolve({id:'earlier'});await pending;assert.equal(shown,0);
});
test('rendered step names actual input parameters and measured effect without an accuracy claim',()=>{
 const h=harness();const html=h.context.demonstrationStepHTML({urls:{before:'before.png',after:'after.png'},metric_notice:'Pixel change is not accuracy.'},
  {title:'Gamma correction',purpose:'Changed intensity with gamma0.85.',inputs:['Colour input'],parameters:{gamma:.85},before:'before',after:'after',effect:{changed_pixels:8,total_pixels:10,changed_percent:80}});
 assert.match(html,/Gamma correction/);assert.match(html,/gamma/);assert.match(html,/0.85/);assert.match(html,/before.png/);assert.match(html,/after.png/);assert.match(html,/8 of 10 pixels differ/);assert.match(html,/not accuracy/);
 assert.doesNotMatch(html,/CLAHE|FFT|Sobel/);
});
