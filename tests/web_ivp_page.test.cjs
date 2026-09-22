const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

function harness(pathname='/',persisted={}){
 const saved=new Map(Object.entries(persisted)),elements=new Map(),events={},requests=[];
 const calls={sceneRestore:0,library:0,pushes:0,closed:0};
 const get=id=>{if(!elements.has(id))elements.set(id,{value:'0',hidden:false,dataset:{},setAttribute(){},removeAttribute(){},focus(){},insertAdjacentHTML(){},querySelector(){return {pause(){}};},close(){calls.closed++;},scrollIntoView(){}});return elements.get(id);};
 const body={dataset:{appPage:pathname.startsWith('/ivp')?'ivp':'full'},classList:{remove(){},toggle(){}}};
 const document={body,title:'',querySelectorAll(){return [];}};
 const window={location:{pathname},history:{pushState(_s,_t,p){window.location.pathname=p;calls.pushes++;}},addEventListener(name,fn){events[name]=fn;}};
 const state={media:{id:'kept',source_revision:'revision',kind:'video'},preview:{timestamp:2.9},settings:{gamma:1.2},region:[.1,.2,.8,.9],job:'existing-job',review:{id:'active-review',cancelling:true},view:'stages',area:'ivp'};
 const calibration=[[.1,.2],[.8,.2],[.9,.9],[.1,.9]],before=JSON.stringify({media:state.media,preview:state.preview,settings:state.settings,region:state.region,job:state.job,review:state.review,calibration});
 const context=vm.createContext({state,document,window,calibrationPoints:calibration,$:get,workspaceRestoring:false,
  localStorage:{getItem:k=>saved.get(k)||null,setItem(k,v){saved.set(k,v);}},
  setArea(area){state.area=area;get('workspace').hidden=false;get('library').hidden=true;},paint(){},paintWorkbench(){},actions(){},
  loadRuns(){calls.library++;},restoreSceneReviews:async()=>{calls.sceneRestore++;},
  api(){throw Error('Navigation must not start processing');},
  fetch:async(route,options)=>{requests.push({route,options});return {ok:true,json:async()=>({ok:true})};},token:'test-token'});
 vm.runInContext(source.slice(source.indexOf('const workbenchTasks='),source.indexOf('function paintWorkbench(')),context);
 return {context,state,get,body,document,window,events,saved,requests,calls,unchanged(){assert.equal(JSON.stringify({media:state.media,preview:state.preview,settings:state.settings,region:state.region,job:state.job,review:state.review,calibration}),before);}};
}

test('AI routes default to detection and restore only AI tool preferences',()=>{
 for(const pathname of ['/','/ai','/ai/']){
  const h=harness(pathname,{'traffic-lab-tool':'tracking','ai-sem5-tool':'lab'});
  h.context.applyPageRoute();h.unchanged();
  assert.equal(h.body.dataset.workbench,'detection');assert.equal(h.state.area,'ai');assert.match(h.document.title,/AI Sem 5/);
 }
 const h=harness('/ai',{'ai-sem5-tool':'overview'});h.context.applyPageRoute();assert.equal(h.body.dataset.workbench,'overview');
});

test('AI navigation does not open the removed IVP page',()=>{
 const h=harness('/');h.context.setupPageNavigation();h.context.navigatePage('/ivp');
 assert.equal(h.calls.pushes,0);h.context.navigatePage('/ai');assert.equal(h.window.location.pathname,'/ai');h.unchanged();
});
