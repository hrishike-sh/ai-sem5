const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

function harness(mode='ivp',calibration=null){
 const media={id:'same-video-path',source_revision:'selected-old-revision',name:'selected.mp4',kind:'video'};
 const state={media,region:null,job:'existing-job'};
 const elements={
  'motion-use-roi':{checked:false},'motion-axis':{value:'horizontal'},'motion-position':{value:'65'},
  'motion-mode':{value:mode},'motion-view':{value:'tracks'},'active-job':{scrollIntoView(){}}
 };
 const requests=[],jobs=[],notices=[];
 let resolve,reject;
 const pending=new Promise((ok,fail)=>{resolve=ok;reject=fail;});
 const context=vm.createContext({state,startingMotion:false,targetRectangle:null,$:id=>elements[id],
  speedCalibration:()=>calibration,incidentConfig:()=>null,motionActions(){},
  api:(route,data)=>{requests.push({route,data:JSON.parse(JSON.stringify(data))});return pending;},
  showJob:run=>{jobs.push(run.id);state.job=run.id;},notify:(text,error)=>notices.push({text,error})});
 vm.runInContext(source.slice(source.indexOf('async function startMotion('),source.indexOf('let selectingTarget=')),context);
 return {context,state,media,requests,jobs,notices,resolve,reject};
}

for(const mode of ['ivp','vehicles'])test(`uncalibrated ${mode} Tracking sends the selected source revision`,async()=>{
 const h=harness(mode),before=JSON.stringify(h.state.media);
 const pending=h.context.startMotion();
 h.resolve({id:'accepted-job'});await pending;
 assert.deepEqual(h.requests,[{route:'/api/motion',data:{media:'same-video-path',
  source_revision:'selected-old-revision',axis:'horizontal',position:.65,rectangle:null,
  calibration:null,target_rectangle:null,tracking_mode:mode,output_view:'tracks',incidents:null}}]);
 assert.equal(JSON.stringify(h.state.media),before);
 assert.deepEqual(h.jobs,['accepted-job']);
 assert.equal(h.context.startingMotion,false);
});

test('calibrated Tracking also pins the top-level browser revision without changing geometry',async()=>{
 const calibration={media_id:'same-video-path',source_revision:'selected-old-revision',reference_frame:0,
  points:[[0,0],[1,0],[1,1],[0,1]],width_m:10,length_m:10};
 const before=JSON.stringify(calibration),h=harness('ivp',calibration);
 const pending=h.context.startMotion();h.resolve({id:'accepted-job'});await pending;
 assert.equal(h.requests[0].data.source_revision,'selected-old-revision');
 assert.deepEqual(h.requests[0].data.calibration,calibration);
 assert.equal(JSON.stringify(calibration),before);
});

test('a queued Tracking submission keeps the captured source identity/version when the workspace changes',async()=>{
 const h=harness(),pending=h.context.startMotion();
 h.state.media={id:'new-video',source_revision:'new-revision',name:'new.mp4',kind:'video'};
 h.resolve({id:'first-source-job'});await pending;
 assert.equal(h.requests[0].data.media,'same-video-path');
 assert.equal(h.requests[0].data.source_revision,'selected-old-revision');
 assert.equal(h.state.media.id,'new-video');
});

test('a server source-change rejection keeps the previous job and allows retry after reopening',async()=>{
 const h=harness(),pending=h.context.startMotion();
 h.reject(Error('The source changed since it was opened. Reopen the footage before continuing.'));
 await pending;
 assert.equal(h.state.job,'existing-job');assert.deepEqual(h.jobs,[]);
 assert.match(h.notices[0].text,/source changed.*Reopen/);assert.equal(h.notices[0].error,true);
 assert.equal(h.context.startingMotion,false);
});
