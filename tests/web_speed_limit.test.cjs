const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/ivp_features.js'),'utf8');
function setup(){
 const nodes=new Map(),events={},node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:false,hidden:false,disabled:false,textContent:'',addEventListener(name,fn){this[name]=fn;}});return nodes.get(id);};
 node('motion-mode').value='vehicles';node('motion-view').value='tracks';node('motion-axis').value='horizontal';node('motion-position').value='65';
 let calibration=null;
 const c=vm.createContext({$:node,workbenchTask:'tracking',targetRectangle:null,ivpPage:()=>true,document:{addEventListener(name,fn){events[name]=fn;}},
  state:{config:{speed_display_v1:true},preview:{timestamp:0},media:{id:'media',source_revision:'r1',kind:'video',width:1280,height:720},settings:{max_width:320}},speedCalibration:()=>calibration,escapeHTML:String,api(){throw Error('No job may start during setup');}});
 vm.runInContext(source,c);c.updateFeatureUnits=()=>{};c.updateGuidedUX=()=>{};c.setupSpeedLimit();
 const enable=(limit='30')=>{node('speed-limit-value').value=limit;node('speed-limit-enabled').checked=true;node('speed-limit-enabled').change();};
 return {c,node,events,enable,setCalibration:value=>{calibration=value;}};
}
test('speed display starts off and permits an uncalibrated video without incident settings',()=>{
 const {c,node}=setup();c.syncSpeedLimit();const r=c.scopedTrafficRequest(c.state.media);assert.equal(r.calibration,null);assert.equal(r.incidents,null);assert.equal(r.speed_display.enabled,false);assert.equal(r.speed_display.unit,'px/s');assert.equal(r.speed_display.limit,null);assert.equal(node('speed-limit-enabled').checked,false);
});
test('enabled display sends a positive limit and captures the source revision without hidden geometry',()=>{
 const {c,enable,node}=setup();node('incident-enabled').checked=true;node('incident-limit').value='60';enable('30.01');const r=c.scopedTrafficRequest(c.state.media);assert.equal(r.speed_display.limit,30.01);assert.equal(r.source_revision,'r1');assert.equal(r.target_rectangle,null);assert.equal(r.rectangle,null);assert.equal(r.incidents,null);
});
for(const value of ['', '0','-1','NaN','Infinity'])test('invalid display limit '+JSON.stringify(value)+' stays entered and cannot submit',()=>{
 const {c,enable,node}=setup();enable(value);assert.throws(()=>c.scopedTrafficRequest(c.state.media),/positive finite/);assert.equal(node('speed-limit-value').value,value);
});
test('disabling preserves normal display and does not convert or submit the entered number',()=>{
 const {c,node,enable}=setup();enable();node('speed-limit-enabled').checked=false;node('speed-limit-enabled').change();assert.equal(c.scopedTrafficRequest(c.state.media).speed_display.limit,null);assert.equal(node('speed-limit-value').value,'30');
});
test('unit and calibration changes reset the filter rather than reinterpret its number',()=>{
 const {c,node,enable,setCalibration}=setup();enable();setCalibration({points:[[0,0],[1,0],[1,1],[0,1]],width_m:16,length_m:40});c.syncSpeedLimit();assert.equal(node('speed-limit-enabled').checked,false);assert.equal(node('speed-limit-value').value,'');assert.match(node('speed-limit-status').textContent,/no conversion/);enable('18');assert.equal(c.scopedTrafficRequest(c.state.media).speed_display.unit,'km/h');setCalibration(null);c.syncSpeedLimit();assert.equal(node('speed-limit-value').value,'');assert.match(node('speed-limit-unit').textContent,/px\/s/);
});
test('source revision, source resolution and tracking mode changes invalidate the prior display binding',()=>{
 for(const change of [c=>c.state.media.source_revision='r2',c=>c.state.media.width=640,(c,node)=>node('motion-mode').value='ivp']){
  const {c,node,enable}=setup();enable();change(c,node);assert.throws(()=>c.scopedTrafficRequest(c.state.media),/setup changed/);c.syncSpeedLimit();assert.equal(node('speed-limit-enabled').checked,false);assert.equal(node('speed-limit-value').value,'');
 }
});
test('preview resolution never relabels analysis pixels or rescales the limit',()=>{
 const {c,node,enable}=setup();enable('30');c.state.settings.max_width=1920;c.syncSpeedLimit();assert.equal(c.scopedTrafficRequest(c.state.media).speed_display.limit,30);assert.match(node('speed-limit-space').textContent,/at most 960 px/);assert.match(node('speed-limit-unit').textContent,/analysis px\/s/);
});
test('old service disables the new controls with an exact reason and cannot silently submit a checked filter',()=>{
 const {c,node,enable}=setup();c.state.config={};c.syncSpeedLimit();assert.equal(node('speed-limit-enabled').disabled,true);assert.equal(node('speed-limit-value').disabled,true);assert.equal(node('motion-run').disabled,false);assert.match(node('speed-limit-status').textContent,/Restart the existing local app/);assert.equal('speed_display' in c.scopedTrafficRequest(c.state.media),false);enable();assert.throws(()=>c.scopedTrafficRequest(c.state.media),/does not support/);
});
test('classical mode is labelled regions and result text preserves the exact units, limit and recorded dimensions',()=>{
 const {c,node}=setup();node('motion-mode').value='ivp';c.syncSpeedLimit();assert.match(node('speed-limit-space').textContent,/regions, not cars/);
 const s=c.speedDisplaySummary({enabled:true,limit:30.01,unit:'px/s',scope:'moving texture regions',analysis_frame_size:[960,540],coordinate_space:'analysis pixels',sample_counts:{'strictly above limit':5}});assert.match(s,/strictly above 30.01 px\/s/);assert.match(s,/960 × 540 px/);assert.match(s,/5 above-limit track samples/);assert.match(s,/not a violation/);
});
test('filter settings and tracking mode participate in saved-result matching',()=>{
 const {c,enable,node}=setup();const before=c.featureSignature('speed_review');enable('30');const after=c.featureSignature('speed_review');assert.notEqual(before,after);node('motion-mode').value='ivp';assert.notEqual(c.featureSignature('speed_review'),after);
});
test('calibrated candidate review remains available with filter off on the old service',()=>{
 const {c,node,setCalibration,enable}=setup();c.workbenchTask='speed_review';c.state.config={};
 const calibration={media_id:'media',source_revision:'r1',reference_frame:0,points:[[0,0],[1,0],[1,1],[0,1]],width_m:16,length_m:40};setCalibration(calibration);
 for(const id of ['incident-enabled','feature-context-confirmed','incident-calibration-confirmed','incident-real-time'])node(id).checked=true;
 node('incident-condition').value='observe';node('incident-limit').value='35';
 const original=c.scopedTrafficRequest(c.state.media);assert.equal(original.calibration,calibration);assert.equal(original.incidents.speed_limit_kmh,35);assert.equal(original.incidents.enabled,true);assert.equal('speed_display' in original,false);
 enable('18');assert.throws(()=>c.scopedTrafficRequest(c.state.media),/does not support/);
 node('speed-limit-enabled').checked=false;node('speed-limit-enabled').change();assert.equal(JSON.stringify(c.scopedTrafficRequest(c.state.media)),JSON.stringify(original));
 c.state.config.speed_display_v1=true;enable('18');const both=c.scopedTrafficRequest(c.state.media);assert.equal(both.incidents.speed_limit_kmh,35);assert.equal(both.speed_display.limit,18);assert.equal(both.speed_display.unit,'km/h');
});
