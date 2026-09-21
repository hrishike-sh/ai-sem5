const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/ivp_features.js'),'utf8');
function setup(feature='red_line'){
 const nodes=new Map(),node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:false,disabled:false,textContent:'',classList:{add(){},remove(){}},scrollIntoView(){},options:[{},{},{}]});return nodes.get(id);};
 for(const [id,value] of Object.entries({'motion-axis':'horizontal','motion-position':'65','incident-start':'15','incident-end':'85','incident-zone-rect':'15, 65, 85, 78','incident-direction':'positive','incident-condition':'unknown','incident-intervals':'','incident-approach':'15,45,85,65','timestamp':'0'}))node(id).value=value;
 node('incident-line').checked=true;node('incident-zone').checked=true;
 const media={id:'source',source_revision:'revision-a',kind:'video',example:{synthetic:true,virtual_setup:{kind:'fixed_translation_fixture_v1',width_m:16,length_m:40,reference_frame:0,threshold_is_legal_limit:false,points:[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]}}};
 const c=vm.createContext({$:node,document:{body:{dataset:{redReview:'true'}}},ivpPage:()=>true,workbenchTask:feature,workspaceRestoring:false,state:{media,preview:{timestamp:0},rendering:false,pending:false},speedCalibration(){throw Error('Must not consult calibration');},sample(){throw Error('Unexpected source open');},api(){throw Error('Unexpected job');}});
 vm.runInContext(source,c);c.paintIncidents=()=>{};c.updateFeatureUnits=()=>{};
 const confirm=()=>{node('feature-context-confirmed').checked=true;c.redSetupChanged({target:{id:'feature-context-confirmed'}});};
 return {c,node,confirm};
}
test('two clicks create a diagonal finite line, preserve unknown, and require explicit confirmation',()=>{
 const {c,node,confirm}=setup();c.beginRedDrawing();c.addRedPoint([.2,.6]);c.addRedPoint([.8,.7]);
 assert.equal(node('incident-condition').value,'unknown');assert.equal(node('feature-context-confirmed').checked,false);
 assert.throws(()=>c.scopedTrafficRequest(c.state.media),/Confirm/);confirm();const r=c.scopedTrafficRequest(c.state.media);
 assert.deepEqual(JSON.parse(JSON.stringify(r.incidents.line)),[[.2,.6],[.8,.7]]);assert.equal(r.calibration,null);assert.equal(r.incidents.zone,null);assert.equal(r.incidents.condition,'unknown');assert.equal(r.incidents.synthetic,true);
});
test('area drawing normalizes opposite corners and needs explicit restriction and context review',()=>{
 const {c,node,confirm}=setup('restricted_red');c.beginRedDrawing();c.addRedPoint([.8,.8]);c.addRedPoint([.2,.4]);confirm();
 assert.throws(()=>c.scopedTrafficRequest(c.state.media),/restricted/);node('incident-restricted').checked=true;c.redSetupChanged({target:{id:'incident-restricted'}});assert.equal(node('feature-context-confirmed').checked,false);confirm();
 const r=c.scopedTrafficRequest(c.state.media);assert.deepEqual(JSON.parse(JSON.stringify(r.incidents.zone)),[.2,.4,.8,.8]);assert.equal(r.incidents.line,null);assert.equal(r.calibration,null);assert.equal(r.incidents.speed_limit_kmh,null);
});
for(const id of ['motion-axis','motion-position','incident-start','incident-end','incident-direction','incident-condition','incident-intervals','incident-synthetic'])test('editing '+id+' invalidates an earlier line confirmation',()=>{
 const {c,node,confirm}=setup();c.placeRedExample();confirm();assert.equal(node('incident-enabled').checked,true);c.redSetupChanged({target:{id}});assert.equal(node('feature-context-confirmed').checked,false);assert.equal(node('incident-enabled').checked,false);assert.throws(()=>c.featureIncidentConfig('red_line'),/Confirm/);confirm();assert.equal(c.featureIncidentConfig('red_line').enabled,true);
});
for(const id of ['incident-zone-rect','incident-condition','incident-intervals'])test('editing '+id+' invalidates zone applicability as well as the overall review',()=>{
 const {c,node,confirm}=setup('restricted_red');c.placeRedExample();node('incident-restricted').checked=true;confirm();c.redSetupChanged({target:{id}});assert.equal(node('incident-restricted').checked,false);assert.equal(node('feature-context-confirmed').checked,false);
});
test('red-only invalidation does not change required-stop confirmation',()=>{
 const {c,node,confirm}=setup();c.placeRedExample();confirm();c.workbenchTask='required_stop';c.redSetupChanged({target:{id:'incident-condition'}});assert.equal(node('feature-context-confirmed').checked,true);assert.equal(node('incident-enabled').checked,true);
});
test('source revision and feature changes cannot reuse drawing or a confirmation',()=>{
 const {c,confirm}=setup();c.placeRedExample();confirm();c.state.media.source_revision='revision-b';assert.throws(()=>c.featureIncidentConfig('red_line'),/Draw/);c.state.media.source_revision='revision-a';c.workbenchTask='restricted_red';assert.throws(()=>c.featureIncidentConfig('restricted_red'),/Draw/);c.clearRedReviewGeometry();assert.equal(c.redBound(),false);
});
test('drawing rejects too-short points, out-of-bounds points and nonzero or unfinished previews',()=>{
 const {c}=setup();c.beginRedDrawing();c.addRedPoint([-.1,.2]);assert.equal(vm.runInContext('redPoints.length',c),0);c.addRedPoint([.2,.2]);assert.throws(()=>c.addRedPoint([.2001,.2]),/close/);assert.equal(vm.runInContext('redPoints.length',c),1);c.state.preview.timestamp=1;assert.throws(()=>c.beginRedDrawing(),/first frame/);c.state.preview.timestamp=0;c.state.pending=true;assert.throws(()=>c.beginRedDrawing(),/wait/);
});
test('synthetic placement leaves consent and condition unknown and refuses unverified or real sources',()=>{
 const {c,node,confirm}=setup();c.placeRedExample();confirm();node('incident-condition').value='red';c.placeRedExample();assert.equal(node('incident-condition').value,'unknown');assert.equal(node('feature-context-confirmed').checked,false);assert.equal(node('incident-enabled').checked,false);
 c.state.media.example.synthetic=false;assert.equal(c.knownVirtualSetup(),null);assert.throws(()=>c.placeRedExample(),/catalog/);c.state.media.example.synthetic=true;c.state.media.example.virtual_setup.points[0]=[.2,.2];assert.equal(c.knownVirtualSetup(),null);assert.throws(()=>c.placeRedExample(),/catalog/);
});
test('virtual dimension note is metadata bound and does not apply calibration or consent',()=>{
 const {c,node}=setup();c.updateVirtualDimensions();assert.match(node('virtual-dimensions-note').textContent,/16 m width across × 40 m length along/);assert.equal(node('feature-context-confirmed').checked,false);c.state.media.source_revision=null;c.updateVirtualDimensions();assert.equal(node('virtual-dimensions-note').hidden,true);
});
test('red annotation distinguishes line/arrow from an area and carries no signal detection claim',()=>{
 const {c}=setup();const line=c.redAnnotation({line:[[.2,.6],[.8,.7]],direction:'both'}),zone=c.redAnnotation({zone:[.2,.4,.8,.8]});assert.match(line,/CROSS THIS LINE/);assert.match(line,/marker-start/);assert.match(zone,/ENTER THIS AREA/);assert.doesNotMatch(zone,/red-arrow-head/);
});
test('custom geometry changes invalidate the run signature without altering hidden speed state',()=>{
 const {c}=setup();c.placeRedExample();const a=c.featureSignature('red_line');c.beginRedDrawing();c.addRedPoint([.1,.3]);c.addRedPoint([.9,.5]);assert.notEqual(c.featureSignature('red_line'),a);
});
test('blank numeric endpoints are retained but cannot silently turn into a zero coordinate',()=>{
 const {c,node,confirm}=setup();c.placeRedExample();node('incident-start').value='';c.redSetupChanged({target:{id:'incident-start'}});confirm();assert.throws(()=>c.featureIncidentConfig('red_line'),e=>e.field==='incident-start');assert.equal(node('incident-start').value,'');node('incident-start').value='20';c.redSetupChanged({target:{id:'incident-start'}});confirm();assert.equal(c.featureIncidentConfig('red_line').line[0][0],.2);
});
test('bad area coordinates lead to their field before asking for confirmation and retain the typed input',()=>{
 const {c,node}=setup('restricted_red');c.placeRedExample();node('incident-zone-rect').value='15, 65, 10, 78';c.redSetupChanged({target:{id:'incident-zone-rect'}});assert.throws(()=>c.featureIncidentConfig('restricted_red'),e=>e.field==='incident-zone-rect');assert.equal(node('incident-zone-rect').value,'15, 65, 10, 78');assert.equal(node('feature-context-confirmed').checked,false);
});
