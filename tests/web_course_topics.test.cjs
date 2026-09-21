const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/ivp_features.js'),'utf8');
function harness(){const controls=new Map(),node=id=>controls.get(id),footer={innerHTML:''};controls.set('course-units',footer);controls.set('ivp-goal',{value:'regions'});controls.set('motion-mode',{value:'ivp'});const c=vm.createContext({$:node,workbenchTask:'lab',state:{settings:{saturation:1},media:{kind:'image'}},escapeHTML:s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')});vm.runInContext(source,c);return {c,controls,footer};}
const units=p=>Array.from(p.rows,r=>r.unit);

test('each image output lists only its own pipeline topics before a run',()=>{
 const {c}=harness();assert.deepEqual(units(c.courseFeatureProfile('lab',{goal:'regions'})),[1,2,4,6]);assert.deepEqual(units(c.courseFeatureProfile('lab',{goal:'edges'})),[1,2,5]);assert.deepEqual(units(c.courseFeatureProfile('lab',{goal:'colour',saturation:1})),[2]);assert.deepEqual(units(c.courseFeatureProfile('lab',{goal:'colour',saturation:1.2})),[2,7]);
 const html=c.featureUnitsHTML(null,'lab');assert.match(html,/Connected region map/);assert.match(html,/Otsu/);assert.doesNotMatch(html,/Canny|Helmet|Unit 3/);assert.match(html,/not a claim that a run has already executed/);assert.doesNotMatch(html,/course-run-evidence/);
});
test('road marking topics match the colour, local-background, mask and component path',()=>{
 const {c}=harness(),p=c.courseFeatureProfile('markings');assert.deepEqual(units(p),[2,4,6,7]);assert.match(p.rows[0].topic,/median/);assert.match(p.rows[3].topic,/HSV/);assert.match(p.rows[1].reference,/Lecture 11/);assert.match(p.extension,/do not establish/);assert.doesNotMatch(c.courseGuideHTML('markings',{}),/Canny|motion estimation|Otsu/);
});
test('tracking mode excludes discovery-only filters from the vehicle-track explanation',()=>{
 const {c}=harness();assert.deepEqual(units(c.courseFeatureProfile('tracking',{mode:'ivp'})),[1,2,4,6,8]);assert.deepEqual(units(c.courseFeatureProfile('tracking',{mode:'vehicles'})),[8]);assert.match(c.courseFeatureProfile('tracking',{mode:'ivp'}).extension,/block-matching.*not implemented/);
});
for(const feature of ['direction','red_line','restricted_red','speed_review','required_stop'])test(feature+' maps only temporal processing and explains its own project logic',()=>{
 const {c}=harness(),p=c.courseFeatureProfile(feature);assert.deepEqual(units(p),[8]);assert.match(p.rows[0].reference,/25 & 27/);assert.doesNotMatch(c.courseGuideHTML(feature,{}),/course-unit-badge">Unit [1-7]</);assert.match(p.extension,/project|Project/);
 if(feature==='red_line')assert.match(p.extension,/user-declared red timeline/);if(feature==='restricted_red')assert.match(p.extension,/not Unit 6/);if(feature==='speed_review')assert.match(p.extension,/not the DFT, DCT or Hadamard/);
});
test('crossing frame and video distinguish single-image geometry from temporal processing',()=>{
 const {c}=harness();assert.deepEqual(units(c.courseFeatureProfile('zebra',{kind:'image'})),[]);assert.deepEqual(units(c.courseFeatureProfile('zebra',{kind:'video'})),[8]);assert.match(c.courseFeatureProfile('zebra',{kind:'image'}).extension,/not image segmentation/);
});
for(const feature of ['helmet','detection','overview','library'])test(feature+' does not borrow image-lab units or claim AI recognition as classical IVP',()=>{
 const {c}=harness();assert.deepEqual(units(c.courseFeatureProfile(feature)),[]);const html=c.featureUnitsHTML(null,feature);assert.doesNotMatch(html,/course-unit-badge/);assert.doesNotMatch(html,/Canny|Otsu|Gaussian/);
});
test('saved-result guide uses its own recorded output and method, never current controls',()=>{
 const {c}=harness();const run={status:{state:'complete'},report:{ivp_demo:{goal:'edges',steps:[]}}};assert.equal(c.courseFeatureContext(run,'lab').goal,'edges');assert.match(c.featureUnitsHTML(run,'lab'),/Units & topics · Edge map/);assert.doesNotMatch(c.featureUnitsHTML(run,'lab'),/course-unit-badge">Unit 4</);assert.equal(c.courseFeatureContext({report:{motion:{uses_ai:true}}},'tracking').mode,'vehicles');
 const app=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');assert.match(app,/footer\.innerHTML=featureUnitsHTML\(run,runFeature\(run\)\)/);
});
test('legacy image without output trace receives no invented processing-path topics',()=>{
 const {c}=harness();const run={status:{state:'complete'},report:{settings:{gamma:1.5}}};assert.deepEqual(units(c.courseFeatureProfile('lab',c.courseFeatureContext(run,'lab'))),[]);assert.match(c.featureUnitsHTML(run,'lab'),/no selected-output trace/);
});
test('visible guide follows output, method and source kind without altering settings or sending requests',()=>{
 const {c,controls,footer}=harness();const original=JSON.stringify(c.state);c.updateFeatureUnits();assert.match(footer.innerHTML,/Connected region map/);controls.get('ivp-goal').value='edges';c.updateFeatureUnits();assert.match(footer.innerHTML,/Edge map/);assert.doesNotMatch(footer.innerHTML,/course-unit-badge">Unit 6</);c.workbenchTask='tracking';controls.get('motion-mode').value='vehicles';c.updateFeatureUnits();assert.match(footer.innerHTML,/vehicle mode/);assert.doesNotMatch(footer.innerHTML,/course-unit-badge">Unit 4</);assert.equal(JSON.stringify(c.state),original);
});
test('recorded evidence remains a separate collapsible block and escapes stored operation text',()=>{
 const {c}=harness();const run={id:'<img src=x>',status:{state:'complete'},report:{ivp_demo:{goal:'edges',steps:[{id:'canny',title:'Canny <script>bad</script>',parameters:{threshold:40}}]}}};const html=c.featureUnitsHTML(run,'lab');assert.match(html,/<details class="course-evidence">/);assert.match(html,/Recorded operations for this result/);assert.match(html,/Unit 5 — Edge Detection/);assert.doesNotMatch(html,/<script>|<img src=x>/);assert.match(html,/&lt;script&gt;/);
});
