const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

function report(method){return {method,timestamp_seconds:0,analysis_size:[300,400],fragments:[],limitations:[],parameters:{},support:{},
 summary:{coverage_percent:10,fragments:0,threshold_high_coverage_percent:9,threshold_low_coverage_percent:11,threshold_mask_iou:.8}};}

test('current and legacy survey views are explicitly review only without mutating configured rules',()=>{
 const context=vm.createContext({escapeHTML:s=>String(s),api(){throw Error('Rendering must not start inference');}});
 vm.runInContext(source.slice(source.indexOf('function surveySummary('),source.indexOf('async function runSurvey(')),context);
 for(const method of ['legacy','classical_marking_survey_v2','classical_marking_survey_v3','classical_marking_survey_v4','classical_marking_survey_v5']){
  const input=report(method),before=JSON.stringify(input),html=context.surveySummary(input);
  assert.match(html,/REVIEW ONLY/);assert.match(html,/cannot configure road, crossing, stop-line or approach geometry/);
  assert.match(html,/cannot supply incident-rule evidence/);assert.match(html,/yellow vehicle roof/);
  assert.equal(JSON.stringify(input),before);
  if(!['legacy','classical_marking_survey_v2'].includes(method))assert.match(html,/unbracketed proposals and per-component flank samples/);
 }
});

test('v4 local-contrast probes show actual thresholds, truthful non-monotonic coverage and no old brightness claim',()=>{
 const context=vm.createContext({escapeHTML:s=>String(s)});
 vm.runInContext(source.slice(source.indexOf('function surveySummary('),source.indexOf('async function runSurvey(')),context);
 const input=report('classical_marking_survey_v4');
 delete input.summary.threshold_low_coverage_percent;delete input.summary.threshold_high_coverage_percent;delete input.summary.threshold_mask_iou;
 input.sensitivity={basis:'minimum_local_contrast',units:'HSV Value levels (0-255)',base_threshold:12,
  low:{multiplier:.75,threshold:9,coverage_percent:8,mask_file:'contrast-075x.png'},
  high:{multiplier:1.25,threshold:15,coverage_percent:11,mask_file:'contrast-125x.png'},probe_mask_iou:1};
 const before=JSON.stringify(input),html=context.surveySummary(input);
 assert.match(html,/8\.00–11\.00%/);assert.match(html,/Local-contrast probes · 0\.75× \/ 1\.25×/);
 assert.match(html,/Base threshold: 12\.00 HSV Value levels/);assert.match(html,/Low probe: 9\.00 → 8\.00%/);
 assert.match(html,/High probe: 15\.00 → 11\.00%/);assert.match(html,/non-monotonic/);
 assert.match(html,/contrast-075x.png/);assert.match(html,/contrast-125x.png/);
 assert.match(html,/not accuracy or certainty/);assert.doesNotMatch(html,/±20|lower and higher brightness/);
 assert.equal(JSON.stringify(input),before);
 for(const method of ['classical_marking_survey_v2','classical_marking_survey_v3']){
  const saved=report(method),old=JSON.stringify(saved),legacy=context.surveySummary(saved);
  assert.match(legacy,/Coverage across ±20 threshold test/);assert.doesNotMatch(legacy,/Local-contrast probes/);
  assert.equal(JSON.stringify(saved),old);
 }
});

test('v5 sparse rescue shows its effective20/15/25 gates while v4 saved probes retain their recorded values',()=>{
 const context=vm.createContext({escapeHTML:s=>String(s)});
 vm.runInContext(source.slice(source.indexOf('function surveySummary('),source.indexOf('async function runSurvey(')),context);
 for(const [method,base,low,high] of [['classical_marking_survey_v4',12,9,15],['classical_marking_survey_v5',20,15,25]]){
  const input=report(method);input.parameters={flat_minority_rescued:method.endsWith('v5')};
  input.sensitivity={basis:'minimum_local_contrast',units:'HSV Value levels (0-255)',base_threshold:base,
   low:{multiplier:.75,threshold:low,coverage_percent:1,mask_file:'contrast-075x.png'},
   high:{multiplier:1.25,threshold:high,coverage_percent:0,mask_file:'contrast-125x.png'},probe_mask_iou:0};
  input.limitations=method.endsWith('v5')?['Low overall percentile range, but minority appearance was retained through stronger local contrast.']:['Recorded v4 explanation'];
  const before=JSON.stringify(input),html=context.surveySummary(input);
  assert.match(html,new RegExp('Base threshold: '+base+'\\.00'));
  assert.match(html,new RegExp('Low probe: '+low+'\\.00'));
  assert.match(html,new RegExp('High probe: '+high+'\\.00'));
  assert.match(html,/not accuracy or certainty/);assert.doesNotMatch(html,/±20|Include both road and markings/);
  if(method.endsWith('v5'))assert.match(html,/minority appearance was retained/);
  assert.equal(JSON.stringify(input),before);
 }
});
