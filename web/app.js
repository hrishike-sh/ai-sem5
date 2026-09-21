'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="traffic-token"]').content;
const state = {config:null, settings:null, media:null, preview:null, view:'stages', mode:'automatic', area:'ivp', step:1, undo:[], redo:[], revision:0, rendering:false, pending:false, region:null, regionTool:false, job:null, runs:[], review:null};
const copy = value => JSON.parse(JSON.stringify(value));
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let toastTimer, previewTimer, jobTimer, reviewTimer;
let workspaceRestoring=true;
function notify(message, error=false){ clearTimeout(toastTimer); $('notification').textContent=message; $('notification').className=error?'error':''; $('notification').hidden=false; if(error&&typeof showGuidedError==='function')showGuidedError(message); toastTimer=setTimeout(()=>$('notification').hidden=true,error?10000:4500); }
async function performApi(path, data){
 if(ivpPage()&&data!==undefined&&(path==='/api/review'||path==='/api/evidence-review'||(path==='/api/job'&&data?.ai)))throw Error('Open the full application for the broad analyzer or scene/LLM review. Bounded traffic tools are available here.');
 const createsJob=data!==undefined&&/^\/api\/(motion|helmet|zebra|job|export|survey)$/.test(path);
 const recovery=createsJob?'The job may already have started. Check Saved runs before starting it again.':'Check that Traffic Lab is running, then retry this action.';
 let response;try{response=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Traffic-Token':token},body:JSON.stringify(data)});}
 catch{throw Error('Connection to the local server was interrupted while requesting '+path+'. '+recovery);}
 let value;try{value=await response.json();}catch{throw Error('The local server returned an unreadable response for '+path+' (HTTP '+response.status+'). '+recovery);}
 if(!response.ok){const error=Error(value.error||'Request failed (HTTP '+response.status+').');error.status=response.status;throw error;}return value;
}
let apiQueue=Promise.resolve();
function queueApi(operation){const task=apiQueue.then(operation);apiQueue=task.catch(()=>{});return task;}
function api(path,data){return queueApi(()=>performApi(path,data));}
function payload(){return {media:state.media?.id,source_revision:state.media?.source_revision,timestamp:Number($('timestamp').value)||0,settings:copy(state.settings),mode:state.mode};}
function masks(){return [...document.querySelectorAll('#mask-list input:checked')].map(e=>e.value);}
function busy(message,error=false){$('preview-state').textContent=message;$('preview-state').classList.toggle('error',error);}
function actions(){directionActions();helmetActions();zebraActions();motionActions();surveyActions();syntheticActions();if(typeof refreshRedReview==='function')refreshRedReview();if(typeof syncSpeedLimit==='function')syncSpeedLimit();const ready=!!state.preview&&!state.rendering&&!state.pending&&!workspaceRestoring; for(const id of ['export-frame','export-video','analyse','region-tool']) $(id).disabled=!ready; $('review').disabled=workspaceRestoring||!state.media||!!state.review||sceneStarting; $('review-cancel').hidden=!state.review;$('review-cancel').disabled=!!state.review?.cancelling; $('undo').disabled=!state.undo.length;$('redo').disabled=!state.redo.length;sourceActions();demoActions();}
function sourceActions(){for(const id of ['open-footage','upload','sample','tracking-open-video'])$(id).disabled=workspaceRestoring;$('try-sample').disabled=workspaceRestoring||!state.config?.samples.length;$('motion-demo').disabled=workspaceRestoring||!state.config?.samples.some(s=>s.name==='india-city.mp4');}

const groups = [
 ['Light & colour',true,[
  ['exposure','Exposure','range',-3,3,.1,'stops','Multiply pixel values by 2 to the power of the exposure. Positive values can clip highlights.'],
  ['gamma','Gamma','range',.2,4,.1,'','Above 1 brightens midtones. This is a nonlinear intensity transform.'],
  ['clahe','CLAHE on colour export','checkbox'],
  ['clip_limit','CLAHE strength','range',1,8,.5,'','Limits local histogram amplification; excessive strength can reveal noise.'],
  ['stretch','Contrast clipping','range',0,10,.5,'%','Percentile stretch. 0 disables it; larger values discard more extreme tones.'],
  ['white_balance','White balance','range',0,1,.05,'','Gray-world correction strength. Assumes average scene colour is neutral; coloured scenes can be distorted.'],
  ['saturation','Saturation','range',0,2,.1,'','Adjust HSV saturation. 0 removes colour; 1 preserves it.']]],
 ['Noise & detail',false,[['denoise','Denoise filter','select',['Off','Gaussian','Median','Bilateral']],['kernel','Filter size','select',[3,5,7,9]],['sharpen','Unsharp masking','range',0,2,.1,'','Boosts existing edges. It cannot recover missing or motion-blurred information.']]],
 ['Edges & morphology',false,[['canny_low','Canny lower threshold','range',0,250,1],['canny_high','Canny upper threshold','range',1,255,1],['threshold_mode','Threshold method','select',['Otsu','Adaptive','Fixed']],['threshold_value','Binary threshold','range',0,255,1],['morphology','Morphology','select',['Off','Open','Close','Open + close']],['morph_kernel','Morphology kernel','select',[3,5,7,9]],['line_length','Minimum line length','range',5,300,1,'px']]],
 ['Colour isolation',false,[['hue_center','Hue centre','range',0,179,1,'','OpenCV hue scale: red 0, amber 20, green 60. Hue wraps at 179.'],['hue_tolerance','Hue tolerance','range',1,45,1],['saturation_min','Minimum saturation','range',0,255,1],['value_min','Minimum value','range',0,255,1]]],
 ['Size & video',false,[['max_width','Maximum output width','select',[0,640,960,1280,1920]],['scene_threshold','Scene change threshold','range',.05,1,.05,'','Lower values are more sensitive, including to lighting changes.']]]
];
function makeControls(){
 $('controls').innerHTML=groups.map(([title,open,controls])=>`<details ${open?'open':''}><summary>${title}</summary>${controls.map(([key,label,type,min,max,step,unit='',hint=''])=>{
 const id='setting-'+key; let control;
 if(type==='select') control=`<select id="${id}" data-setting="${key}">${min.map(v=>`<option value="${v}">${key==='max_width'?(v?v+' px':'Original size'):v}</option>`).join('')}</select>`;
 else if(type==='checkbox') control=`<input id="${id}" data-setting="${key}" type="checkbox">`;
 else control=`<input id="${id}" data-setting="${key}" type="range" min="${min}" max="${max}" step="${step}">`;
 return `<div class="control"><div class="control-top"><label for="${id}">${label}</label>${type==='range'?`<output id="value-${key}"></output>`:''}</div>${control}${hint?`<small>${hint}</small>`:''}</div>`;
 }).join('')}</details>`).join('');
 for(const element of document.querySelectorAll('[data-setting]')){
  element.addEventListener('input',()=>{const key=element.dataset.setting;let value=element.type==='checkbox'?element.checked:typeof state.config.defaults[key]==='number'?Number(element.value):element.value; record();state.settings[key]=value;$('settings-label').textContent='Custom'; sync();schedule();});
 }
 sync();
}
function sync(){
 const matching=Object.entries(state.config.presets).find(([,preset])=>Object.keys(preset).every(key=>preset[key]===state.settings[key]));$('preset').value=matching?matching[0]:'';
 for(const element of document.querySelectorAll('[data-setting]')){const key=element.dataset.setting; if(element.type==='checkbox')element.checked=state.settings[key]; else element.value=state.settings[key];if($('value-'+key))$('value-'+key).textContent=Number(state.settings[key]).toFixed(Number(element.step)<1?2:0);}
 $('setting-clip_limit').disabled=false;$('setting-threshold_value').disabled=state.settings.threshold_mode!=='Fixed';actions();
}
function record(){state.undo.push(copy(state.settings));if(state.undo.length>40)state.undo.shift();state.redo=[];}
function changeSettings(settings,label){setMode('manual',false);record();state.settings=copy(settings);$('settings-label').textContent=label;sync();schedule();}
function schedule(){clearSurvey();state.revision++;state.pending=!!state.media;actions();clearTimeout(previewTimer);if(state.media){busy('Adjusting…');previewTimer=setTimeout(renderPreview,300);}}
async function renderPreview(){
 if(state.rendering||!state.media)return; state.rendering=true;state.pending=false;const revision=state.revision;const request=payload();busy('Processing…');actions();
 try{const preview=await api('/api/preview',request);if(revision===state.revision){state.preview=preview;if(state.mode==='automatic'){state.settings=copy(preview.settings);sync();}paint();busy('Preview ready');if(state.region)measureRegion();}}
 catch(error){if(revision===state.revision){notify('Preview could not be prepared: '+error.message,true);busy('Preview out of date',true);state.preview=null;}}
 finally{state.rendering=false;actions();if(state.pending)renderPreview();}
}
async function openMedia(media, timestamp=0){
 if(typeof clearSpeedLimit==='function')clearSpeedLimit('New source: speed limit reset.');
 if(typeof setSourcePlayback==='function')setSourcePlayback(media);
 clearIncidents();
 if(typeof resetFeatureUnits==='function')resetFeatureUnits();clearDirection();clearZebra();clearTarget();clearCalibration();clearSurvey();$('motion-use-roi').checked=false;state.revision++;state.media=media;state.preview=null;state.region=null;state.regionTool=false;$('region-box').hidden=true;$('frame').classList.remove('selecting');$('region-tool').setAttribute('aria-pressed','false');$('region-stats').innerHTML='<p class="hint">No region selected.</p>';paintSceneReview();
 $('media-name').textContent=media.name;$('media-meta').textContent=`${media.width} × ${media.height} · ${(media.size_bytes/1048576).toFixed(1)} MB`+(media.kind==='video'?` · ${media.duration.toFixed(2)} s · ${media.fps.toFixed(2)} fps · ${media.has_audio?'Audio present':'No audio'}`:' · RGB');
 $('example-note').hidden=!media.example?.calibration_status;$('example-note').textContent=media.example?.calibration_status||'';
 if(media.example?.synthetic&&$('synthetic-choice')){const option=Array.from($('synthetic-choice').options||[]).find(o=>o.value.split('/').pop()===media.name);if(option)$('synthetic-choice').value=option.value;}
 const video=media.kind==='video';$('video-timeline').hidden=!video;$('export-video').hidden=!video;$('timestamp').max=Math.max(0,Number(media.duration||0)-.001);$('timestamp').value=Math.min(timestamp,Number($('timestamp').max));$('time-label').textContent=Number($('timestamp').value).toFixed(2)+' s';$('export-frame').textContent=video?'Save frame + IVP stages ↓':'Save image + IVP stages ↓';$('analyse').textContent=video?'Analyse full video':'Analyse image';
 $('export-note').textContent=video?'The stage ZIP covers this frame. Full video export is enhanced colour with timed audio and sampled feature masks; automatic settings stay fixed from frame zero.':'Save every IVP stage, its explanation, colour output and measurements.';
 if(media.name==='india-city.mp4'){state.region=[.04,.28,.94,.98];$('motion-use-roi').checked=true;$('motion-axis').value='horizontal';$('motion-position').value='70';drawRegion();}else if(media.name.startsWith('india-')){state.region=[0,.4,1,.98];$('motion-use-roi').checked=true;$('motion-axis').value=media.name.includes('overhead')?'horizontal':'vertical';$('motion-position').value=media.name.includes('overhead')?'65':'50';drawRegion();} $('mask-list').innerHTML='';state.pending=true;actions();await renderPreview();
}
async function sample(name){if(!name)return;try{busy('Opening…');const media=await api('/api/sample',{name});$('sample').value=name;await openMedia(media);}catch(e){notify(e.message,true);busy('Could not open',true);}}
async function upload(file){if(!file)return;if(workspaceRestoring){notify('Wait for workspace restoration to finish, then open footage.');return;}if(file.size>1024**3){notify('Choose a file smaller than 1 GB.',true);return;}busy('Uploading…');try{
 const result=await queueApi(()=>new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();xhr.open('POST','/api/upload?name='+encodeURIComponent(file.name));xhr.setRequestHeader('X-Traffic-Token',token);xhr.upload.onprogress=e=>{if(e.lengthComputable)busy(`Uploading ${Math.round(e.loaded/e.total*100)}%`);};xhr.onload=()=>{try{const value=JSON.parse(xhr.responseText);xhr.status===200?resolve(value):reject(Error(value.error));}catch{reject(Error('Upload failed. Check the local server.'));}};xhr.onerror=()=>reject(Error('Upload interrupted. Please retry.'));xhr.send(file);}));$('sample').value='';await openMedia(result);
 }catch(e){notify('Could not open uploaded file '+file.name+': '+e.message,true);busy('Upload failed',true);}finally{$('upload').value='';}}

const concepts={
 'Grayscale':'Luminance combines RGB channels into a single intensity plane. Colour information is removed.',
 'Hue':'Hue from the original colour frame, scaled for display. This is a channel visualization, not natural colour.',
 'Saturation':'Saturation measures colourfulness in the original HSV frame.', 'Value':'HSV value is the largest original RGB channel value.',
 'Colour mask':'Hue, saturation and value thresholds on the original frame isolate a colour range. A red patch alone is not a traffic signal.',
 'White marking mask':'Low saturation and high value isolate white regions. Buildings and reflections can also appear in this mask.',
 'Edges':'Canny uses intensity gradients and two thresholds to connect strong edges with nearby weaker ones.',
 'Threshold':'Binarization separates intensities using Otsu, local adaptive thresholding, or a fixed cutoff.',
 'Morphology':'Opening removes small bright regions; closing fills small dark gaps. Kernel size controls the scale.',
 'Contours':'Connected boundaries from the cleaned binary mask. Small contours are filtered; these are shapes, not classified objects.',
 'Lines':'Probabilistic Hough transform finds line segments in the Canny edge map. Segments are not automatically lanes.',
 'Road candidates':'Local brightness contrast, original white/yellow colour, shape filtering and Hough lines produce appearance candidates. Reflections and vehicle parts can still enter the mask.',
 'Stripe mask':'Geometric white-stripe candidates. This mask does not establish a zebra crossing or stop line.',
 'Sobel magnitude':'Combines horizontal and vertical intensity derivatives. The display is normalized per frame, so brightness is not comparable across frames.',
 'Laplacian':'Absolute second spatial derivative, clipped to 8-bit display. Responds strongly to fine edges and noise.',
 'Morphological gradient':'Dilation minus erosion reveals local boundaries at the chosen morphology kernel size.',
 'Top hat':'Original grayscale minus its opening highlights bright details smaller than the chosen kernel.',
 'Black hat':'Closing minus original grayscale highlights dark details smaller than the chosen kernel.'
};
function paint(){
 const p=state.preview;if(!p)return;$('empty-state').hidden=true;$('frame').hidden=false;$('compare-control').hidden=state.view!=='compare';
 try{localStorage.setItem('traffic-lab-workspace',JSON.stringify({media:state.media.id,settings:state.settings,mode:state.mode,timestamp:Number($('timestamp').value)||0}));}catch{}
 $('frame').style.setProperty('--image-ratio',p.size[0]/p.size[1]);
 const selector=$('view-choice');let choices=state.view==='features'?p.features:state.view==='stages'?Object.fromEntries(p.ivp_steps.map((step,i)=>[String(i+1).padStart(2,'0')+' · '+step.title,step.url])):null;const selected=selector.value;
 selector.hidden=!choices;if(choices){selector.innerHTML=Object.keys(choices).map(k=>`<option>${escapeHTML(k)}</option>`).join('');if(selected in choices)selector.value=selected;else if(state.view==='stages')selector.selectedIndex=Math.min(state.step,p.ivp_steps.length-1);}
 $('base-image').src=choices?choices[selector.value]:p.processed;$('base-image').alt=choices?selector.value:'Processed traffic frame';$('before-image').src=p.original;
 for(const id of ['before-image','wipe-line','original-label','processed-label'])$(id).hidden=state.view!=='compare';
 $('concept-text').textContent=state.view==='features'?concepts[selector.value]||'Intermediate feature view.':state.view==='stages'?`Processing stage: ${selector.value}. Inspect changes between consecutive stages to understand each operation.`:'Compare at matching resolution. The original source stays intact; colour masks are extracted from the original colour frame.';
 $('time-label').textContent=p.timestamp.toFixed(3)+' s';quality(p.measurements.original,p.measurements.processed,'quality-stats');drawHistogram(p.histograms);qualityHints(p.measurements);
 if(!$('mask-list').children.length)$('mask-list').innerHTML=Object.keys(p.features).map(k=>`<label><input type="checkbox" value="${escapeHTML(k)}" ${['Edges','Colour mask','White marking mask'].includes(k)?'checked':''}>${escapeHTML(k)}</label>`).join('');
 paintStages();updateWipe();paintMotionLine();if(typeof paintWorkbench==='function')paintWorkbench();
}
function updateWipe(){const percent=$('wipe').value;$('before-image').style.clipPath=`inset(0 ${100-percent}% 0 0)`;$('wipe-line').style.left=percent+'%';}
const metrics=[['brightness','Brightness','0–255',1,1],['contrast','Contrast','Standard deviation',1,1],['laplacian_variance','Edge detail','Laplacian variance',1,1],['shadow_fraction','Deep shadows','Pixels ≤ 10',100,1],['highlight_fraction','Highlights','Pixels ≥ 245',100,1],['entropy_bits','Entropy','0–8 bits',1,2]];
function quality(before,after,target){$(target).innerHTML=metrics.map(([key,label,unit,scale,places])=>`<div class="stat"><div class="stat-label">${label}<small>${unit}</small></div><div class="stat-value">${((after[key]||0)*scale).toFixed(places)}${scale===100?'%':''}<small>was ${((before[key]||0)*scale).toFixed(places)}${scale===100?'%':''}</small></div></div>`).join('');}
function drawHistogram(hist){const max=Math.max(1,...hist.original,...hist.processed);$('histogram').innerHTML=['original','processed'].map((name,i)=>`<polyline fill="none" stroke="${i?'#216755':'#a8b4a2'}" stroke-width="1.5" points="${hist[name].map((v,x)=>`${x},${98-v/max*94}`).join(' ')}"/>`).join('');}
function qualityHints(measurements){const hints=[];const before=measurements.original,after=measurements.processed;if(before.brightness<55)hints.push('Dark input: try Low light or a small gamma increase. Inspect highlights after adjustment.');if(after.highlight_fraction>before.highlight_fraction+.03)hints.push('More highlights are clipped. Consider lowering exposure, gamma, or contrast clipping.');if(after.laplacian_variance<70)hints.push('Low edge detail may indicate blur, or simply a smooth scene. Sharpening cannot recover missing detail.');if(after.contrast<28)hints.push('Low contrast: compare CLAHE with a small percentile stretch. Watch for amplified noise.');if(!hints.length)hints.push('No strong heuristic quality flags. Check the image visually before exporting.');$('quality-hints').innerHTML=hints.map(h=>`<p>${h}</p>`).join('');}
function inspectTab(name){for(const panel of ['quality','region','ai'])$(panel+'-panel').hidden=name!==panel;document.querySelectorAll('[data-inspect]').forEach(b=>{b.classList.toggle('active',b.dataset.inspect===name);b.setAttribute('aria-pressed',String(b.dataset.inspect===name));});if(name==='region'&&$('measurements-disclosure'))$('measurements-disclosure').open=true;}
let dragStart=null, dragWipe=false;
function regionPoint(event){const rect=$('frame').getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];}
function drawRegion(){if(!state.region)return;const [x1,y1,x2,y2]=state.region;Object.assign($('region-box').style,{left:x1*100+'%',top:y1*100+'%',width:(x2-x1)*100+'%',height:(y2-y1)*100+'%'});$('region-box').hidden=false;}
async function measureRegion(){clearSurvey();const revision=state.revision;try{const result=await api('/api/region',{...payload(),rectangle:state.region});if(revision!==state.revision)return;quality(result.original,result.processed,'region-stats');const text=document.createElement('p');text.className='hint';text.textContent=`Processed pixel bounds: ${result.bounds_processed_xyxy_exclusive.join(', ')}. Source corner centres: ${result.source_corner_pixel_centres.map(p=>p.map(v=>v.toFixed(1)).join(', ')).join(' → ')}.`;$('region-stats').append(text);const button=document.createElement('button');button.className='quiet';button.textContent='Download region measurements';button.onclick=()=>downloadJSON({source:state.media.name,timestamp:state.preview.timestamp,settings:state.preview.settings,...result},'region-measurements.json');$('region-stats').append(button);}catch(e){notify(e.message,true);}}
function downloadJSON(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function links(items){return items.map(item=>`<a href="${escapeHTML(item.url)}">↓ ${escapeHTML(item.label)}</a>`).join('');}
function jobMessage(run){
 const s=run.status,parts=[s.message||''];
 const total=run.report?.info?.frame_count||run.report?.source_info?.frame_count;
 if(Number.isFinite(s.frames))parts.push(`${s.frames}${total?' / '+total:''} ${s.frames===1&&!total?'frame':'frames'} processed`);
 if(Number.isFinite(s.fps)&&s.frames>=30)parts.push(`${s.fps.toFixed(1)} processing frames/s`);
 return parts.filter(Boolean).join(' · ');
}
function showJob(run){if(typeof rememberFeatureRun==='function')rememberFeatureRun(run);showMotionResult(run);state.job=run.id;$('active-job').hidden=false;$('job-title').textContent=run.source+' · '+run.status.state;$('job-progress').value=run.status.progress||0;$('job-message').textContent=jobMessage(run);$('job-downloads').innerHTML=links(run.downloads);const active=['queued','running','encoding'].includes(run.status.state);try{if(active||run.status.state==='unknown')localStorage.setItem('traffic-lab-active-job',run.id);else if(localStorage.getItem('traffic-lab-active-job')===run.id)localStorage.removeItem('traffic-lab-active-job');}catch{}$('cancel-job').hidden=!active;$('cancel-job').disabled=false;$('dismiss-job').hidden=active||run.status.state==='unknown';$('job-output-options').hidden=active||!run.downloads.length;const resultButton=$('view-job-result');if(resultButton){resultButton.hidden=run.status.state!=='complete';resultButton.onclick=()=>openRun(run.id);}clearTimeout(jobTimer);if(active||run.status.state==='unknown')jobTimer=setTimeout(pollJob,1500);}
function showFrameExport(run){if(typeof mayPresentRun==='function'&&!mayPresentRun(run))return;if(typeof rememberFeatureRun==='function')rememberFeatureRun(run);const card=$('frame-export-result');card.hidden=false;card.innerHTML=`<strong>Saved frame · ${escapeHTML(run.source)}</strong><p>${escapeHTML(run.status.message||'Frame export ready.')}</p><div class="downloads">${links(run.downloads)}</div>`;}
async function pollJob(){
 const id=state.job;if(!id)return;
 try{const run=await api('/api/run?id='+encodeURIComponent(id));if(state.job===id)showJob(run);}
 catch(e){if(state.job!==id)return;$('job-message').textContent=e.message;
  if([400,401,403,404,410].includes(e.status)){
   clearTimeout(jobTimer);jobTimer=null;state.job=null;
   $('cancel-job').hidden=true;$('dismiss-job').hidden=false;$('job-title').textContent='Run status unavailable';
   $('job-message').textContent='Stopped checking this record. This does not establish whether a worker is still running. '+e.message;
   $('job-downloads').innerHTML='';$('job-output-options').hidden=true;if($('view-job-result'))$('view-job-result').hidden=true;
   try{if(localStorage.getItem('traffic-lab-active-job')===id)localStorage.removeItem('traffic-lab-active-job');}catch{}
   notify('This saved run cannot be opened: '+e.message+' Open Saved runs and select an available record. Processing was not restarted.',true);
   if(typeof updateGuidedUX==='function')updateGuidedUX();
  }else{jobTimer=setTimeout(pollJob,5000);}
 }
}
async function startJob(ai){try{const context=typeof captureFeatureSubmission==='function'?captureFeatureSubmission():null;const run=await api('/api/job',{...payload(),masks:masks(),ai});if(context)bindFeatureSubmission(run,context);showJob(run);notify('Processing started. You can continue inspecting the frame.');}catch(e){notify(e.message,true);}}
let libraryRevision=0,libraryNext=null,libraryTimer=null;
async function loadRuns(append=false){
 append=append===true;if(append&&!libraryNext)return;
 const revision=++libraryRevision,text=$('run-search').value,filter=$('run-filter').value,before=append?libraryNext:'';
 const more=$('library-more'),summary=$('library-summary');more.disabled=true;
 if(!append){libraryNext=null;more.hidden=true;}summary.textContent=append?'Loading older results…':'Searching all saved runs…';
 try{
  const page=await api('/api/run-library?q='+encodeURIComponent(text)+'&status='+encodeURIComponent(filter)+'&before='+encodeURIComponent(before));
  if(revision!==libraryRevision||text!==$('run-search').value||filter!==$('run-filter').value)return;
  const rows=append?[...state.runs,...page.runs]:page.runs;state.runs=[...new Map(rows.map(r=>[r.id,r])).values()];
  libraryNext=page.has_more?page.next_before:null;more.hidden=!page.has_more;paintRuns();
  summary.textContent=`${state.runs.length} saved ${state.runs.length===1?'run':'runs'} shown${page.has_more?' · More available':''}`;
 }catch(e){if(revision===libraryRevision){summary.textContent=e.message;notify(e.message,true);}}
 finally{if(revision===libraryRevision)more.disabled=false;}
}
function scheduleRuns(){libraryRevision++;clearTimeout(libraryTimer);libraryNext=null;$('library-more').hidden=true;$('library-summary').textContent='Searching all saved runs…';libraryTimer=setTimeout(()=>loadRuns(),200);}
function paintRuns(){const runs=state.runs;$('runs').innerHTML=runs.map(run=>`<article class="run-row"><div><h3>${escapeHTML(run.source)}</h3><p>${escapeHTML(run.id)} · ${escapeHTML(run.status.message||'')}</p></div><span class="status-chip">${escapeHTML(run.status.state)}</span><button class="quiet" data-run="${escapeHTML(run.id)}">Open result ↗</button></article>`).join('')||`<p class="hint">${$('run-search').value||$('run-filter').value?'No saved runs match this search.':'No saved runs yet. Export a frame or process a video to start your library.'}</p>`;$('runs').querySelectorAll('[data-run]').forEach(button=>button.onclick=()=>openRun(button.dataset.run));}
async function openRun(id){
 const revision=++runOpenRevision;
 try{const run=await api('/api/run?id='+encodeURIComponent(id));if(revision!==runOpenRevision)return;$('result-title').textContent=run.source;const report=run.report,ai=report.ai;let html=`<p>${escapeHTML(run.status.message||'')}</p>`;if(run.preview)html+=run.preview.video?`<video controls src="${escapeHTML(run.preview.url)}"></video>`:`<img src="${escapeHTML(run.preview.url)}" alt="${run.preview.name==='detections.png'?'Object detections, not violation decisions':'Processed export'}">`;if(run.preview?.name==='detections.png')html+='<p>Cyan boxes show object detections, not violations. Download the processed image to inspect reported events.</p>';
 if(ai){const total=Object.values(ai.events_by_rule||{}).reduce((a,b)=>a+b,0);html+=`<h3>${escapeHTML(total)} reported events</h3><p>Zero events does not prove the scene is violation-free. “Cannot assess” means evidence or scene context was insufficient.</p><table><thead><tr><th>Rule / assessment</th><th>Frame assessments</th></tr></thead><tbody>${Object.entries(ai.assessment_counts||{}).map(([k,v])=>`<tr><td>${escapeHTML(k.replaceAll('_',' '))}</td><td>${escapeHTML(v)}</td></tr>`).join('')}</tbody></table>`;html+=aiAssessmentPanel(run);}
 if(run.status.state==='complete'){if(report.helmet)html+=helmetSummary(report.helmet);if(report.zebra)html+=zebraSummary(report.zebra);}html+=`<div class="downloads">${links(run.downloads)}</div>`;if(report.motion&&run.status.state==='complete')html+=motionSummary(report.motion);html+=evidencePanel(run);if(report.kind==='ivp_marking_survey')html+=surveySummary(report);if(report.settings)html+='<button id="reuse-settings" class="quiet">Use these adjustments</button>';html+=`<details><summary>Full processing record</summary><pre>${escapeHTML(JSON.stringify(report,null,2))}</pre></details>`;$('result-body').innerHTML=html;bindEvidencePanel($('result-body').querySelector('.evidence-panel:not(.ai-assessment-panel)'));bindAiAssessmentPanel($('result-body').querySelector('.ai-assessment-panel'));if($('reuse-settings'))$('reuse-settings').onclick=()=>{changeSettings({...state.config.defaults,...report.settings},'Saved run');$('result-dialog').close();selectWorkbench('lab');notify('Saved adjustments applied.');};if(report.motion&&run.status.state==='complete')showMotionResult(run);if(typeof organizeSavedResult==='function')organizeSavedResult(run);if(typeof rememberFeatureRun==='function'){rememberFeatureRun(run);const panel=$('saved-overview');if(panel){const footer=document.createElement('section');footer.className='course-units';footer.innerHTML=featureUnitsHTML(run,runFeature(run));panel.append(footer);}}$('result-dialog').showModal();if(['queued','running','encoding','unknown'].includes(run.status.state))showJob(run);
 }catch(e){if(revision===runOpenRevision)notify(e.message,true);}}

function aiAssessmentPanel(run){
 if(run.status.state!=='complete')return '';
 const rules=['helmet_absence','triple_riding','red_light_crossing','stop_line_encroachment','crossing_obstruction_during_red','wrong_way','footpath_riding','restricted_entry','prohibited_turn'];
 return `<details class="ai-assessment-panel evidence-panel" data-ai-run="${escapeHTML(run.id)}"><summary>Inspect recorded AI assessments</summary><p>Read why each saved decision was made. This reads the existing export; it does not run AI or change a verdict.</p><div class="ai-assessment-controls"><label>Verdict<select class="ai-assessment-verdict"><option value="cannot_assess">Cannot assess</option><option value="no_violation_observed">No violation observed</option><option value="violation">Violation decisions</option><option value="all">All verdicts</option></select></label><label>Rule<select class="ai-assessment-rule"><option value="all">All rules</option>${rules.map(r=>`<option value="${r}">${escapeHTML(r.replaceAll('_',' '))}</option>`).join('')}</select></label></div><button class="ai-assessment-load quiet">Load recorded assessments</button><p class="ai-assessment-status" role="status">Reasons are loaded only when requested.</p><nav class="ai-assessment-nav" aria-label="Saved AI assessment pages" hidden><button class="ai-assessment-prev quiet">← Previous</button><span class="ai-assessment-page"></span><button class="ai-assessment-next quiet">Next →</button></nav><div class="ai-assessment-content"></div></details>`;
}
function aiAssessmentCards(view){
 const recorded=value=>value===null||value===undefined?'Unrecorded':String(value);
 return `<p>${escapeHTML(view.limits)}</p><p>${escapeHTML(view.provenance_notice)}</p>${view.decisions.length?view.decisions.map(card=>{const d=card.decision,c=card.frame_context,s=card.source;return `<article class="evidence-assessment ai-assessment-card"><h4>${escapeHTML(d.verdict.replaceAll('_',' '))} · ${escapeHTML(d.rule.replaceAll('_',' '))}</h4><p>Source assessment ${card.source_position} · ${escapeHTML(recorded(d.timestamp))} s · track ${escapeHTML(recorded(d.track_id))}</p><p>${escapeHTML(typeof d.reason==='string'?d.reason:'Reason not recorded as text.')}</p><details><summary>Recorded source and evidence</summary><p>${escapeHTML(s.file)} · line ${s.line} · ${escapeHTML(s.pointer)}</p><p>Recorded frame index: ${escapeHTML(recorded(c.frame))}; frame timestamp: ${escapeHTML(recorded(c.timestamp))}; scene: ${escapeHTML(recorded(c.scene_id))}; revision: ${escapeHTML(recorded(c.scene_revision))}; camera stability gate: ${escapeHTML(recorded(c.camera_stable))}.</p><p>Frame synthetic flag: ${escapeHTML(recorded(c.synthetic))}. Missing provenance is not proof of real footage. Boxes retain the export's coordinate space; use its coordinate-mapping download.</p><pre>${escapeHTML(JSON.stringify({decision:d,frame_context:c,source:s},null,2))}</pre></details></article>`;}).join(''):'<p>No recorded assessments match these filters.</p>'}<details><summary>Export fingerprints and inspection limits</summary><p>${escapeHTML(view.integrity)}</p><p>Report count check: ${escapeHTML(view.report_counts)}. Up to 20 decisions per page; 64 MiB stream, 2 MiB source-line and 64 KiB card limits. Larger exports remain available through the original download.</p><pre>${escapeHTML(JSON.stringify(view.fingerprints,null,2))}</pre></details>`;
}
function bindAiAssessmentPanel(panel){
 if(!panel)return;
 const element=name=>panel.querySelector('.ai-assessment-'+name),loadButton=element('load'),verdict=element('verdict'),rule=element('rule'),status=element('status'),content=element('content'),nav=element('nav'),previous=element('prev'),next=element('next'),label=element('page');
 let current=null,generation=0,started=false,busy=false;
 function navigation(){nav.hidden=!current;previous.disabled=busy||!current?.pagination.has_previous;next.disabled=busy||!current?.pagination.next_cursor;if(current){const p=current.pagination;label.textContent=`${p.start}–${p.end} of ${p.filtered_total} matching · ${p.total} recorded`;}}
 async function load(cursor=null){
  const ticket=++generation,v=verdict.value,r=rule.value;started=true;busy=true;loadButton.disabled=true;status.textContent='Reading saved assessment reasons…';navigation();
  try{const view=await api('/api/ai-assessments?id='+encodeURIComponent(panel.dataset.aiRun)+'&verdict='+encodeURIComponent(v)+'&rule='+encodeURIComponent(r)+(cursor?'&cursor='+encodeURIComponent(cursor):''));if(ticket!==generation||!panel.isConnected)return;current=view;content.innerHTML=aiAssessmentCards(view);verdict.value=view.pagination.verdict;rule.value=view.pagination.rule;status.textContent='Recorded decisions loaded. Original verdicts are unchanged.';loadButton.textContent='Refresh saved assessments';}
  catch(error){if(ticket!==generation||!panel.isConnected)return;if(error.status===400){current=null;content.innerHTML='';}else if(current){verdict.value=current.pagination.verdict;rule.value=current.pagination.rule;}status.textContent=error.message;}
  finally{if(ticket===generation&&panel.isConnected){busy=false;loadButton.disabled=false;navigation();}}
 }
 loadButton.onclick=()=>load();verdict.onchange=()=>started?load():undefined;rule.onchange=()=>started?load():undefined;
 previous.onclick=()=>current?.pagination.has_previous?load(current.pagination.previous_cursor):undefined;
 next.onclick=()=>current?.pagination.next_cursor?load(current.pagination.next_cursor):undefined;
 navigation();
}

let sceneStarting=false;
const sceneResults=new Map();
function sceneSource(){return state.media?state.media.id+':'+state.media.source_revision:'';}
function sceneStore(){try{localStorage.setItem('traffic-lab-scene-active',JSON.stringify(state.review));localStorage.setItem('traffic-lab-scene-results',JSON.stringify([...sceneResults].map(([source,value])=>[source,value.id])));}catch{}}
function paintSceneReview(){
 const saved=sceneResults.get(sceneSource()),output=$('review-output');output.textContent='';
 if(saved){const review=saved.review;output.textContent=review.observation.summary;
  const note=document.createElement('p');note.textContent=`Original frame at ${review.timestamp}s · ${(review.total_seconds??review.duration_seconds).toFixed(1)}s total · Unverified feature categories.`;output.append(note);
  const limits=document.createElement('p');limits.className='hint';limits.textContent=review.observation.uncertainties.join(' ');output.append(limits);
  const button=document.createElement('button');button.className='quiet';button.textContent='Download overview';button.onclick=()=>downloadJSON(review,'scene-review.json');output.append(button);
 }
 $('review-status').textContent=state.review?'A local overview is active. You can cancel it; previous successful results remain below.':'';
}
async function startReview(){
 if(!state.media||state.review||sceneStarting)return;
 const source=sceneSource(),stamp=Number($('timestamp').value)||0,request=payload();sceneStarting=true;actions();
 $('review-status').textContent='Starting local CPU overview… Previous successful results are retained.';
 try{const result=await api('/api/review',request);state.review={id:result.id,source,stamp};sceneStore();reviewTimer=setTimeout(pollReview,300);}
 catch(e){$('review-status').textContent=e.message;notify(e.message,true);}
 finally{sceneStarting=false;actions();}
}
async function cancelReview(){
 const active=state.review;if(!active)return;active.cancelling=true;actions();$('review-status').textContent='Cancelling the local overview…';sceneStore();
 try{await api('/api/review/cancel',{id:active.id});clearTimeout(reviewTimer);await pollReview();}
 catch(e){if(state.review!==active)return;active.cancelling=false;$('review-status').textContent=e.message;notify(e.message,true);actions();}
}
async function pollReview(){
 const active=state.review;if(!active)return;
 try{const result=await api('/api/review?id='+encodeURIComponent(active.id));if(state.review!==active)return;
  if(['running','cancelling'].includes(result.state)){
   $('review-status').textContent=`${result.message} ${Number(result.elapsed_seconds||0).toFixed(1)} s elapsed. You can continue IVP work.`;
   reviewTimer=setTimeout(pollReview,result.state==='cancelling'?300:1500);return;
  }
  if(result.state==='complete'){
   sceneResults.delete(active.source);sceneResults.set(active.source,{id:active.id,review:result.review});if(sceneResults.size>8)sceneResults.delete(sceneResults.keys().next().value);
   state.review=null;paintSceneReview();$('review-status').textContent=active.source===sceneSource()?'Overview complete. Categories require visual checking.':'Overview completed for the previous source; reopen it to inspect the result.';
  }else{state.review=null;$('review-status').textContent=result.message||result.error||'Local overview unavailable. Previous successful results are retained.';}
  sceneStore();actions();
 }catch(e){if(state.review!==active)return;if(e.status===400||e.status===404){state.review=null;sceneStore();actions();$('review-status').textContent=e.message+' Previous successful results are retained.';}else{$('review-status').textContent='Connection interrupted; retrying overview status. '+e.message;reviewTimer=setTimeout(pollReview,1500);}}
}
async function restoreSceneReviews(){
 try{const saved=JSON.parse(localStorage.getItem('traffic-lab-scene-results')||'[]');
  if(Array.isArray(saved))await Promise.all(saved.slice(-8).filter(v=>Array.isArray(v)&&v.length===2).map(async([source,id])=>{try{const result=await api('/api/review?id='+encodeURIComponent(id));if(result.state==='complete'&&result.media_id+':'+result.source_revision===source)sceneResults.set(source,{id,review:result.review});}catch{}}));
  const active=JSON.parse(localStorage.getItem('traffic-lab-scene-active')||'null');if(active&&typeof active.id==='string'&&typeof active.source==='string'){state.review=active;reviewTimer=setTimeout(pollReview,100);}
  paintSceneReview();actions();
 }catch{notify('Could not restore the previous overview. IVP results remain available.',true);}
}

async function init(){setupWorkbench();try{state.config=await api('/api/config');if(ivpPage()||state.config.vehicle_tracking_ready===false)$('motion-mode').value='ivp';motionModeChanged();if(!state.config.samples.some(s=>s.name==='india-city.mp4')){$('motion-demo').disabled=true;$('motion-demo').textContent='Indian demo unavailable on this computer';}state.settings=copy(state.config.defaults);$('connection').textContent='Restoring…';$('preset').innerHTML='<option value="" disabled>Custom adjustments</option>'+Object.keys(state.config.presets).map(k=>`<option>${escapeHTML(k)}</option>`).join('');$('sample').insertAdjacentHTML('beforeend',state.config.samples.map(s=>`<option value="${escapeHTML(s.name)}">${escapeHTML(s.label)}</option>`).join(''));makeControls();setMode('automatic',false);}catch(e){$('connection').textContent='Offline';notify(e.message,true);return;}
 setupZebra();setupDirection();
 $('target-mark').onclick=()=>{if(!state.preview)return;$('timestamp').value=0;schedule();calibrating=false;selectingTarget=true;targetRectangle=null;targetStart=null;state.regionTool=false;$('viewport').scrollIntoView({behavior:'smooth',block:'center'});$('target-status').textContent='Drag a tight box around the vehicle in frame zero.';};$('target-clear').onclick=clearTarget;$('target-demo').onclick=()=>{if(state.media?.name!=='india-delhi-overhead.mp4'){notify('Load the Delhi intersection sample first.',true);return;}$('timestamp').value=0;schedule();targetRectangle=[.49,.60,.64,.69];selectingTarget=false;drawTarget();};
 $('speed-mark').onclick=()=>{if(state.media?.kind!=='video'||!state.preview||state.rendering||state.pending){notify('Wait for the video preview to finish.',true);return;}$('timestamp').value=0;$('time-label').textContent='0.00 s';state.step=0;document.querySelector('[data-view="stages"]').click();$('view-choice').selectedIndex=0;paint();schedule();selectingTarget=false;targetStart=null;calibrating=true;calibrationPoints=[];calibrationSource={media_id:state.media.id,source_revision:state.media.source_revision,reference_frame:0};$('speed-enabled').checked=false;state.regionTool=false;$('region-tool').setAttribute('aria-pressed','false');drawCalibration();$('viewport').scrollIntoView({behavior:'smooth',block:'center'});};$('speed-clear').onclick=clearCalibration;$('speed-enabled').onchange=drawCalibration;
 $('motion-demo').onclick=()=>sample('india-city.mp4');$('motion-run').onclick=startMotion;$('motion-mode').onchange=motionModeChanged;$('motion-axis').onchange=paintMotionLine;$('motion-position').oninput=paintMotionLine;
 $('synthetic-open').onclick=openSyntheticExample;$('synthetic-apply').onclick=()=>{try{applySyntheticSetup();}catch(e){notify(e.message,true);}};
 document.querySelectorAll('.incident-setup input,.incident-setup select,.incident-setup textarea').forEach(el=>el.addEventListener('input',paintIncidents));
 $('incident-frame-zero').onclick=()=>{$('timestamp').value=0;state.step=0;schedule();$('frame').scrollIntoView({behavior:'smooth',block:'center'});};
 $('survey-run').onclick=runSurvey;$('survey-select').onclick=selectSurveyRegion;
 $('processing-mode').onchange=()=>setMode($('processing-mode').value);$('ivp-workspace').onclick=()=>typeof navigateTool==='function'?navigateTool('lab'):selectWorkbench('lab',true);$('ai-workspace').onclick=()=>selectWorkbench('detection',true);$('previous-stage').onclick=()=>selectStage(state.step-1);$('next-stage').onclick=()=>selectStage(state.step+1);
 $('open-footage').onclick=()=>$('upload').click();$('tracking-open-video').onclick=()=>$('upload').click();$('jump-controls').onclick=()=>document.querySelector('.tools').scrollIntoView({behavior:'smooth'});$('jump-inspector').onclick=()=>document.querySelector('.inspector').scrollIntoView({behavior:'smooth'});
 $('preset').onchange=()=>changeSettings(state.config.presets[$('preset').value],$('preset').value);$('reset').onclick=()=>{$('preset').value='Original';changeSettings(state.config.defaults,'Original');};$('undo').onclick=()=>{state.redo.push(copy(state.settings));state.settings=state.undo.pop();$('settings-label').textContent='Custom';sync();schedule();};$('redo').onclick=()=>{state.undo.push(copy(state.settings));state.settings=state.redo.pop();sync();schedule();};
 $('sample').onchange=()=>sample($('sample').value);$('try-sample').onclick=()=>{const name=state.config.samples.find(s=>s.name==='street.jpg')?.name||state.config.samples[0]?.name;$('sample').value=name;sample(name);};$('upload').onchange=()=>upload($('upload').files[0]);$('dropzone').ondragover=e=>{e.preventDefault();$('dropzone').classList.add('dragging');};$('dropzone').ondragleave=()=>$('dropzone').classList.remove('dragging');$('dropzone').ondrop=e=>{e.preventDefault();$('dropzone').classList.remove('dragging');upload(e.dataTransfer.files[0]);};
 document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{state.view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',b===button?'true':'false');});$('view-choice').innerHTML='';paint();});$('view-choice').onchange=()=>{state.step=$('view-choice').selectedIndex;paint();};$('wipe').oninput=updateWipe;$('timestamp').oninput=()=>{$('time-label').textContent=Number($('timestamp').value).toFixed(2)+' s';schedule();};
 document.querySelectorAll('[data-inspect]').forEach(button=>button.onclick=()=>inspectTab(button.dataset.inspect));$('region-tool').onclick=()=>{state.regionTool=!state.regionTool;$('frame').classList.toggle('selecting',state.regionTool);$('region-tool').setAttribute('aria-pressed',String(state.regionTool));if(state.regionTool)inspectTab('region');};$('frame').onpointerdown=e=>{if(state.rendering||state.pending||!state.preview)return;if(directionDrawing){addDirectionPoint(regionPoint(e));return;}if(zebraDrawing){addZebraPoint(regionPoint(e));return;}if(selectingTarget){targetStart=regionPoint(e);$('frame').setPointerCapture(e.pointerId);return;}if(calibrating){if(!$('base-image').complete||!$('base-image').naturalWidth)return;if(Number($('timestamp').value)!==0||state.preview.timestamp!==0){notify('Return to frame zero before marking road corners.',true);return;}calibrationPoints.push(regionPoint(e));if(calibrationPoints.length===4){calibrating=false;$('speed-enabled').checked=true;}drawCalibration();return;}if(state.regionTool)dragStart=regionPoint(e);else if(state.view==='compare'){dragWipe=true;$('wipe').value=regionPoint(e)[0]*100;updateWipe();}else return;$('frame').setPointerCapture(e.pointerId);};$('frame').onpointermove=e=>{if(targetStart){const end=regionPoint(e);targetRectangle=[Math.min(targetStart[0],end[0]),Math.min(targetStart[1],end[1]),Math.max(targetStart[0],end[0]),Math.max(targetStart[1],end[1])];drawTarget();return;}if(dragWipe){$('wipe').value=regionPoint(e)[0]*100;updateWipe();return;}if(!dragStart)return;const end=regionPoint(e);state.region=[Math.min(dragStart[0],end[0]),Math.min(dragStart[1],end[1]),Math.max(dragStart[0],end[0]),Math.max(dragStart[1],end[1])];drawRegion();};$('frame').onpointerup=()=>{if(targetStart){targetStart=null;selectingTarget=false;drawTarget();return;}dragWipe=false;if(dragStart&&state.region){dragStart=null;measureRegion();}dragStart=null;};$('frame').onpointercancel=()=>{targetStart=null;selectingTarget=false;dragStart=null;dragWipe=false;};$('clear-region').onclick=()=>{state.region=null;clearSurvey();$('region-box').hidden=true;$('region-stats').innerHTML='<p class="hint">No region selected.</p>';};
 $('export-frame').onclick=async()=>{try{$('export-frame').disabled=true;const context=typeof captureFeatureSubmission==='function'?captureFeatureSubmission():null;const run=await api('/api/export',{...payload(),masks:masks()});if(context)bindFeatureSubmission(run,context);showFrameExport(run);notify('Frame and report saved. Downloads are ready below.');}catch(e){notify(e.message,true);}finally{actions();}};$('export-video').onclick=()=>startJob(false);$('analyse').onclick=()=>startJob(true);$('review').onclick=startReview;$('review-cancel').onclick=cancelReview;$('cancel-job').onclick=async()=>{try{await api('/api/cancel',{id:state.job});$('cancel-job').disabled=true;notify('Cancellation requested.');}catch(e){notify(e.message,true);}};
 $('save-settings').onclick=()=>downloadJSON(state.settings,'traffic-lab-settings.json');$('import-settings').onchange=async()=>{try{const file=$('import-settings').files[0];if(!file)return;if(file.size>65536)throw Error('Settings file is too large.');const value=JSON.parse(await file.text());const settings=value.settings||value;if(!settings||Array.isArray(settings)||typeof settings!=='object'||Object.keys(settings).some(k=>!(k in state.config.defaults)))throw Error('This is not a supported processing settings file.');await api('/api/validate-settings',{settings});changeSettings({...state.config.defaults,...settings},'Imported');notify('Settings loaded.');}catch(e){notify(e.message,true);}finally{$('import-settings').value='';}};
 $('history-button').onclick=()=>typeof navigateTool==='function'?navigateTool('library'):selectWorkbench('library',true);$('back-workspace').onclick=()=>typeof navigateTool==='function'?navigateTool(lastWorkbenchTask):selectWorkbench(lastWorkbenchTask,true);$('refresh-runs').onclick=()=>loadRuns();$('run-search').oninput=scheduleRuns;$('run-filter').onchange=()=>loadRuns();$('library-more').onclick=()=>loadRuns(true);$('close-result').onclick=()=>$('result-dialog').close();
 try{const saved=JSON.parse(localStorage.getItem('traffic-lab-workspace')||'null');if(saved){state.settings=await api('/api/validate-settings',{settings:saved.settings});setMode(saved.mode==='manual'?'manual':'automatic',false);sync();$('settings-label').textContent='Restored';await openMedia(await api('/api/media?id='+encodeURIComponent(saved.media)),saved.timestamp||0);}}catch{notify('Open footage to continue. Previously exported runs are still in Saved runs.');}
 try{const id=localStorage.getItem('traffic-lab-active-job');if(id)showJob(await api('/api/run?id='+encodeURIComponent(id)));}catch{notify('Could not restore job progress. Check Saved runs or reload to retry.',true);}
 if(!ivpPage())await restoreFullAppTools();
 workspaceRestoring=false;$('connection').textContent='Connected';actions();applyPageRoute();
}

function setMode(mode, rerender=true){
 state.mode=mode;$('processing-mode').value=mode;$('manual-settings').hidden=mode!=='manual';$('automatic-reasons').hidden=mode!=='automatic';
 if(rerender)schedule();
}
function setArea(area){
 $('workspace').hidden=false;$('library').hidden=true;
 state.area=area;document.body.classList.toggle('ai-workspace',area==='ai');
 $('ivp-workspace').className=area==='ivp'?'primary':'quiet';$('ai-workspace').className=area==='ai'?'primary':'quiet';
 $('workspace-label').textContent=area==='ivp'?'IVP WORKSPACE · AI ASSIST OPTIONAL':'AI WORKSPACE · EXPERIMENTAL';
 document.querySelector('.inspector-tabs').hidden=area==='ai';inspectTab(area==='ai'?'ai':'quality');
 state.view=area==='ivp'?'stages':'processed';document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b.dataset.view===state.view);b.setAttribute('aria-pressed',String(b.dataset.view===state.view));});
 $('view-choice').innerHTML='';paint();
}
function selectStage(index){
 if(!state.preview)return;state.step=Math.max(0,Math.min(state.preview.ivp_steps.length-1,index));state.view='stages';
 document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b.dataset.view==='stages');b.setAttribute('aria-pressed',String(b.dataset.view==='stages'));});
 $('view-choice').innerHTML='';paint();$('stage-navigation').scrollIntoView({behavior:'smooth',block:'nearest'});
}
function paintStages(){
 const p=state.preview;if(!p)return;const steps=p.ivp_steps;
 $('ivp-board').hidden=state.area!=='ivp';$('stage-navigation').hidden=state.view!=='stages';
 $('automatic-reasons').innerHTML='<strong>Automatic plan</strong><ul>'+(p.automatic.reasons||[]).map(v=>'<li>'+escapeHTML(v)+'</li>').join('')+'</ul><p>'+escapeHTML(p.automatic.scope||'')+'</p><p>Automatic choices are conservative heuristics, not guaranteed optimal settings.</p>';
 if(state.view==='stages'){
 state.step=$('view-choice').selectedIndex;const step=steps[state.step];
 $('stage-title').textContent=`${state.step+1} / ${steps.length} · ${step.title}`;
 $('previous-stage').disabled=state.step===0;$('next-stage').disabled=state.step===steps.length-1;
 $('download-stage').href=step.download;
 const parent=steps.find(s=>s.id===step.parent);
 $('concept-text').textContent=`${step.branch} · Input: ${parent?parent.title:'uploaded frame'}. ${step.explanation} ${Object.keys(step.parameters).length?'Parameters: '+JSON.stringify(step.parameters):''}`;
 }
 $('stage-cards').innerHTML=steps.map((step,i)=>`<button class="stage-card ${state.view==='stages'&&state.step===i?'selected':''}" data-step="${i}" aria-label="Inspect stage ${i+1}: ${escapeHTML(step.title)}"><img loading="lazy" src="${escapeHTML(step.url)}" alt="${escapeHTML(step.title)}"><span class="step-number">${String(i+1).padStart(2,'0')} / ${escapeHTML(step.branch)}</span><strong>${escapeHTML(step.title)}</strong><small>Input: ${escapeHTML(steps.find(s=>s.id===step.parent)?.title||'uploaded frame')}</small></button>`).join('');
 $('stage-cards').querySelectorAll('[data-step]').forEach(button=>button.onclick=()=>selectStage(Number(button.dataset.step)));
}

let surveying=false, surveyRevision=0;
function surveyActions(){
 $('survey-select').disabled=!state.preview||state.rendering||state.pending||surveying;
 $('survey-run').disabled=!state.preview||!state.region||state.rendering||state.pending||surveying;
}
function clearSurvey(){
 clearDemonstration();
 surveyRevision++;$('survey-result').hidden=true;
 $('survey-status').textContent=state.region?'Selection changed. Run a fresh survey to save measurements.':'Select road only, excluding vehicles and buildings where possible. For video, this surveys the selected frame.';
 surveyActions();
}
function selectSurveyRegion(){
 state.view='stages';state.step=0;$('view-choice').innerHTML='';selectStage(0);
 state.regionTool=true;$('frame').classList.add('selecting');$('region-tool').setAttribute('aria-pressed','true');inspectTab('region');
 $('survey-status').textContent='Drag a rectangle over the road in the original-colour image below.';
 $('frame').scrollIntoView({behavior:'smooth',block:'center'});
}
function surveySummary(report){
 const s=report.summary;
 const rows=report.fragments.slice(0,100);
 const localSupport=['classical_marking_survey_v2','classical_marking_survey_v3','classical_marking_survey_v4','classical_marking_survey_v5'].includes(report.method);
 const flanks=['classical_marking_survey_v3','classical_marking_survey_v4','classical_marking_survey_v5'].includes(report.method);
 const contrastProbes=['classical_marking_survey_v4','classical_marking_survey_v5'].includes(report.method),p=report.sensitivity;
 const coverage=contrastProbes?[p?.low?.coverage_percent,p?.high?.coverage_percent]:[s.threshold_low_coverage_percent,s.threshold_high_coverage_percent];
 const range=coverage.every(Number.isFinite)?`${Math.min(...coverage).toFixed(2)}–${Math.max(...coverage).toFixed(2)}%`:'Not recorded';
 const agreement=contrastProbes?p?.probe_mask_iou:s.threshold_mask_iou;
 const probeDetails=contrastProbes?(p?.basis==='minimum_local_contrast'&&[p.base_threshold,p.low?.threshold,p.high?.threshold,p.low?.multiplier,p.high?.multiplier,...coverage].every(Number.isFinite)?`<details class="contrast-probes"><summary>Local-contrast probes · ${p.low.multiplier}× / ${p.high.multiplier}×</summary><p>Base threshold: ${p.base_threshold.toFixed(2)} ${escapeHTML(p.units)}. Low probe: ${p.low.threshold.toFixed(2)} → ${p.low.coverage_percent.toFixed(2)}% coverage. High probe: ${p.high.threshold.toFixed(2)} → ${p.high.coverage_percent.toFixed(2)}% coverage.</p><p>Component filtering can make coverage non-monotonic. The range above uses the smaller and larger measured probe coverage, regardless of threshold order. Inspect ${escapeHTML(p.low.mask_file)} and ${escapeHTML(p.high.mask_file)} in the evidence ZIP.</p></details>`:'<p class="hint">Local-contrast probe details were not recorded consistently in this report.</p>'):'';
 return `<h2>Road appearance survey · REVIEW ONLY</h2><p>Original colour · frame ${report.timestamp_seconds.toFixed(3)} s · ${report.analysis_size.join(' × ')} analysis pixels</p>
 <p class="hint"><strong>Review-only appearance proposals.</strong> A yellow vehicle roof, glare or pavement may pass. These fragments cannot configure road, crossing, stop-line or approach geometry and cannot supply incident-rule evidence. ${report.status?'':'This legacy survey has no machine-readable eligibility record; it remains review only.'}</p>
 <div class="survey-metrics"><div><strong>${s.coverage_percent.toFixed(2)}%</strong><span>Candidate coverage in selection</span></div><div><strong>${s.fragments}</strong><span>Connected fragments</span></div><div><strong>${range}</strong><span>${contrastProbes?'Measured coverage across local-contrast probes':'Coverage across ±20 threshold test'}</span></div></div>
 <p>Mask agreement across ${contrastProbes?'local-contrast probes':'thresholds'}: ${agreement===null?'not available (both empty)':Number.isFinite(agreement)?(100*agreement).toFixed(1)+'% IoU':'not recorded'}. ${contrastProbes?'This measures processing sensitivity, not accuracy or certainty. Identical masks do not establish semantic paint recognition.':'This measures threshold stability, not detection accuracy.'}</p>${probeDetails}
 <details open><summary>Interpretation and limits</summary><ul>${report.limitations.map(t=>'<li>'+escapeHTML(t)+'</li>').join('')}</ul></details>
 <details><summary>How IVP produced these measurements</summary><p>${localSupport?'Original RGB → selected road region → HSV colour support + local median brightness background → positive local contrast → opening / closing → split sparse joined bars → rotated shape filtering → '+(flanks?'two-sided darker-flank support → ':'')+'connected components, area and orientation.':'Original RGB → selected road region → HSV white/yellow segmentation → opening / closing → connected components, area and orientation.'} ${contrastProbes?'A background-relative contrast gate can retain paint under shadow. Repeat at 0.75× and 1.25× the measured local-contrast threshold to inspect sensitivity.':'Repeat at lower and higher brightness thresholds to expose sensitivity.'} ${localSupport?'The evidence download includes background, local contrast and rejected-seed masks.':''} ${flanks?'Raw colour seeds, pre-flank shape proposals, unbracketed proposals and per-component flank samples are retained. Unbracketed candidates are withheld from coverage; rejected pixels do not prove paint is absent.':''}</p><pre>${escapeHTML(JSON.stringify({parameters:report.parameters,...(localSupport?{support:report.support}:{})},null,2))}</pre></details>
 <details><summary>Fragment measurements (${s.fragments}; showing up to 100)</summary><div class="survey-table"><table><thead><tr><th>ID</th><th>Area px</th><th>Long side px</th><th>Short side px</th><th>Angle °</th><th>Clipped by ROI</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${r.id}</td><td>${r.area_px}</td><td>${r.long_side_px.toFixed(1)}</td><td>${r.short_side_px.toFixed(1)}</td><td>${r.orientation_deg_clockwise_from_right===null?'ambiguous':r.orientation_deg_clockwise_from_right.toFixed(1)}</td><td>${r.touches_roi_boundary?'yes':'no'}</td></tr>`).join('')}</tbody></table></div><p>Distances use pixel-centre geometry at analysis resolution; angle is clockwise from image right. CSV contains every fragment and source-image centres. No metre calibration is applied.</p></details>`;
}
async function runSurvey(){
 if(!state.region||surveying)return;
 const revision=surveyRevision;const request={...payload(),rectangle:[...state.region],capture_presentation:ivpPage()};surveying=true;surveyActions();
 $('survey-status').textContent='Measuring original pixels and saving evidence…';
 try{
  const context=typeof captureFeatureSubmission==='function'?captureFeatureSubmission():null;
  const run=await api('/api/survey',request);if(context)bindFeatureSubmission(run,context);
  if(revision!==surveyRevision){notify('Survey saved for the earlier selection. Find it in Saved runs.');return;}
  if(ivpPage()&&run.demonstration)renderDemonstration($('survey-result'),run);
  else $('survey-result').innerHTML=surveySummary(run.report)+`<img class="survey-overlay" src="${escapeHTML(run.preview.url)}" alt="Selected road with candidate pixels in cyan and numbered fragments"><div class="downloads">${links(run.downloads)}</div>`;
  $('survey-result').hidden=false;$('survey-status').textContent='REVIEW ONLY · Survey saved. Cyan pixels are appearance proposals; numbered boxes match the CSV. Geometry and incident-rule eligibility are false.';
  $('survey-result').scrollIntoView({behavior:'smooth',block:'start'});
 }catch(e){$('survey-status').textContent=e.message;notify(e.message,true);}
 finally{surveying=false;surveyActions();}
}

let lastMotionRun=null, startingMotion=false;
function motionModeChanged(){
 document.querySelector('.target-setup').hidden=$('motion-mode').value==='vehicles';
 $('motion-mode-note').textContent=$('motion-mode').value==='vehicles'?'Whole-vehicle boxes from local YOLO11s; ByteTrack with bottom-centre continuity, IVP camera correction and speed. No LLM. Auto-rickshaw types are not classified. New tracks require at least four observations and 0.3 seconds before appearing.':'No AI model runs. Background subtraction discovers foreground regions; corner tracking follows their texture. A track may cover part of a vehicle, several touching vehicles, or another moving object. Use this mode to study motion and foreground activity; whole-vehicle tracking is the separate detector + IVP mode.';
 if($('motion-mode').value==='vehicles')clearTarget();
}
function motionActions(){
 paintTrackingSource();
 const video=state.media?.kind==='video';$('motion-options').hidden=!video;
 $('speed-mark').disabled=!video||!state.preview||state.rendering||state.pending;
 const unavailable=$('motion-mode').value==='vehicles'&&state.config?.vehicle_tracking_ready===false;
 $('motion-run').disabled=!video||!state.preview||state.rendering||state.pending||startingMotion||unavailable;
 const incident=typeof IVP_INCIDENT_FEATURES!=='undefined'&&ivpPage()&&IVP_INCIDENT_FEATURES.includes(workbenchTask);
 $('motion-note').textContent=unavailable?'The local vehicle detector is unavailable. Recheck its installation or choose moving texture regions for IVP-only tracking.':!video?'Choose a video above. Still images cannot show movement.':incident?'Complete the setup above, inspect the preview, then confirm your review. Unknown context produces withheld conclusions.':'No per-vehicle selection needed. Fixed, elevated cameras work best. Speed appears in px/s; measured road calibration enables km/h.';
}
function paintMotionLine(){
 const line=$('motion-count-line'),axis=$('motion-axis').value,pos=Number($('motion-position').value);
 $('motion-line-value').textContent=pos+'%';line.hidden=state.area!=='ivp'||state.media?.kind!=='video';
 line.style.cssText=axis==='horizontal'?`left:0;top:${pos}%;width:100%;height:2px`:`left:${pos}%;top:0;height:100%;width:2px`;
 paintIncidents();
}
async function startMotion(){
 try{
  if((typeof workbenchTask==='undefined'||workbenchTask==='tracking')&&$('motion-use-roi').checked&&!state.region)throw Error('Select a road region first, or uncheck the region option.');
  if(typeof beginFeatureAttempt==='function')beginFeatureAttempt();startingMotion=true;motionActions();
  const source=state.media;
  const request=typeof scopedTrafficRequest==='function'?scopedTrafficRequest(source):{media:source.id,source_revision:source.source_revision,axis:$('motion-axis').value,position:Number($('motion-position').value)/100,rectangle:$('motion-use-roi').checked?state.region:null,calibration:speedCalibration(),target_rectangle:$('motion-mode').value==='ivp'?targetRectangle:null,tracking_mode:$('motion-mode').value,output_view:$('motion-view').value,incidents:incidentConfig()};

  const context=typeof captureFeatureSubmission==='function'?captureFeatureSubmission():null;
  const run=await api('/api/motion',request);if(context)bindFeatureSubmission(run,context);
  if(state.media?.id===source.id&&state.media?.source_revision===source.source_revision&&source.example?.synthetic===true)$('synthetic-status').textContent=`Run submitted for ${source.name}. ${request.calibration?'Uses declared virtual geometry, not measured real-world speed.':'No virtual calibration applied: speed stays in px/s.'} Optional TEST threshold review is ${request.incidents?.enabled?'on':'off'}. Follow the processing status and saved result.`;
  showJob(run);$('active-job').scrollIntoView({behavior:'smooth',block:'center'});notify('Automatic tracking started. The video and speed measurements will appear when ready.');
 }catch(e){notify(e.message,true);}finally{startingMotion=false;motionActions();}
}
let selectingTarget=false, targetRectangle=null, targetStart=null;
function clearTarget(){selectingTarget=false;targetRectangle=null;targetStart=null;drawTarget();}
function drawTarget(){const box=$('target-box');box.hidden=!targetRectangle;if(targetRectangle){const [x,y,x2,y2]=targetRectangle;box.style.cssText=`left:${x*100}%;top:${y*100}%;width:${(x2-x)*100}%;height:${(y2-y)*100}%`;$('target-status').textContent='Selected target ready. Track positions and speed will be exported under ID SELECTED.';}else $('target-status').textContent='No selected target.';}
let calibrationPoints=[], calibrating=false, calibrationSource=null;
function clearCalibration(){calibrationPoints=[];calibrating=false;calibrationSource=null;$('speed-enabled').checked=false;$('speed-width').value='';$('speed-length').value='';drawCalibration();}
function drawCalibration(){
 if(typeof syncSpeedLimit==='function')syncSpeedLimit();
 const points=calibrationPoints.map(p=>p.map(v=>v*100).join(',')).join(' ');
 $('speed-overlay').innerHTML=(calibrationPoints.length>1?`<polygon points="${points}" fill="#c88cff22" stroke="#c88cff" stroke-width=".4"/>`:'')+calibrationPoints.map((p,i)=>`<circle cx="${p[0]*100}" cy="${p[1]*100}" r=".8" fill="#c88cff"/><text x="${p[0]*100+1}" y="${p[1]*100-1}" font-size="3" fill="#fff">${i+1}</text>`).join('');
 $('frame').classList.toggle('calibrating',calibrating);
 $('speed-status').textContent=calibrating?`Click corner ${calibrationPoints.length+1} of 4: ${['far-left','far-right','near-right','near-left'][calibrationPoints.length]}.`:calibrationPoints.length===4?($('speed-enabled').checked?'Four corners selected. Enter measured dimensions before analysing.':'Calibration disabled. Region estimates use px/s.'):'No calibration. Region estimates use px/s.';
}
function speedCalibration(){
 if(!$('speed-enabled').checked)return null;
 if(calibrationSource?.media_id!==state.media?.id||calibrationSource?.source_revision!==state.media?.source_revision)throw Error('Mark the road corners on this video before enabling km/h.');
 if(calibrationPoints.length!==4)throw Error('Mark all four road corners before enabling km/h.');
 const width=Number($('speed-width').value),length=Number($('speed-length').value);
 if(!(width>=.1&&width<=1000&&length>=.1&&length<=1000))throw Error('Enter measured road width and length (0.1–1000 metres).');
 return {...calibrationSource,points:calibrationPoints,width_m:width,length_m:length};
}
function syntheticActions(){
 const ready=!workspaceRestoring&&!state.pending&&!state.rendering;
 $('synthetic-open').disabled=!ready;
 $('synthetic-apply').disabled=!ready||state.media?.kind!=='video'||!state.media?.source_revision||state.media?.example?.synthetic!==true||state.media?.example?.virtual_setup?.kind!=='fixed_translation_fixture_v1';
}
async function openSyntheticExample(){await sample($('synthetic-choice').value);}
function applySyntheticSetup(){
 const media=state.media,setup=media?.example?.virtual_setup;
 if(media?.kind!=='video'||media?.example?.synthetic!==true||setup?.kind!=='fixed_translation_fixture_v1'||setup.threshold_is_legal_limit!==false||!media.source_revision)throw Error('Open one of the four known synthetic examples before applying virtual geometry. Real footage is never calibrated by this helper.');
 const text=$('synthetic-threshold').value.trim(),threshold=Number(text),review=$('synthetic-review').checked;
 if(!text||!Number.isFinite(threshold)||threshold<1||threshold>160)throw Error('Enter a finite TEST threshold between 1 and 160 virtual km/h. This is not a legal speed limit.');
 if(review&&!state.config.vehicle_tracking_ready)throw Error('Vehicle mode is unavailable. Turn off threshold review to use the IVP-only virtual calibration.');
 clearCalibration();
 $('synthetic-review').checked=review;
 calibrationPoints=setup.points.map(point=>[...point]);
 calibrationSource={media_id:media.id,source_revision:media.source_revision,reference_frame:0};
 $('speed-width').value=String(setup.width_m);$('speed-length').value=String(setup.length_m);$('speed-enabled').checked=true;
 const feature=typeof workbenchTask==='undefined'?'tracking':workbenchTask;
 if(['tracking','speed_review'].includes(feature))$('incident-limit').value=String(threshold);
 $('incident-synthetic').checked=true;
 if(review)$('incident-enabled').checked=true;
 $('incident-calibration-confirmed').checked=false;$('incident-real-time').checked=false;
 if($('feature-context-confirmed'))$('feature-context-confirmed').checked=false;
 if(review){$('motion-mode').value='vehicles';motionModeChanged();}
 drawCalibration();drawRegion();paintMotionLine();paintIncidents();
 $('synthetic-status').textContent=`Virtual ${setup.width_m} × ${setup.length_m} m setup applied to ${media.name}. TEST threshold ${threshold} virtual km/h. This is not a legal speed limit. Inspect the corners and confirm dimensions and timing yourself. Other feature geometry and conditions are retained. No video job or inference has started.`;
 if(Number($('timestamp').value)!==0){$('timestamp').value='0';schedule();}
 motionActions();
}
function clearIncidents(){
 if(typeof clearRedReviewGeometry==='function')clearRedReviewGeometry();
 if($('feature-context-confirmed'))$('feature-context-confirmed').checked=false;
 for(const id of ['incident-enabled','incident-calibration-confirmed','incident-real-time','incident-synthetic'])$(id).checked=false;
 $('incident-condition').value=typeof redPage==='function'&&redPage()?'unknown':'observe';$('incident-intervals').value='';$('incident-restricted').checked=false;$('incident-limit').value='';
 if($('synthetic-review'))$('synthetic-review').checked=false;
 if($('synthetic-status'))$('synthetic-status').textContent='TEST values are not verified legal limits. Open a synthetic example, then apply its setup explicitly. Use the separate tracking button when ready.';
 paintIncidents();
}
function incidentConfig(){
 if(typeof featureIncidentConfig==='function'&&ivpPage()&&IVP_INCIDENT_FEATURES.includes(workbenchTask))return featureIncidentConfig(workbenchTask);
 if(!$('incident-enabled').checked)return null;
 const rectangle=id=>{const v=$(id).value.split(',').map(x=>Number(x.trim())/100);if(v.length!==4||v.some(n=>!Number.isFinite(n)||n<0||n>1)||v[2]-v[0]<.02||v[3]-v[1]<.02)throw Error('Enter a rectangle as left, top, right, bottom percentages, at least 2% wide/high.');return v;};
 const start=Number($('incident-start').value)/100,end=Number($('incident-end').value)/100,pos=Number($('motion-position').value)/100;
 if($('incident-line').checked&&(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end>1||end-start<.02))throw Error('Incident line end must be at least 2% beyond its start, within 0–100%.');
 const intervals=$('incident-intervals').value.trim().split('\n').filter(x=>x.trim()).map(row=>{const parts=row.split(',').map(s=>s.trim());if(parts.length!==3||parts[0]===''||parts[1]==='')throw Error('Condition timeline needs start, end, state on each line.');return {start:Number(parts[0]),end:Number(parts[1]),state:parts[2]};});
 return {enabled:true,line:$('incident-line').checked?($('motion-axis').value==='horizontal'?[[start,pos],[end,pos]]:[[pos,start],[pos,end]]):null,
  direction:$('incident-direction').value,condition:$('incident-condition').value,condition_intervals:intervals,
  zone:$('incident-zone').checked?rectangle('incident-zone-rect'):null,zone_role:$('incident-restricted').checked?'restricted':'observe',
  approach_zone:rectangle('incident-approach'),speed_limit_kmh:$('incident-limit').value===''?null:Number($('incident-limit').value),
  calibration_confirmed:$('incident-calibration-confirmed').checked,real_time_confirmed:$('incident-real-time').checked,synthetic:$('incident-synthetic').checked};
}
function paintIncidents(){
 const container=$('frame');if(!container)return;
 let overlay=$('incident-overlay');if(!overlay){overlay=document.createElementNS('http://www.w3.org/2000/svg','svg');overlay.id='incident-overlay';overlay.setAttribute('viewBox','0 0 100 100');overlay.setAttribute('preserveAspectRatio','none');overlay.setAttribute('aria-label','Configured incident geometry');container.appendChild(overlay);}
 const red=typeof redPage==='function'&&redPage();
 overlay.innerHTML='';overlay.style.display=((!red&&!$('incident-enabled').checked)||state.area!=='ivp')?'none':'block';
 if(!red&&!$('incident-enabled').checked){$('incident-status').textContent='Off. Existing tracking and speed analysis are unchanged.';return;}
 try{const draft=typeof featureIncidentConfig==='function'&&ivpPage()&&IVP_INCIDENT_FEATURES.includes(workbenchTask);const c=draft?featureIncidentConfig(workbenchTask,true):incidentConfig();let shapes='';if(c.line)shapes+=`<line x1="${c.line[0][0]*100}" y1="${c.line[0][1]*100}" x2="${c.line[1][0]*100}" y2="${c.line[1][1]*100}" stroke="#fa783c" stroke-width=".6"/>`;
 for(const [rect,color] of [[c.zone,'#a078fa'],[draft||c.condition==='stop_required'?c.approach_zone:null,'#50d2d2']])if(rect)shapes+=`<rect x="${rect[0]*100}" y="${rect[1]*100}" width="${(rect[2]-rect[0])*100}" height="${(rect[3]-rect[1])*100}" fill="none" stroke="${color}" stroke-width=".4"/>`;
 if(red){if(!redBound())shapes='';shapes+=redAnnotation(redBound()?c:{});}
 overlay.innerHTML=shapes;overlay.setAttribute('aria-label',draft&&!$('feature-context-confirmed').checked?'Unreviewed draft geometry':'Configured incident geometry');$('incident-status').textContent=`${draft&&!$('feature-context-confirmed').checked?'UNREVIEWED DRAFT · inspect this geometry before confirming. ':''}${c.synthetic?'Synthetic test. ':''}Geometry belongs to the first frame. ${Number($('timestamp').value)!==0?'Return to frame zero to verify placement. ':''}${red?'Your signal context is user-declared. No road calibration is needed for this check.':'Speeding and missed-stop review withhold conclusions without confirmed calibration and real-time footage.'}`;
 }catch(e){$('incident-status').textContent=e.message;}
}
function evidencePanel(run){
 if(run.status.state!=='complete'||!run.report.motion?.incidents?.enabled)return '';
 return `<section class="evidence-panel" data-evidence-run="${escapeHTML(run.id)}"><div class="eyebrow">SAVED EVIDENCE · LOCAL REVIEW</div><h3>Understand these assessments</h3><p>Read the recorded facts, uncertainty and questions to check. Original statuses and measurements remain attached to every assessment.</p><p class="hint ivp-page-only">Saved evidence is read-only here. Open Full application to build a review or use an optional model.</p><label class="evidence-model" data-full-app-only><input type="checkbox" class="evidence-use-model"> Ask an optional local model to prioritise existing facts and questions</label><p class="hint" data-full-app-only>No footage is sent to the model. It receives text facts, including recorded calibration context. The full offline review remains usable if the local model is unavailable.</p><button class="quiet evidence-build" data-full-app-only>Build evidence review</button><p class="evidence-status hint" role="status">No review requested.</p><p class="evidence-integrity hint" role="status"></p><div class="evidence-navigation" hidden><label>Assessments <select class="evidence-filter" aria-label="Filter saved assessments"><option value="all">All assessments</option><option value="candidate">Candidates</option><option value="observation">Observations</option><option value="withheld">Withheld</option></select></label><div><button class="quiet evidence-prev">← Previous</button><span class="evidence-page-label" role="status"></span><button class="quiet evidence-next">Next →</button></div><p class="hint">Source export order is preserved. Event times may go backwards.</p></div><div class="evidence-content"></div></section>`;
}
function evidenceCitations(fact){
 if(!Array.isArray(fact.citations)||!fact.citations.length)return '';
 return `<details class="fact-citations"><summary>Source references (${fact.citations.length})</summary><ul>${fact.citations.map(ref=>`<li><code>${escapeHTML(fact.pointer_base||'event')}:${escapeHTML(ref.pointer)}</code> · ${ref.recorded===false?'Unknown — not recorded':escapeHTML(JSON.stringify(ref.value))}</li>`).join('')}</ul></details>`;
}
function evidenceFacts(item){
 const secondary=new Set(['status','timestamp','type','synthetic','source','detector_score','metric_context','evidence_timestamp','crossing_bracket','track_context']);
 const focus=new Set(item.language_model?.fact_ids||[]);
 const list=facts=>`<ul>${facts.map(f=>`<li class="${focus.has(f.id)?'model-focus':''}">${escapeHTML(f.text)}${evidenceCitations(f)}</li>`).join('')}</ul>`;
 const primary=item.facts.filter(f=>!secondary.has(f.id)),support=item.facts.filter(f=>secondary.has(f.id));
 return list(primary)+(support.length?`<details><summary>Supporting facts and provenance (${support.length})</summary>${list(support)}</details>`:'');
}
function evidenceReviewSummary(result){
 const review=result.review;if(!review)return '';
 const counts=review.summary.statuses,integrity=result.source_integrity||{};
 const models=result.model_states||{};
 const g=review.grounding;
 const grounding=g?.state==='source_citations_checked'?`<details class="grounding-context"><summary>Recorded source-reference check</summary><p>At publication: ${Number(g.event_count)} source assessments, ${Number(g.fact_count)} facts and ${Number(g.citation_count)} references checked. ${Number(g.unrecorded_field_citations)} references point to unrecorded fields.</p><p class="hint">${escapeHTML(g.limitation)}</p></details>`:'';
 const page=review.pagination;
 const range=page?`Showing ${page.start}–${page.end} of ${page.filtered_total} matching assessments (${review.summary.event_count} total).`:`Showing ${review.indexed_events} saved cards of ${review.summary.event_count} assessments. This older snapshot has no page navigation; build a fresh review to inspect every assessment in-app.`;
 return `<div class="review-summary"><strong>${review.summary.event_count} saved assessments</strong> · ${counts.candidate||0} candidates · ${counts.observation||0} observations · ${counts.withheld||0} withheld<p>${review.summary.synthetic_events} labelled synthetic · ${review.summary.unknown_provenance_events} with unrecorded provenance</p></div>${grounding}<p class="${integrity.state==='changed'?'error':'hint'}">${escapeHTML(integrity.message||'Source fingerprints are recorded in the full JSON.')}</p>${models.unavailable?'<p class="hint">Local model unavailable. The complete offline evidence review is retained.</p>':''}${review.warnings.map(w=>`<p class="error">${escapeHTML(w.text)}</p>`).join('')}<p class="hint">${escapeHTML(range)} The full JSON retains all events and bounded CSV excerpts.</p>${page?.filtered_total===0?'<p>No assessments match this status.</p>':''}${review.events.map(item=>{const e=item.event,focus=item.language_model||{},facts=new Set(focus.fact_ids||[]),questions=new Set(focus.question_ids||[]);return `<article class="evidence-assessment">${item.thumbnail_url?`<img loading="lazy" src="${escapeHTML(item.thumbnail_url)}" alt="Original evidence for ${escapeHTML(e.event_id)}">`:`<p class="hint">${escapeHTML(item.thumbnail_notice||'No exported thumbnail is available for this assessment.')}</p>`}<h4>${escapeHTML(e.status)} · ${escapeHTML(e.type.replaceAll('_',' '))}${e.synthetic===true?' · SYNTHETIC':''}</h4><p>${Number(e.timestamp_seconds).toFixed(2)} s · track ${escapeHTML(e.track_id??'unrecorded')} · ${escapeHTML(e.event_id)}${Number.isInteger(item.source_position)?' · source assessment '+item.source_position:''}</p>${focus.state==='selected'?'<p class="hint">Local model focus marks existing evidence below. All recorded facts remain available below.</p>':focus.state==='unavailable'?'<p class="hint">Model focus unavailable; showing offline facts.</p>':''}${e.measurements?.stop_evidence?stopEvidenceSummary(e.measurements.stop_evidence):''}${evidenceFacts(item)}${item.warnings.map(w=>`<p class="review-warning">${escapeHTML(w.text)}</p>`).join('')}<details><summary>Questions to check</summary><ul>${item.review_questions.map(q=>`<li class="${questions.has(q.id)?'model-focus':''}">${escapeHTML(q.text)}</li>`).join('')}</ul></details><details><summary>Original assessment and citation</summary><p>${escapeHTML(item.evidence_ref.file)} · ${item.evidence_ref.line?'line '+Number(item.evidence_ref.line):escapeHTML(item.evidence_ref.pointer||'')}</p><pre>${escapeHTML(JSON.stringify(e,null,2))}</pre></details></article>`;}).join('')}<details><summary>Source fingerprints and review limits</summary><ul>${review.limitations.map(text=>`<li>${escapeHTML(text)}</li>`).join('')}</ul><pre>${escapeHTML(JSON.stringify(review.source.files,null,2))}</pre></details><div class="downloads">${links(result.downloads)}</div>`;
}
function bindEvidencePanel(panel){
 if(!panel)return;
 const button=panel.querySelector('.evidence-build'),model=panel.querySelector('.evidence-use-model'),status=panel.querySelector('.evidence-status'),content=panel.querySelector('.evidence-content');
 const integrityNotice=panel.querySelector('.evidence-integrity');
 const navigation=panel.querySelector('.evidence-navigation'),filter=panel.querySelector('.evidence-filter'),previous=panel.querySelector('.evidence-prev'),next=panel.querySelector('.evidence-next'),pageLabel=panel.querySelector('.evidence-page-label');
 let timer=null,renderedVersion=null,generation=0,current=null,pageCursor=null,statusFilter='all';
 function updateNavigation(result){
  if(!navigation)return;
  const page=result?.review?.pagination;navigation.hidden=!page||result.artifact_integrity?.state==='changed';
  if(!page)return;
  filter.disabled=false;filter.value=page.status_filter;previous.disabled=page.start<=1;next.disabled=!page.next_cursor;
  pageLabel.textContent=`${page.start}–${page.end} of ${page.filtered_total}`;
 }
 function show(result){
  if(!panel.isConnected)return;
  current=result;clearTimeout(timer);status.textContent=result.message||(result.state==='none'?'No review requested.':'Preparing review…');
  if(integrityNotice){integrityNotice.textContent=result.artifact_integrity?.message||'';integrityNotice.className=result.artifact_integrity&&result.artifact_integrity.state!=='verified'?'evidence-integrity review-warning':'evidence-integrity hint';}
  const active=['queued','running'].includes(result.state);button.disabled=active;model.disabled=active;
  if(result.artifact_integrity?.state==='changed'){content.innerHTML='';renderedVersion=null;pageCursor=null;}
  else if(result.review){pageCursor=result.review.pagination?.current_cursor??null;statusFilter=result.review.pagination?.status_filter||'all';const version=JSON.stringify([result.id,result.review.generated_at,result.review.artifact,result.review.pagination,result.source_integrity,result.artifact_integrity,result.model_states]);if(version!==renderedVersion){content.innerHTML=evidenceReviewSummary(result);renderedVersion=version;}}
  updateNavigation(result);
  if(active){const ticket=generation;timer=setTimeout(()=>poll(result.id,ticket),1500);}
 }
 async function poll(id,ticket=generation){
  const cursor=pageCursor;
  try{const result=await api('/api/evidence-review?id='+encodeURIComponent(id)+'&status='+encodeURIComponent(statusFilter)+(cursor?'&cursor='+encodeURIComponent(cursor):''));if(ticket===generation)show(result);}
  catch(e){if(!panel.isConnected||ticket!==generation)return;if(cursor&&e.status===400){pageCursor=null;renderedVersion=null;content.innerHTML='';return poll(id,ticket);}status.textContent=e.message;button.disabled=false;model.disabled=false;pageCursor=current?.review?.pagination?.current_cursor??null;statusFilter=current?.review?.pagination?.status_filter||'all';updateNavigation(current);}
 }
 async function navigate(cursor,chosenFilter=statusFilter){
  if(!current?.id)return;generation++;clearTimeout(timer);pageCursor=cursor;statusFilter=chosenFilter;status.textContent='Loading saved assessments…';if(navigation){previous.disabled=true;next.disabled=true;pageLabel.textContent='Loading…';}return poll(current.id,generation);
 }
 if(navigation){previous.onclick=()=>navigate(current.review.pagination.previous_cursor);next.onclick=()=>navigate(current.review.pagination.next_cursor);filter.onchange=()=>navigate(null,filter.value);}
 button.onclick=async()=>{const ticket=++generation;clearTimeout(timer);pageCursor=null;statusFilter='all';button.disabled=true;model.disabled=true;if(navigation){previous.disabled=true;next.disabled=true;filter.disabled=true;}status.textContent='Preparing an evidence snapshot…';try{const result=await api('/api/evidence-review',{source_run:panel.dataset.evidenceRun,use_model:model.checked});if(ticket===generation)show(result);}catch(e){if(ticket===generation){status.textContent=e.message;button.disabled=false;model.disabled=false;updateNavigation(current);}}};
 const restoring=generation;api('/api/evidence-review?source='+encodeURIComponent(panel.dataset.evidenceRun)).then(result=>{if(generation===restoring)show(result);}).catch(e=>{if(generation===restoring)status.textContent=e.message;});
}
function stopEvidenceSummary(proof){
 if(!proof||!['start_seconds','end_seconds','duration_seconds','observed_samples','position_diameter_bound_m','fitted_speed_kmh'].every(k=>Number.isFinite(proof[k]))||proof.duration_seconds<1-1e-6||!Number.isInteger(proof.observed_samples)||proof.observed_samples<8||proof.end_seconds<=proof.start_seconds||proof.position_diameter_bound_m<0||proof.fitted_speed_kmh<0)return '';
 return `<p class="hint stop-measurement">Observed low-motion interval: <strong>${proof.start_seconds.toFixed(2)}–${proof.end_seconds.toFixed(2)} s</strong> · ${proof.observed_samples} observed frames · position envelope ≈ ${proof.position_diameter_bound_m.toFixed(2)} m · fitted speed ${proof.fitted_speed_kmh.toFixed(2)} km/h during this interval.</p>`;
}
function incidentSummary(review){
 if(!review?.enabled)return '';
 const events=review.events||[],counts=review.counts||{};
 return `<section class="incident-review"><h3>${review.configuration.synthetic?'Synthetic incident review':'Incident review'}</h3><p><strong>${counts.candidate||0} review candidates</strong> · ${counts.observation||0} neutral observations · ${counts.withheld||0} withheld assessments.</p><p class="hint">Conditions were configured by the user. These are review candidates, not legal findings. Zero candidates does not establish compliant driving.</p>${review.invalid_reason?`<p class="error">${escapeHTML(review.invalid_reason)}</p>`:''}<details open><summary>Review evidence · first ${Math.min(events.length,20)} of ${review.event_count} assessments</summary><div class="incident-cards">${events.slice(0,20).map(e=>`<article class="incident-card">${e.thumbnail_url?`<img loading="lazy" src="${escapeHTML(e.thumbnail_url)}" alt="Evidence for ${escapeHTML(e.event_id)}">`:''}<strong>${escapeHTML(e.status)} · ${escapeHTML(e.type.replaceAll('_',' '))}</strong><p>${Number(e.timestamp_seconds).toFixed(2)} s · track ${escapeHTML(e.track_id)}${e.synthetic?' · SYNTHETIC':''}</p><p>${escapeHTML(e.reason)}</p>${e.measurements.speed!=null?`<p>${Number(e.measurements.speed).toFixed(2)} ${escapeHTML(e.measurements.speed_unit)}${e.measurements.threshold_kmh!=null?' · threshold '+Number(e.measurements.threshold_kmh)+' km/h':''}</p>`:''}${stopEvidenceSummary(e.measurements.stop_evidence)}</article>`).join('')}</div></details><p class="hint">${escapeHTML(review.thumbnail_policy)} Full timestamps, quality, coordinates, settings and all assessments are in the incident JSONL/CSV and evidence ZIP.</p></section>`;
}
function motionSummary(m){
 const names=m.line.axis==='horizontal'?['Down','Up']:['Right','Left'];
 const total=m.crossings.positive+m.crossings.negative;
 const bars=m.timeline.map((b,i)=>`<rect x="${i/m.timeline.length*600}" y="${b.activity_percent===null?0:100-b.activity_percent}" width="${Math.max(1,600/m.timeline.length)}" height="${b.activity_percent===null?100:b.activity_percent}" fill="${b.activity_percent===null?'#e6e7e2':'#267f76'}"><title>${b.start.toFixed(2)}–${b.end.toFixed(2)}s: ${b.activity_percent===null?'excluded':b.activity_percent.toFixed(2)+'% moving area'}</title></rect>`).join('');
 const speed=m.selected_target?.speed||m.speed;
 const synthetic=m.synthetic===true||m.source_provenance?.synthetic===true||m.incidents?.configuration?.synthetic===true;
 const speedMeaning=m.uses_ai&&!m.selected_target?'Estimates use observed vehicle-box bottom-centre positions; real-world speed accuracy is unverified. Road calibration assumes a flat rectangular patch with measured dimensions and real-time footage (not slow motion or time-lapse).':speed?.meaning;
 const speedInfo=speed?`<div class="speed-report"><h3>${speed.calibrated?(synthetic?'Synthetic speed · declared virtual scale':'Approximate road speed'):'Image motion speed'}</h3><strong>${speed.sample_mean===null?'No stable estimates':speed.sample_mean.toFixed(1)+' '+escapeHTML(speed.unit)}</strong><p>${m.selected_target?'Selected target':m.uses_ai?'Vehicle tracks':'Moving texture regions'}: mean of accepted track samples · ${speed.accepted_track_samples} samples. ${speed.calibrated?(synthetic?'Uses declared synthetic dimensions; this is not a measured real-world speed.':'Uses your measured road patch.'):'No physical calibration: these are pixels/second, not km/h.'} Estimates appear beside stable tracks and in the trajectories CSV.</p><p class="hint">${escapeHTML(speedMeaning)} ${speed.invalid_reason?escapeHTML(speed.invalid_reason):''}</p>${speedOutcomeSummary(speed)}</div>`:'';
 return `${m.direction?directionSummary(m.direction):''}${m.zebra?zebraSummary(m.zebra):''}<h2>${m.uses_ai?'Automatic vehicle tracking + IVP':'IVP foreground and texture tracking'}</h2>${m.tracking?`<p><strong>${m.tracking.confirmed_tracks} confirmed track IDs</strong> · longest ${m.tracking.longest_track_seconds.toFixed(1)} s · ${m.tracking.recovered_after_short_occlusion} short-gap recoveries. IDs are not a verified unique vehicle count.</p>`:''}${m.selected_target?`<p><strong>Selected target:</strong> ${escapeHTML(m.selected_target.status)} · ${m.selected_target.tracked_frames} frames tracked. ${escapeHTML(m.selected_target.limitation)}</p>`:''}${m.tracking?.recoveries_from_repeated_weak_observations?`<p class="hint">Across the full frame, ${m.tracking.recoveries_from_repeated_weak_observations} short gaps recovered through two consecutive observed detections. Competing matches are rejected; each recovery collects a fresh speed window.${m.tracking.recoveries_from_robust_geometry?` ${m.tracking.recoveries_from_robust_geometry} used robust recent-position geometry to recover from prediction drift.`:''}</p>`:''}${m.tracking?.ambiguous_overlap_frames?`<p class="hint">${m.tracking.ambiguous_overlap_frames} frames had overlapping vehicle hypotheses. IDs marked ? have speed and crossing measurements withheld.</p>`:''}${speed?.rejected_position_jumps?`<p class="hint">${speed.rejected_position_jumps} abrupt position changes reset the speed window. Track IDs remain unchanged; speed resumes after enough stable observations.</p>`:''}${!m.uses_ai?'<p class="hint">These are foreground texture regions. A vehicle can produce several tracks, and touching vehicles can share a region. This view measures image motion and occupied foreground area; its track and crossing totals are not vehicle counts.</p>':''}${speedInfo}${typeof speedDisplaySummary==='function'?speedDisplaySummary(m.speed_display):''}<div class="survey-metrics"><div><strong>${total}</strong><span>${m.uses_ai?'Vehicle-candidate':'Moving-region'} line crossings · ${names[0]} ${m.crossings.positive} / ${names[1]} ${m.crossings.negative}</span></div><div><strong>${m.peak_activity_percent.toFixed(1)}%</strong><span>Peak foreground area in analysis region</span></div><div><strong>${m.measured_seconds.toFixed(2)} s</strong><span>Measured footage · ${m.suppressed_frames} frames excluded</span></div></div>
 <p class="hint">${m.ambiguous_frames||0} frames contained large or merged foreground regions. Existing tracks can continue when their evidence remains usable. Crossings can undercount or overcount where traffic overlaps.</p><h3>Movement over time</h3><svg class="activity-chart" viewBox="0 0 600 100" role="img" aria-label="Moving area percentage from 0 to 100 across the clip">${bars}</svg><p class="hint">Height = moving area (0–100%). Grey = no valid measurement. Left to right: 0–${m.timeline.at(-1)?.end.toFixed(2)} s. Hover for timestamps.</p>
 <details><summary>Camera correction and coordinate mapping</summary><p>${m.stabilization?escapeHTML(m.stabilization.method):'No camera correction in this earlier run.'} ${m.stabilization?escapeHTML(m.stabilization.coordinate_frame):''}</p></details><details><summary>How the IVP analysis works</summary><p>Gaussian smoothing → adaptive background model (MOG2) → shadow removal → opening and closing → connected components → automatic Shi-Tomasi corner discovery and forward/backward Lucas-Kanade tracking (IVP-only), or local vehicle proposals and ByteTrack (vehicle mode) → trajectory and line-crossing logic. Forward/backward checked optical flow shows local movement as green arrows, including in merged traffic. It also guards against coherent camera movement. Speed needs at least eight observed positions and 0.5 seconds. Its window extends up to ${Number(speed?.maximum_window_seconds||1.2).toFixed(1)} seconds for this run; uncertain trajectories and abrupt position changes are withheld. A homography maps a measured road patch to metres. Heatmap values accumulate foreground duration per pixel.</p></details>
 <details><summary>What the result can and cannot tell you</summary><ul>${m.limitations.map(t=>'<li>'+escapeHTML(t)+'</li>').join('')}</ul></details>${incidentSummary(m.incidents)}`;
}
function speedOutcomeSummary(speed){
 const counts=speed?.status_counts;
 const unavailable='<p class="hint">Sample outcome totals are unavailable for this run. Inspect the trajectories CSV for per-sample statuses.</p>';
 if(!counts||typeof counts!=='object'||Array.isArray(counts))return unavailable;
 const entries=Object.entries(counts);
 if(entries.some(([,count])=>!Number.isSafeInteger(count)||count<0))return unavailable;
 const total=entries.reduce((sum,[,count])=>sum+count,0),accepted=counts.estimated||0,withheld=total-accepted;
 if(!Number.isSafeInteger(total)||total!==speed.observed_track_samples||accepted!==speed.accepted_track_samples||withheld!==speed.withheld_track_samples)return unavailable;
 if(!total)return '<p class="hint">No track observations reached the speed estimator. Frames without a visible track cannot supply speed.</p>';
 const reasons=entries.filter(([status,count])=>status!=='estimated'&&count>0).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));
 return `<div class="speed-outcomes"><p><strong>${accepted} accepted · ${withheld} withheld</strong> from ${total} observed track samples.</p>${withheld?`<details><summary>Why speed was withheld</summary><ul>${reasons.map(([status,count])=>`<li>${escapeHTML(status)} <span>· ${count} samples</span></li>`).join('')}</ul></details>`:''}<p class="hint">These counts are track samples, not vehicles or an accuracy score. Frames without a visible track are excluded. The trajectories CSV records each sample and its status.</p></div>`;
}

function showMotionResult(run){
 if(typeof mayPresentRun==='function'&&!mayPresentRun(run))return;
 if(typeof motionRunFeature==='function'&&motionRunFeature(run)!==workbenchTask)return;
 if(run.status.state!=='complete'||!run.report.motion||!run.preview?.video||lastMotionRun===run.id)return;
 if($('motion-result').removeAttribute)$('motion-result').removeAttribute('data-previous-result');lastMotionRun=run.id;try{localStorage.setItem('traffic-lab-last-motion',run.id);}catch{}const poster=run.downloads.find(d=>d.name===(run.report.motion.output_view==='tracks'?'motion-tracks.png':'motion-mosaic.png'));const heat=run.downloads.find(d=>d.name==='motion-heatmap.png');
 $('motion-result').innerHTML=`<div class="eyebrow">COMPLETED ${run.report.motion.uses_ai?'VEHICLE + IVP':'IVP'} RUN · ${escapeHTML(run.source)}</div>`+`<button id="motion-play" class="quiet">Play result</button><video controls playsinline preload="metadata" poster="${poster?escapeHTML(poster.url.replace('/download/','/view/')):''}" src="${escapeHTML(run.preview.url)}"></video>`+motionSummary(run.report.motion)+`<p class="hint">Automatic IDs and speed estimates are drawn on the stabilized footage. Amber marks the counting line. The stage images below show the background model, shadow removal, cleaned foreground, optical flow and heatmap. Speed is withheld when a track is uncertain.</p>`+(heat?`<details><summary>Final movement heatmap</summary><img src="${escapeHTML(heat.url.replace('/download/','/view/'))}" alt="Accumulated motion heatmap"></details>`:'')+`<div class="downloads">${links(run.downloads)}</div>`;
 if(run.report.motion.incidents?.enabled){const reviewButton=document.createElement('button');reviewButton.id='motion-evidence-review';reviewButton.className='quiet';reviewButton.textContent='Review saved evidence';reviewButton.onclick=async()=>{await openRun(run.id);if(typeof showSavedResultTab==='function')showSavedResultTab('evidence');$('result-body').querySelector('.evidence-panel')?.scrollIntoView({block:'start',behavior:'smooth'});};$('motion-result').insertBefore(reviewButton,$('motion-result').querySelector('video'));}
 const movie=$('motion-result').querySelector('video'),play=$('motion-play');
 play.onclick=()=>{if(movie.paused)movie.play().catch(e=>notify('Playback failed: '+e.message,true));else movie.pause();};movie.onplay=()=>play.textContent='Pause result';movie.onpause=()=>play.textContent='Play result';
 const stages=run.downloads.filter(d=>['motion-background.png','motion-raw-foreground.png','motion-shadows.png','motion-mask.png','motion-flow.png','motion-tracks.png'].includes(d.name));if(stages.length)$('motion-result').insertAdjacentHTML('beforeend','<details><summary>Inspect the motion-processing stages at peak activity</summary><div class="motion-stage-grid">'+stages.map(d=>`<figure><img src="${escapeHTML(d.url.replace('/download/','/view/'))}" alt="${escapeHTML(d.label)}"><figcaption>${escapeHTML(d.label)}</figcaption></figure>`).join('')+'</div></details>');
 $('motion-result').hidden=false;$('motion-result').scrollIntoView({behavior:'smooth',block:'start'});
}

const workbenchTasks={
 lab:{title:'Image laboratory',category:'IVP LABORATORY',description:'Follow each classical image-processing step.',note:'Explore preparation, edges, segmentation, shapes, colour and frequency. No AI model runs.'},
 tracking:{title:'Tracking & speed',category:'VIDEO ANALYSIS',description:'Automatic tracks, measured motion and one road calibration.',note:'Choose local vehicle detection + IVP, or classical moving texture regions. No per-vehicle selection is needed.'},
 zebra:{title:'Zebra occupancy',category:'IVP · DETECTOR ASSISTED',description:'Review a full crossing band and observe vehicle rectangle overlap.',note:'Explicit local detector action; neutral occupancy, never a legal finding.'},
 direction:{title:'Direction review',category:'IVP · DETECTOR ASSISTED',description:'Compare observed motion with a reviewed expected-travel arrow.',note:'The arrow is a configured assumption, not an inferred legal direction.'},
 red_line:{title:'Line crossing during red',category:'IVP · CONFIGURED VIDEO REVIEW',description:'Draw a boundary and review vehicles that cross it under your declared signal context.',note:'Only this boundary/context is submitted. Signal state is configured, not recognized from pixels.'},
 restricted_red:{title:'Area entry during red',category:'IVP · CONFIGURED VIDEO REVIEW',description:'Mark a restricted area and review vehicles entering it under your declared signal context.',note:'Only this zone/context is submitted. No speed or hidden stop-rule fields apply.'},
 speed_review:{title:'Calibrated speed review',category:'IVP · CONFIGURED VIDEO REVIEW',description:'Compare supported calibrated measurements with an explicit threshold.',note:'Measured road geometry and real-time confirmation required; virtual fixtures remain synthetic. The optional display filter does not change incident decisions.'},
 required_stop:{title:'Required-stop review',category:'IVP · CONFIGURED VIDEO REVIEW',description:'Review an observed approach and finite stop boundary with calibrated motion.',note:'Only this approach/stop context is submitted; incomplete evidence stays withheld.'},
 markings:{title:'Road markings',category:'IVP LABORATORY',description:'Inspect and measure candidate markings in a road patch.',note:'An independent IVP application. Uses original pixels and classical segmentation, without an AI model. Candidates need visual review.'},
 detection:{title:'Detection & rules',category:'AI WORKSPACE · EXPERIMENTAL',description:'Inspect detections and the evidence behind rule assessments.',note:'Cyan boxes identify detections. Green boxes mark reported events. Unknown scene context stays “cannot assess”.'},
 overview:{title:'Scene overview',category:'AI WORKSPACE · EXPERIMENTAL',description:'A local category overview of the original frame.',note:'An optional local vision model. It does not measure positions, speed or traffic-rule decisions. Continue IVP work while it runs.'},
 library:{title:'Saved runs',category:'SAVED WORK',description:'Reopen results, inspect evidence and download your work.'}
};
let workbenchTask='lab',lastWorkbenchTask='lab',workbenchSource=null,runOpenRevision=0;
const lastWorkbenchByPage={full:'lab',ivp:'lab'};
let fullAppRestorePromise=null;

function ivpPage(){return document.body.dataset.appPage==='ivp';}
function workbenchStorageKey(){return ivpPage()?'traffic-lab-ivp-tool':'traffic-lab-tool';}
function savedWorkbench(){try{return localStorage.getItem(workbenchStorageKey())||'lab';}catch{return 'lab';}}
async function restoreFullAppTools(){
 if(ivpPage()||fullAppRestorePromise)return fullAppRestorePromise;
 fullAppRestorePromise=(async()=>{
  try{const id=localStorage.getItem('traffic-lab-last-motion');if(id){const run=await api('/api/run?id='+encodeURIComponent(id));if(!ivpPage())showMotionResult(run);}}catch{}
  if(!ivpPage())await restoreSceneReviews();
 })();
 await fullAppRestorePromise;
 if(ivpPage())fullAppRestorePromise=null;
}
function syncPageAppearance(){
 const dedicated=/^\/ivp\/?$/.test(window.location.pathname);
 document.body.dataset.appPage=dedicated?'ivp':'full';
 document.title=dedicated?'Traffic Lab — IVP Laboratory':'Traffic Lab — Image & Video Studio';
 $('app-brand-scope').textContent=dedicated?'IVP LABORATORY':'IVP + AI WORKBENCH';
 $('app-footer-scope').textContent=dedicated?'IVP':'IVP + AI';
}
function applyPageRoute(focus=false){
 syncPageAppearance();
 selectWorkbench(ivpPage()&&typeof featureFromQuery==='function'&&featureFromQuery()||savedWorkbench(),focus);
 if(!workspaceRestoring&&!ivpPage())restoreFullAppTools();
}
function navigatePage(path){
 if(!['/','/ivp'].includes(path)||window.location.pathname===path)return;
 $('result-dialog').close();
 window.history.pushState(null,'',path);applyPageRoute(true);
}
function setupPageNavigation(){
 syncPageAppearance();
 for(const [id,path] of [['ivp-page-link','/ivp'],['full-page-link','/'],['app-home-link','/']])$(id).onclick=event=>{
  if(event.button!==0||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey)return;
  event.preventDefault();navigatePage(path);
 };
 window.addEventListener('popstate',()=>{$('result-dialog').close();applyPageRoute(true);});
}

function selectWorkbench(requested,focus=false){
 runOpenRevision++;
 $('source-video')?.pause?.();
 const known=Object.hasOwn(workbenchTasks,requested)?requested:'lab';
 const task=ivpPage()&&!['lab','tracking','zebra','direction','red_line','restricted_red','speed_review','required_stop','markings','library'].includes(known)?'lab':known;
 const page=ivpPage()?'ivp':'full';
 workbenchTask=task;document.body.dataset.workbench=task;
 document.body.classList.remove('nav-open');$('nav-toggle').setAttribute('aria-expanded','false');
 if(task==='library'){
  lastWorkbenchTask=lastWorkbenchByPage[page];
  $('workspace').hidden=true;$('library').hidden=false;loadRuns();
 }else{
  lastWorkbenchTask=task;lastWorkbenchByPage[page]=task;setArea(['detection','overview'].includes(task)?'ai':'ivp');
  state.view=task==='lab'?'stages':'processed';
  document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b.dataset.view===state.view);b.setAttribute('aria-pressed',String(b.dataset.view===state.view));});
  $('view-choice').innerHTML='';paint();
  $('survey-task').open=task==='markings';
 }
 if(typeof configureFeatureLayout==='function')configureFeatureLayout();
 if(task==='tracking'&&typeof motionActions==='function')motionActions();
 if(task!=='tracking')$('motion-result').querySelector('video')?.pause();
 paintWorkbench();try{localStorage.setItem(workbenchStorageKey(),task);}catch{}
 if(focus)$('tool-title').focus();
}

function paintTrackingSource(){
 const video=state.media?.kind==='video';
 document.body.dataset.videoSource=String(video);
 const videoTool=typeof isVideoFeature==='function'?isVideoFeature(workbenchTask):workbenchTask==='tracking';document.body.dataset.videoTool=String(videoTool);
 $('tracking-empty').hidden=!videoTool||video;
 $('upload').accept=videoTool?'.mp4,.mov,.avi':'.jpg,.jpeg,.png,.mp4,.mov,.avi';
}

function paintWorkbench(){
 paintTrackingSource();
 paintDemonstrationInput();if(typeof paintFeatureInput==='function')paintFeatureInput();if(typeof updateFeatureUnits==='function')updateFeatureUnits();
 const task=workbenchTasks[workbenchTask];
 $('tool-title').textContent=task.title;$('tool-category').textContent=task.category;$('tool-description').textContent=task.description;
 $('workspace-label').textContent='SELECTED SOURCE · ORIGINAL FILE PRESERVED';
 $('feature-note').textContent=task.note||'';
 if(ivpPage()&&workbenchTask==='lab'){
  $('tool-description').textContent='Choose an output and inspect the operations that produced it.';
  $('feature-note').textContent='The saved demonstration follows one output path. Colour enhancement, region analysis and road surveys are separate.';
 }
 document.querySelectorAll('.app-nav [data-workbench]').forEach(button=>{
  if(button.dataset.workbench===workbenchTask)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');
 });
 if(workbenchSource!==state.media?.id){workbenchSource=state.media?.id;$('source-picker').open=!state.media;}
 const steps=state.preview?.ivp_steps||[];
 const selected=steps[state.step]||steps[0];
 const branches=[...new Set(steps.map(step=>step.branch))];
 const bar=$('stage-branches');bar.hidden=workbenchTask!=='lab'||state.view!=='stages'||!steps.length;
 if(bar.dataset.branches!==JSON.stringify(branches)){
  bar.replaceChildren();bar.dataset.branches=JSON.stringify(branches);
  for(const branch of branches){const button=document.createElement('button');button.type='button';button.className='quiet';button.textContent=branch;button.dataset.branch=branch;button.onclick=()=>selectStage(state.preview.ivp_steps.findIndex(s=>s.branch===branch));bar.append(button);}
 }
 bar.querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.branch===selected?.branch)));
 $('stage-cards').querySelectorAll('[data-step]').forEach(card=>card.hidden=state.view==='stages'&&steps[Number(card.dataset.step)]?.branch!==selected?.branch);
 if(selected)$('stage-board-title').textContent=selected.branch+' · '+steps.filter(step=>step.branch===selected.branch).length+' stages';
}

function setupWorkbench(){
 if(typeof setupFeatureUnits==='function')setupFeatureUnits();
 setupPageNavigation();
 setupDemonstration();
 const tools=document.querySelector('.tools');
 for(const id of ['motion-task','survey-task','ai-panel'])tools.insertBefore($(id),$('processing-controls'));
 $('job-dock').append($('active-job'));
 $('dismiss-job').onclick=()=>{$('active-job').hidden=true;};
 const method=document.createElement('details'),summary=document.createElement('summary');
 summary.textContent='How this tracking method works';method.append(summary,$('motion-mode-note'));
 $('motion-options').insertBefore(method,$('motion-options').children[1]);
 document.querySelectorAll('.app-nav [data-workbench]').forEach(button=>button.onclick=()=>typeof navigateTool==='function'?navigateTool(button.dataset.workbench):selectWorkbench(button.dataset.workbench,true));
 $('nav-toggle').onclick=()=>{const open=document.body.classList.toggle('nav-open');$('nav-toggle').setAttribute('aria-expanded',String(open));if(open)$('app-nav').querySelector('[aria-current="page"]')?.focus();};
 document.addEventListener('keydown',event=>{if(event.key==='Escape'&&document.body.classList.contains('nav-open')){document.body.classList.remove('nav-open');$('nav-toggle').setAttribute('aria-expanded','false');$('nav-toggle').focus();}});
 $('result-dialog').addEventListener('close',()=>{$('result-body').querySelectorAll('video').forEach(movie=>movie.pause());});
}

function showSavedResultTab(name,focus=false){
 const target=['overview','evidence','downloads','record'].includes(name)?name:'overview';
 if(target!=='overview')$('saved-overview').querySelectorAll('video').forEach(movie=>movie.pause());
 document.querySelectorAll('#result-tabs [role="tab"]').forEach(button=>{
  const active=button.dataset.resultTab===target;
  button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;
  $('saved-'+button.dataset.resultTab).hidden=!active;if(active&&focus)button.focus();
 });
}

function organizeSavedResult(run){
 const root=$('result-body');
 const nodes=[...root.children],sections={};
 const context=document.createElement('p');context.className='saved-run-context';
 const synthetic=run.report.synthetic===true||run.report.ai?.synthetic===true||run.report.motion?.synthetic===true||run.report.motion?.source_provenance?.synthetic===true||run.report.motion?.incidents?.configuration?.synthetic===true;
 context.textContent=run.status.state+' · '+run.id+(synthetic?' · SYNTHETIC':'');
 const tabs=document.createElement('div');tabs.id='result-tabs';tabs.className='result-tabs';tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','Saved result sections');
 const labels={overview:'Overview',evidence:'Evidence',downloads:'Downloads',record:'Full record'};
 const body=document.createElement('div');body.className='saved-result-layout';
 for(const [name,label] of Object.entries(labels)){
  const button=document.createElement('button');button.type='button';button.textContent=label;button.id='result-tab-'+name;button.dataset.resultTab=name;button.setAttribute('role','tab');button.setAttribute('aria-controls','saved-'+name);button.onclick=()=>showSavedResultTab(name);
  button.onkeydown=event=>{const names=Object.keys(labels),index=names.indexOf(name);let next;
   if(event.key==='ArrowRight')next=(index+1)%names.length;else if(event.key==='ArrowLeft')next=(index+names.length-1)%names.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=names.length-1;else return;
   event.preventDefault();showSavedResultTab(names[next],true);
  };
  tabs.append(button);
  const panel=document.createElement('section');panel.id='saved-'+name;panel.className='saved-pane';panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',button.id);panel.tabIndex=0;sections[name]=panel;body.append(panel);
 }
 for(const node of nodes){
  const heading=node.querySelector(':scope > summary');
  const name=node.classList.contains('downloads')?'downloads':node.classList.contains('evidence-panel')||node.classList.contains('incident-review')?'evidence':heading?.textContent==='Full processing record'?'record':'overview';
  sections[name].append(node);
 }
 if(ivpPage()&&run.demonstration){sections.overview.replaceChildren();renderDemonstration(sections.overview,run);}
 for(const name of ['evidence','downloads'])if(!sections[name].children.length){const empty=document.createElement('p');empty.className='hint';empty.textContent=name==='evidence'?'No evidence viewer is available for this run. The full processing record remains available.':'Downloads appear after a run completes successfully.';sections[name].append(empty);}
 root.replaceChildren(context,tabs,body);showSavedResultTab('overview');
}

function setSourcePlayback(media){
 const player=$('source-video');if(!player)return;
 player.pause();player.removeAttribute('src');player.load();player.muted=true;
 $('source-playback').open=false;$('source-playback').hidden=!media.original_video;
 if(media.original_video)player.src=media.original_video;
}

let demoTicket=0,demoGenerating=false;
function demoActions(){
 const button=$('ivp-generate');if(button)button.disabled=!state.preview||state.pending||state.rendering||workspaceRestoring||demoGenerating;
}
function clearDemonstration(){
 demoTicket++;const result=$('ivp-demo-result');if(result)result.hidden=true;
 if($('ivp-demo-status'))$('ivp-demo-status').textContent='Choose a result. Its executed steps, before/after images and settings will be saved together.';
 if($('ivp-input-preview'))$('ivp-input-preview').hidden=false;
}
function paintDemonstrationInput(){
 const input=$('ivp-input');if(input&&state.preview)input.src=state.preview.original;
 document.body.dataset.ivpExplore=String($('ivp-explore')?.checked===true);
 demoActions();
}
function setupDemonstration(){
 $('ivp-generate').onclick=generateDemonstration;
 $('ivp-goal').onchange=clearDemonstration;
 $('ivp-explore').onchange=paintDemonstrationInput;
}
async function generateDemonstration(){
 if(!state.preview||state.pending||state.rendering||demoGenerating||workspaceRestoring)return;
 if(typeof beginFeatureAttempt==='function')beginFeatureAttempt();const ticket=++demoTicket,request={...payload(),masks:masks(),ivp_goal:$('ivp-goal').value};demoGenerating=true;demoActions();
 $('ivp-demo-status').textContent='Processing the selected result and saving its actual steps…';
 try{
  const context=typeof captureFeatureSubmission==='function'?captureFeatureSubmission():null;
  const run=await api('/api/export',request);if(context)bindFeatureSubmission(run,context);
  if(ticket!==demoTicket||!ivpPage()||workbenchTask!=='lab'){notify('Result saved for the earlier task. Find it in Saved runs.');return;}
  renderDemonstration($('ivp-demo-result'),run);$('ivp-demo-result').hidden=false;$('ivp-input-preview').hidden=true;
  $('ivp-demo-status').textContent='Saved. Shown steps lie on this output path; compare their visual effects below.';
 }catch(error){if(ticket===demoTicket){$('ivp-demo-status').textContent=error.message;notify(error.message,true);}}
 finally{demoGenerating=false;demoActions();}
}
function demonstrationSummary(demo){
 const s=demo.summary||{},number=value=>Number.isFinite(value)?value.toFixed(2):'not recorded';
 if(demo.goal==='road_appearance')return `${s.fragments} retained fragments · ${number(s.coverage_percent)}% candidate coverage in the selected crop. This is appearance coverage, not accuracy.`;
 if(demo.goal==='regions')return `${s.regions} connected regions of at least ${s.minimum_region_area_px} pixels. These are unclassified intensity regions.`;
 if(demo.goal==='edges')return `${s.edge_pixels} edge pixels. Edges alone do not identify objects or markings.`;
 const m=s.colour_metrics;return m?`Mean brightness ${number(m.original.brightness)} → ${number(m.processed.brightness)} / 255. Highlight fraction ${number(m.original.highlight_fraction*100)}% → ${number(m.processed.highlight_fraction*100)}%. Inspect the image; these changes do not establish better quality.`:'Colour output.';
}
function demonstrationStepHTML(demo,step){
 const inputs=demo.goal==='colour'?['Captured colour image entering this operation']:step.inputs;
 const params=Object.entries(step.parameters||{}).filter(([key])=>key!=='meaning').map(([key,value])=>`<li><strong>${escapeHTML(key.replaceAll('_',' '))}</strong>: ${escapeHTML(typeof value==='object'?JSON.stringify(value):value)}</li>`).join('');
 return `<h3>${escapeHTML(step.title)}</h3><p>${escapeHTML(step.purpose)}</p><p class="hint">Input: ${inputs.map(escapeHTML).join(' + ')}</p><ul class="demo-params">${params}</ul><div class="demo-pair"><figure><img src="${escapeHTML(demo.urls[step.before])}" alt="Before ${escapeHTML(step.title)}"><figcaption>Before this operation</figcaption></figure><figure><img src="${escapeHTML(demo.urls[step.after])}" alt="After ${escapeHTML(step.title)}"><figcaption>Captured output of this operation</figcaption></figure></div><p class="demo-effect">${step.effect.changed_pixels.toLocaleString()} of ${step.effect.total_pixels.toLocaleString()} pixels differ (${step.effect.changed_percent.toFixed(2)}%). ${escapeHTML(demo.metric_notice)}</p>`;
}
function renderDemonstration(root,run){
 if(root.removeAttribute)root.removeAttribute('data-previous-result');
 if(typeof rememberFeatureRun==='function')rememberFeatureRun(run);
 const demo=run.demonstration;if(!demo||run.status.state!=='complete'){root.textContent='Captured steps are unavailable. Existing reports retain their original provenance.';return;}
 root.classList.add('ivp-demonstration');
 const scope=demo.goal==='colour'?'Classical colour processing. Enhancement changes appearance and cannot recover missing detail.':demo.scope;
 const limits=demo.goal==='road_appearance'?`<p class="demo-scope">${escapeHTML(run.report.limitations?.[0]||'Bright objects and reflections can pass.') } Inspect shadowed or missed paint in the original; coverage is not ground-truth road-marking area.</p><details><summary>Interpretation and limits</summary><ul>${(run.report.limitations||[]).map(t=>'<li>'+escapeHTML(t)+'</li>').join('')}</ul></details>`:'';
 root.innerHTML=`<div class="eyebrow">SAVED IVP RESULT · NO AI</div><h2>${escapeHTML(demo.title)}</h2><p>${escapeHTML(demonstrationSummary(demo))}</p><p class="demo-scope">${escapeHTML(scope)}</p>${limits}<p class="demo-run">${escapeHTML(run.source)} · selected frame at ${Number(demo.source_binding.timestamp_seconds).toFixed(3)} s · run ${escapeHTML(run.id)}</p><div class="demo-steps" role="group" aria-label="Contributing executed steps">${demo.steps.map((step,i)=>`<button type="button" data-demo-step="${i}" aria-pressed="false">${i+1}. ${escapeHTML(step.title)}</button>`).join('')}</div><div class="demo-step"></div><p class="hint">${demo.omitted_steps.length} unchanged stage boundaries are omitted from this presentation. Alternative branches are outside this selected output.</p><div class="downloads"><a href="${escapeHTML(demo.result_download)}">Download this result ↓</a>${links(run.downloads)}</div>`;
 const panel=root.querySelector('.demo-step');
 const show=index=>{const step=demo.steps[index];panel.innerHTML=demonstrationStepHTML(demo,step);root.querySelectorAll('[data-demo-step]').forEach((button,i)=>button.setAttribute('aria-pressed',String(i===index)));};
 root.querySelectorAll('[data-demo-step]').forEach((button,i)=>button.onclick=()=>show(i));
 if(demo.steps.length)show(demo.steps.length-1);
 else panel.innerHTML=`<p>No selected colour operation changed this image. The original is retained; no improvement is claimed.</p><img class="demo-single" src="${escapeHTML(demo.urls[demo.final])}" alt="Unchanged colour output">`;
}

init();


// A reviewed crossing band is independent of speed and legal rule geometry.
let zebraDrawing=false, zebraPoints=[], zebraSource=null, zebraReviewed=null, zebraStarting=false;
function clearZebra(){if($('motion-zebra'))$('motion-zebra').checked=false;zebraDrawing=false;zebraPoints=[];zebraSource=null;zebraReviewed=null;drawZebra();}
function zebraMatches(){return !!zebraSource&&zebraSource.media_id===state.media?.id&&zebraSource.source_revision===state.media?.source_revision&&Math.abs(zebraSource.reference_seconds-(state.preview?.timestamp??-1))<1e-6;}
function drawZebra(){
 const svg=$('zebra-overlay');if(!svg)return;
 svg.innerHTML=zebraMatches()?`<polygon points="${zebraPoints.map(p=>p.map(v=>v*100).join(',')).join(' ')}" fill="#efac2920" stroke="#db950f" stroke-width=".5"/>`+zebraPoints.map((p,i)=>`<circle cx="${p[0]*100}" cy="${p[1]*100}" r=".7" fill="#db950f"/><text x="${p[0]*100+1}" y="${p[1]*100-1}" font-size="3">${i+1}</text>`).join(''):'';
 zebraActions();
}
function zebraActions(){
 if(!$('zebra-mark'))return;
 const ready=!!state.preview&&!state.rendering&&!state.pending&&!workspaceRestoring&&!zebraStarting;
 const matches=zebraMatches();$('zebra-overlay').style.display=matches?'':'none';
 $('zebra-mark').disabled=!ready;$('zebra-first').disabled=!ready||state.media?.kind!=='video';
 $('zebra-review').disabled=!ready||!matches||zebraPoints.length!==4||!!zebraReviewed;
 $('zebra-frame').disabled=!ready||!matches||!zebraReviewed||state.config?.vehicle_tracking_ready===false;
 $('zebra-video').disabled=!ready||!matches||!zebraReviewed||state.config?.vehicle_tracking_ready===false||state.media?.kind!=='video'||zebraSource.reference_seconds!==0;
 $('zebra-status').textContent=zebraDrawing?`Click corner ${zebraPoints.length+1} of 4 in order around the full crossing band.`:zebraReviewed&&matches?`Reviewed on this source at ${zebraSource.reference_seconds.toFixed(3)} s. ${state.media?.example?.synthetic?'SYNTHETIC example. ':''}Frame check ready${zebraSource.reference_seconds===0&&state.media?.kind==='video'?' and video tracking available.':'.'}`:zebraPoints.length===4&&matches?'Inspect the full band, then explicitly confirm your review.':zebraSource&&!matches?'Preview/source changed. Mark and review the band on this frame.':'No reviewed crossing band. Paint masks do not configure this feature.';
}
function addZebraPoint(point){
 if(!zebraDrawing||!zebraMatches())return;
 zebraPoints.push(point);
 if(zebraPoints.length===4)zebraDrawing=false;
 drawZebra();
}
function zebraConfig(video=false){
 if(!zebraMatches()||!zebraReviewed)throw Error('Review the crossing band on this source and frame first.');
 if(video&&(state.media?.kind!=='video'||zebraSource.reference_seconds!==0))throw Error('Go to the first video frame, then mark and review the crossing band.');
 return copy(zebraReviewed);
}
async function startZebra(video=false){
 try{
  const region=zebraConfig(video),source=state.media;
  if(typeof beginFeatureAttempt==='function')beginFeatureAttempt();zebraStarting=true;zebraActions();
  const request=video?{media:source.id,source_revision:source.source_revision,axis:'horizontal',position:.65,rectangle:null,calibration:null,target_rectangle:null,tracking_mode:'vehicles',output_view:'tracks',incidents:null,ui_feature:'zebra',zebra:region}:{media:source.id,source_revision:source.source_revision,timestamp:region.reference_seconds,region};
  const context=typeof captureFeatureSubmission==='function'?captureFeatureSubmission():null;
  const run=await api(video?'/api/motion':'/api/zebra',request);if(context)bindFeatureSubmission(run,context);showJob(run);
  $('active-job').scrollIntoView({behavior:'smooth',block:'center'});notify('Crossing occupancy check started. This reports image-space overlap only.');
 }catch(e){notify(e.message,true);}finally{zebraStarting=false;zebraActions();}
}
function setupZebra(){
 $('zebra-mark').onclick=()=>{
  if(!state.preview||state.pending||state.rendering)return;
  clearZebra();directionDrawing=false;selectingTarget=false;calibrating=false;state.regionTool=false;
  $('frame').classList.remove('selecting');$('region-tool').setAttribute('aria-pressed','false');
  selectStage(0);zebraSource={media_id:state.media.id,source_revision:state.media.source_revision,reference_seconds:state.preview.timestamp};
  zebraDrawing=true;drawZebra();$('viewport').scrollIntoView({behavior:'smooth',block:'center'});
 };
 $('zebra-clear').onclick=clearZebra;
 $('zebra-first').onclick=()=>{clearZebra();$('timestamp').value=0;schedule();};
 $('zebra-review').onclick=()=>{if(zebraMatches()&&zebraPoints.length===4){zebraReviewed={schema_version:1,semantic:'zebra_crossing',reviewed:true,...zebraSource,points:copy(zebraPoints)};zebraActions();}};
 $('zebra-frame').onclick=()=>startZebra(false);$('zebra-video').onclick=()=>startZebra(true);
}
function zebraSummary(z){
 const counts=z.counts||z.sample_counts||{};
 return `<section class="zebra-summary"><h3>Crossing occupancy · ${z.synthetic?'SYNTHETIC · ':''}image-space observation</h3><p>${escapeHTML(z.notice)}</p><p><strong>${counts.observed||0} observed samples</strong> · ${counts.withheld||0} withheld${z.episodes!==undefined?' · '+escapeHTML(z.episodes)+' occupancy episodes':''}.</p>${z.review_invalidated?'<p class="error">Crossing review invalidated by camera/scene discontinuity. Review fresh geometry before another run.</p>':''}${z.observations?'<ul>'+z.observations.map(r=>`<li>${escapeHTML(r.label||r.status)} · ${escapeHTML(r.vehicle_label)} · ${escapeHTML(r.reason)}${r.vehicle_overlap_fraction!==null?' · '+(r.vehicle_overlap_fraction*100).toFixed(1)+'% of vehicle rectangle overlaps':''}</li>`).join('')+'</ul>':''}<p>Amber “on zebra crossing” labels are neutral. Full metrics, source binding and withheld reasons are in the downloads.</p></section>`;
}

let helmetStarting=false;
function helmetActions(){
 if(!$('helmet-frame'))return;
 $('helmet-frame').disabled=!state.preview||state.rendering||state.pending||workspaceRestoring||helmetStarting||state.config?.helmet?.ready!==true;
 $('helmet-status').textContent=helmetStarting?'Submitting this frame…':state.config?.helmet?.message||'Helmet models unavailable. No model download is started by this button.';
}
async function startHelmet(){
 try{
  if(state.config?.helmet?.ready!==true)throw Error(state.config?.helmet?.message||'Helmet models unavailable locally.');
  const source=state.media,stamp=state.preview?.timestamp;
  if(!source||!Number.isFinite(stamp))throw Error('Open a frame first.');
  if(typeof beginFeatureAttempt==='function')beginFeatureAttempt();helmetStarting=true;helmetActions();
  const context=typeof captureFeatureSubmission==='function'?captureFeatureSubmission():null;
  const run=await api('/api/helmet',{media:source.id,source_revision:source.source_revision,timestamp:stamp});if(context)bindFeatureSubmission(run,context);
  showJob(run);$('active-job').scrollIntoView({behavior:'smooth',block:'center'});notify('Helmet observation check started for the displayed frame.');
 }catch(e){notify(e.message,true);}finally{helmetStarting=false;helmetActions();}
}
function helmetSummary(summary){
 const counts=summary.counts||{};
 return `<section class="zebra-summary"><h3>Helmet observations · ${summary.synthetic?'SYNTHETIC · ':''}model assisted</h3><p>${escapeHTML(summary.notice)}</p><p>${counts.helmet_present||0} helmet observed · ${counts.bare_head||0} bare head observed · ${counts.unknown||0} unknown associations.</p>${!summary.motorcycle_candidates?'<p>No motorcycle candidate was detected. This is not proof of helmet compliance.</p>':''}<ul>${(summary.observations||[]).map(r=>`<li>${escapeHTML(r.label)} · ${escapeHTML(r.reason)} · motorcycle #${r.motorcycle_detection_index}, person ${r.person_detection_index===null?'unassociated':'#'+r.person_detection_index}, supporting heads ${r.head_detection_indices.join(', ')||'none'}</li>`).join('')}</ul><p>Indices are zero-based positions in the exported detection list. Neutral model observations only; no green violation boxes.</p></section>`;
}

let directionDrawing=false,directionPoints=[],directionArrow=[],directionSource=null,directionReviewed=null,directionStarting=false;
function directionMatches(){return !!directionSource&&state.media?.kind==='video'&&directionSource.media_id===state.media.id&&directionSource.source_revision===state.media.source_revision&&state.preview?.timestamp===0;}
function clearDirection(){if($('motion-direction'))$('motion-direction').checked=false;directionDrawing=false;directionPoints=[];directionArrow=[];directionSource=null;directionReviewed=null;drawDirection();}
function drawDirection(){
 const svg=$('direction-overlay');if(!svg)return;const all=directionPoints.concat(directionArrow);
 svg.innerHTML=directionMatches()?`<polygon points="${directionPoints.map(p=>p.map(v=>v*100).join(',')).join(' ')}" fill="#c08cf522" stroke="#a05ad5" stroke-width=".5"/>`+(directionArrow.length===2?`<path d="M ${directionArrow[0].map(v=>v*100).join(' ')} L ${directionArrow[1].map(v=>v*100).join(' ')}" stroke="#a05ad5" stroke-width=".7"/>`:'')+all.map((p,i)=>`<circle cx="${p[0]*100}" cy="${p[1]*100}" r=".7" fill="#a05ad5"/><text x="${p[0]*100+1}" y="${p[1]*100-1}" font-size="3">${i<4?i+1:i===4?'TAIL':'TIP'}</text>`).join(''):'';directionActions();
}
function directionActions(){
 if(!$('direction-mark'))return;const ready=!!state.preview&&!state.rendering&&!state.pending&&!workspaceRestoring&&!directionStarting,match=directionMatches();
 $('direction-first').disabled=!ready||state.media?.kind!=='video';$('direction-mark').disabled=!ready||state.media?.kind!=='video'||state.preview.timestamp!==0;
 $('direction-review').disabled=!ready||!match||directionPoints.length!==4||directionArrow.length!==2||!!directionReviewed;
 $('direction-run').disabled=!ready||!match||!directionReviewed||state.config?.vehicle_tracking_ready===false;$('direction-overlay').style.display=match?'':'none';
 $('direction-status').textContent=directionDrawing?(directionPoints.length<4?`Click region corner ${directionPoints.length+1} of 4 in order.`:`Click expected-travel arrow ${directionArrow.length?'TIP':'TAIL'} inside the region.`):directionReviewed&&match?'Reviewed first-frame direction ready. This is a configured assumption, not a verified legal direction.':directionPoints.length===4&&directionArrow.length===2&&match?'Inspect both region and tail-to-tip direction, then explicitly review.':'No current reviewed direction. Go to frame zero, mark a region and a tail-to-tip arrow.';
}
function addDirectionPoint(point){if(!directionDrawing||!directionMatches())return;if(directionPoints.length<4)directionPoints.push(point);else directionArrow.push(point);if(directionArrow.length===2)directionDrawing=false;drawDirection();}
function directionConfig(){if(!directionMatches()||!directionReviewed)throw Error('Review the direction region and arrow on this source at frame zero first.');return copy(directionReviewed);}
async function startDirection(){
 try{const config=directionConfig(),source=state.media;if(typeof beginFeatureAttempt==='function')beginFeatureAttempt();directionStarting=true;directionActions();const context=typeof captureFeatureSubmission==='function'?captureFeatureSubmission():null;const run=await api('/api/motion',{media:source.id,source_revision:source.source_revision,axis:'horizontal',position:.65,rectangle:null,calibration:null,target_rectangle:null,tracking_mode:'vehicles',output_view:'tracks',incidents:null,ui_feature:'direction',direction:config});if(context)bindFeatureSubmission(run,context);showJob(run);notify('Configured-direction video review started.');}
 catch(e){notify(e.message,true);}finally{directionStarting=false;directionActions();}
}
function setupDirection(){
 $('direction-first').onclick=()=>{clearDirection();$('timestamp').value=0;schedule();};$('direction-clear').onclick=clearDirection;
 $('direction-mark').onclick=()=>{if(state.media?.kind!=='video'||state.preview?.timestamp!==0||state.pending||state.rendering)return;clearDirection();zebraDrawing=false;calibrating=false;selectingTarget=false;state.regionTool=false;$('frame').classList.remove('selecting');$('region-tool').setAttribute('aria-pressed','false');selectStage(0);directionSource={media_id:state.media.id,source_revision:state.media.source_revision,reference_seconds:0};directionDrawing=true;drawDirection();$('viewport').scrollIntoView({behavior:'smooth',block:'center'});};
 $('direction-review').onclick=()=>{if(directionMatches()&&directionPoints.length===4&&directionArrow.length===2){directionReviewed={schema_version:1,type:'configured_direction',reviewed:true,...directionSource,points:copy(directionPoints),arrow:copy(directionArrow)};directionActions();}};
 $('direction-run').onclick=startDirection;
 for(const id of ['speed-mark','target-mark','region-tool','survey-select'])$(id).addEventListener('click',()=>{directionDrawing=false;zebraDrawing=false;});
}
function directionSummary(d){return `<section class="zebra-summary"><h3>Configured-direction review · ${d.synthetic?'SYNTHETIC · ':''}video</h3><p>${escapeHTML(d.notice)}</p><p>${d.episodes||0} opposite-direction episodes. Samples: ${escapeHTML(JSON.stringify(d.sample_counts||{}))}.</p>${d.review_invalidated?'<p class="error">Review invalidated by camera/time/size discontinuity.</p>':''}<p>${escapeHTML(d.method)} Full samples and source mappings are in the downloads.</p></section>`;}
