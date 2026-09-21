'use strict';
const IVP_VIDEO_FEATURES=['tracking','direction','red_line','restricted_red','speed_review','required_stop'];
const IVP_INCIDENT_FEATURES=['red_line','restricted_red','speed_review','required_stop'];
let featureLayoutCurrent=null;
const unitRuns=new Map(),unitSubmissions=new Map();
function isVideoFeature(task){return IVP_VIDEO_FEATURES.includes(task);}
function featureFromQuery(){const match=(window.location.search||'').match(/[?&]feature=([^&]+)/);try{return match?decodeURIComponent(match[1]):null;}catch{return null;}}
function navigateTool(task){if(ivpPage()){const url='/ivp?feature='+encodeURIComponent(task);window.history.pushState(null,'',url);}selectWorkbench(task,true);}
function featureIncidentConfig(feature,draft=false){
 const invalid=(message,field)=>{const error=Error(message);error.field=field;throw error;};
 const drawnLine=typeof redRequestLine==='function'?redRequestLine(feature,draft):undefined;
 if(!draft&&!$('incident-enabled').checked&&!(typeof redPage==='function'&&redPage(feature)))invalid('Inspect the setup and confirm your review before starting.','incident-enabled');
 if(!draft&&state.preview?.timestamp!==0)invalid('Return to frame zero to review the feature geometry and context.','incident-frame-zero');
 const rect=id=>{const raw=$(id).value.split(','),p=raw.map(v=>Number(v.trim())/100);if(p.length!==4||raw.some(v=>!v.trim())||p.some(v=>!Number.isFinite(v)||v<0||v>1)||p[2]-p[0]<.02||p[3]-p[1]<.02)invalid('Enter a valid reviewed rectangle as left, top, right, bottom percentages, at least 2% wide and high.',id);return p;};
 const lineFeature=['red_line','required_stop'].includes(feature),pos=Number($('motion-position').value)/100,start=Number($('incident-start').value)/100,end=Number($('incident-end').value)/100;
 if(lineFeature&&!$('incident-line').checked)invalid('Enable and inspect this feature’s finite stop boundary.','incident-line');
 if(feature==='restricted_red'&&!$('incident-zone').checked)invalid('Enable and inspect this feature’s zone before starting.','incident-zone');
 if(lineFeature&&drawnLine===undefined&&(!$('incident-start').value.trim()||!Number.isFinite(start)||start<0||start>.98))invalid('Enter a finite line start from 0 to 98%.','incident-start');
 if(lineFeature&&drawnLine===undefined&&(!$('incident-end').value.trim()||!Number.isFinite(end)||end>1||end-start<.02))invalid('Enter a finite stop line end at least 2% beyond the start, at most 100%.','incident-end');
 if(lineFeature&&drawnLine===undefined&&(!Number.isFinite(pos)||pos<.1||pos>.9))invalid('Set the line position between 10 and 90%.','motion-position');
 const zone=feature==='restricted_red'?rect('incident-zone-rect'):null,approach=feature==='required_stop'?rect('incident-approach'):null;
 const intervals=draft||feature==='speed_review'?[]:$('incident-intervals').value.trim().split('\n').filter(v=>v.trim()).map(row=>{const a=row.split(',').map(v=>v.trim());if(a.length!==3||!a[0]||!a[1]||!Number.isFinite(Number(a[0]))||!Number.isFinite(Number(a[1]))||Number(a[0])<0||Number(a[1])<=Number(a[0]))invalid('Condition rows need finite start seconds, later end seconds, and condition.','incident-intervals');return {start:Number(a[0]),end:Number(a[1]),state:a[2]};});
 const allowed=feature==='required_stop'?['stop_required','unknown']:['red','green','unknown'];
 if(!draft&&feature!=='speed_review'&&!allowed.includes($('incident-condition').value))invalid('This page accepts only its own configured condition or unknown; remove conditions belonging to another feature.','incident-condition');
 if(intervals.some(v=>!allowed.includes(v.state)))invalid('Condition rows accept only this feature’s conditions or unknown.','incident-intervals');
 const threshold=feature==='speed_review'?Number($('incident-limit').value):null;
 if(!draft&&feature==='speed_review'&&(!$('incident-limit').value.trim()||!Number.isFinite(threshold)||threshold<1||threshold>160))invalid('Enter the configured speed threshold (1–160 km/h, or explicitly synthetic virtual km/h).','incident-limit');
 if(!draft&&feature==='restricted_red'&&!$('incident-restricted').checked)invalid('Explicitly confirm that this reviewed zone is restricted during the configured red context.','incident-restricted');
 if(!draft&&(!$('feature-context-confirmed').checked||!$('incident-enabled').checked))invalid('Confirm that this feature context applies to this source after inspecting the draft geometry.','feature-context-confirmed');
 return {enabled:!draft,line:lineFeature?(drawnLine!==undefined?drawnLine:($('motion-axis').value==='horizontal'?[[start,pos],[end,pos]]:[[pos,start],[pos,end]])):null,
  direction:lineFeature?$('incident-direction').value:'both',condition:feature==='speed_review'?'observe':$('incident-condition').value,
  condition_intervals:intervals,zone,
  zone_role:feature==='restricted_red'&&!draft?'restricted':'observe',approach_zone:approach,
  speed_limit_kmh:threshold,calibration_confirmed:['speed_review','required_stop'].includes(feature)&&$('incident-calibration-confirmed').checked,
  real_time_confirmed:['speed_review','required_stop'].includes(feature)&&$('incident-real-time').checked,
  synthetic:$('incident-synthetic').checked};
}
function scopedTrafficRequest(source){
 if(typeof ivpPage==='function'&&ivpPage()&&['tracking','speed_review'].includes(workbenchTask)&&$('speed-limit-enabled')?.checked&&!speedLimitAvailable())throw Error(SPEED_LIMIT_RESTART);
 const feature=workbenchTask,incident=IVP_INCIDENT_FEATURES.includes(feature),metric=['tracking','speed_review','required_stop'].includes(feature);
 if(!isVideoFeature(feature)||feature==='direction')throw Error('Use this feature’s own reviewed action.');
 const calibration=metric?speedCalibration():null;
 if(['speed_review','required_stop'].includes(feature)&&!calibration)throw Error('This feature requires measured road calibration, or an explicitly labelled synthetic virtual setup.');
 const usesLine=['tracking','red_line','required_stop'].includes(feature);
 return {media:source.id,source_revision:source.source_revision,ui_feature:feature,axis:usesLine?$('motion-axis').value:'horizontal',position:usesLine?Number($('motion-position').value)/100:.65,
  rectangle:feature==='tracking'&&$('motion-use-roi').checked?state.region:null,calibration,
  target_rectangle:feature==='tracking'&&$('motion-mode').value==='ivp'?targetRectangle:null,
  tracking_mode:incident?'vehicles':$('motion-mode').value,output_view:incident?'tracks':$('motion-view').value,
  incidents:incident?featureIncidentConfig(feature):null,
  ...(['tracking','speed_review'].includes(feature)&&speedLimitAvailable()?{speed_display:speedLimitRequest(calibration)}:{})};
}
function motionRunFeature(run){
 const m=run.report?.motion;if(!m)return null;
 if(IVP_VIDEO_FEATURES.includes(run.report?.source_info?.ui_feature))return run.report.source_info.ui_feature;
 if(m.direction&&!m.zebra&&!m.incidents?.enabled)return 'direction';
 if(m.zebra&&!m.direction&&!m.incidents?.enabled)return 'zebra';
 const c=m.incidents?.configuration;
 if(c?.enabled){const states=new Set((c.condition_intervals?.length?c.condition_intervals.map(v=>v.state):[c.condition]).filter(v=>v!=='unknown'));if(states.size!==1)return null;const condition=[...states][0];if(condition==='stop_required'&&c.approach_zone&&c.line)return 'required_stop';if(c.zone_role==='restricted'&&c.zone&&condition==='red')return 'restricted_red';if(c.speed_limit_kmh!==null&&c.speed_limit_kmh!==undefined&&!c.line&&!c.zone&&condition==='observe')return 'speed_review';if(c.line&&!c.zone&&['red','green'].includes(condition))return 'red_line';return null;}
 return !m.direction&&!m.zebra?'tracking':null;
}
function configureFeatureLayout(){
 if(typeof restoreRedControls==='function')restoreRedControls();
 const task=workbenchTask,incident=IVP_INCIDENT_FEATURES.includes(task),video=isVideoFeature(task);
 if($('traffic-review-nav'))$('traffic-review-nav').open=incident||['helmet','zebra','direction'].includes(task);
 document.body.dataset.videoTool=String(video);document.body.dataset.motionOutput=String(video||task==='zebra');document.body.dataset.incidentFeature=incident?task:'none';
 if(featureLayoutCurrent!==task){featureLayoutCurrent=task;if($('motion-result'))$('motion-result').hidden=true;if(incident){clearIncidents();$('incident-line').checked=['red_line','required_stop'].includes(task);$('incident-zone').checked=task==='restricted_red';$('incident-condition').value=task==='speed_review'?'observe':'unknown';}}
 const showLabel=(id,show)=>{const e=$(id);if(e)e.closest('label').hidden=!show;};
 for(const id of ['incident-enabled','incident-synthetic'])showLabel(id,incident);
 for(const id of ['incident-line','incident-start','incident-end','incident-direction'])showLabel(id,incident&&['red_line','required_stop'].includes(task));
 for(const id of ['incident-condition','incident-intervals'])showLabel(id,incident&&task!=='speed_review');
 for(const id of ['incident-zone','incident-zone-rect','incident-restricted'])showLabel(id,task==='restricted_red');
 showLabel('incident-approach',task==='required_stop');showLabel('incident-limit',task==='speed_review');
 for(const id of ['incident-calibration-confirmed','incident-real-time'])showLabel(id,['speed_review','required_stop'].includes(task));
 document.querySelector('.incident-setup').open=incident;
 document.querySelector('.incident-setup > summary').textContent=incident?'Required review setup':'Incident review · optional detector-assisted checks';
 const metricRequired=['speed_review','required_stop'].includes(task),speedSetup=document.querySelector('.speed-setup');
 if(speedSetup){if(metricRequired)speedSetup.open=true;speedSetup.querySelector('summary').textContent=metricRequired?'Required · calibrate road dimensions':'Speed measurement · optional calibration for km/h';}
 if($('incident-limit-label'))$('incident-limit-label').textContent=task==='speed_review'?'Required speed threshold · km/h':'Speed threshold (km/h) · optional';
 $('incident-limit').placeholder=task==='speed_review'?'Enter 1–160; use TEST values only for synthetic footage':'Leave empty to skip speeding review';
 for(const option of $('incident-condition').options){const allowed=task==='required_stop'?['stop_required','unknown']:['red','green','unknown'];option.disabled=incident&&task!=='speed_review'&&!allowed.includes(option.value);option.hidden=option.disabled;}
 for(const p of document.querySelectorAll('.incident-setup p')){if(p.textContent.startsWith('Missed-stop'))p.hidden=task!=='required_stop';}
 $('motion-mode').closest('.motion-controls').hidden=incident;
 $('motion-axis').closest('.motion-controls').hidden=!['tracking','red_line','required_stop'].includes(task);
 $('motion-use-roi').closest('label').hidden=task!=='tracking';
 $('synthetic-demo').hidden=!['tracking','speed_review','required_stop'].includes(task);
 if($('synthetic-review'))$('synthetic-review').closest('label').hidden=ivpPage();
 $('motion-demo').hidden=ivpPage();
 if($('motion-intro'))$('motion-intro').textContent=incident?GUIDED_TASKS[task][1]:'Start with classical IVP movement in pixels/second. Vehicle mode is an optional local detector; calibration is optional here.';
 if($('motion-method-help'))$('motion-method-help').hidden=incident;
 $('motion-mode-note').hidden=incident;
 if($('motion-line-help'))$('motion-line-help').hidden=!['tracking','red_line','required_stop'].includes(task);
 for(const option of $('sample').options){const unsuitable=video&&option.value&&!/\.(mp4|mov|avi)$/i.test(option.value);option.hidden=!!unsuitable;option.disabled=!!unsuitable;}
 $('motion-run').textContent=incident?'Run '+workbenchTasks[task].title.toLowerCase():'Track automatically & calculate speed';
 $('motion-task').querySelector('h2').textContent=incident?workbenchTasks[task].title:'Automatic tracking & image speed';
 if(video){$('tracking-empty-title').textContent='Open a video for '+workbenchTasks[task].title.toLowerCase();}
 if(typeof configureRedReview==='function')configureRedReview();
 syncSpeedLimit();
 updateFeatureUnits();
}
const COURSE_UNITS={
 1:['Image fundamentals','Section 3.1 lecture 1; Section 5 lab week 1 (Color to Gray)'],
 2:['Image enhancement','Section 3.1 lectures 3–8; gamma/power-law, smoothing and histogram equalization. CLAHE is an extension.'],
 4:['Morphological image processing','Section 3.1 lectures 11–14; only recorded opening/closing/other executed morphology'],
 5:['Edge Detection','Section 3.1 lectures 15–16; Canny/Sobel/Laplacian when actually executed'],
 6:['Image segmentation','Section 3.1 lectures 17–18; Otsu/connectivity. Adaptive thresholding: Section 5 lab week 10. Component/shape filtering extends connectivity.'],
 7:['Colour Image Processing','Section 3.1 lectures 22–23; HSV/model transforms/colour slicing. No colour-histogram claim without execution.'],
 8:['Fundamentals of digital video and its compression','Section 3.1 lectures 25 and 27; actual temporal correlation/motion estimation. Optical flow and tracking are application extensions.']
};
function unitsForRun(run,feature){
 const groups={},extensions=[],report=run?.report||{},trace=report.ivp_demo||run?.demonstration||{},steps=trace.steps||[];
 const add=(n,step)=>{(groups[n]??=[]).push({title:step.title||step.id,parameters:step.parameters||{}});};
 if(run?.status?.state!=='complete')return {units:[],groups,omitted:[],extensions:['No completed selected run: no numbered unit is claimed as executed.'],notice:'Select or process a result for execution-based course evidence.'};
 for(const step of steps){const id=step.id,title=(step.title||'').toLowerCase();
  if(id==='gray'||title.includes('grayscale'))add(1,step);
  if(['smooth','background'].includes(id)||/gamma|noise reduction|gaussian|median|percentile stretch|contrast stretch|clahe/.test(title))add(2,step);
  if(/bilateral|unsharp|sharpen/.test(title))extensions.push(step.title+': enhancement variant beyond the explicitly named syllabus operations.');
  if(/opening|closing|erosion|dilation/.test(title))add(4,step);
  if(/canny|sobel|laplacian|prewitt/.test(title))add(5,step);
  if(['selected','components','shapes'].includes(id)){add(6,step);if(id!=='selected')extensions.push('Connected-component/area-shape filtering extends the listed connectivity topics.');}
  if(['value','seed'].includes(id)||/hsv|white and yellow/.test(title))add(7,step);
 }
 const m=report.motion;
 if(m&&Number(m.frames||report.frames)>1)add(8,{title:'Temporal motion analysis over '+Number(m.frames||report.frames)+' decoded video frames',parameters:{method:m.method}});
 if(report.helmet)extensions.push('Application extension: AI-based helmet recognition; not explicitly covered by a numbered syllabus unit. Single-frame motorcycle/person/head association is application geometry.');
 if(report.zebra||m?.zebra)extensions.push('Detector recognition and reviewed polygon/rectangle occupancy are application logic; polygon inclusion is not Unit 6 segmentation.');
 if(m?.direction||m?.incidents?.enabled)extensions.push('Configured direction, finite-line/zone and traffic-rule review are application logic; no direct numbered-unit match.');
 if(m?.uses_ai)extensions.push('Vehicle detection and identity association use an external trained model, beyond the numbered classical IVP topics.');
 if(m?.speed?.calibrated)extensions.push('Homography/perspective calibration and displacement/time speed are application logic, not Unit 3 transforms.');
 if(m)extensions.push('Optical flow/tracking extend Unit 8 motion topics; the named block-matching algorithms are not claimed.');
 const prep=report.helmet?.input_preparation||report.zebra?.input_preparation;
 if(prep){const enabled=prep.enabled_colour_operations||[];extensions.push('Input preparation enabled operations: '+(enabled.join(', ')||'none')+'. Disabled/identity settings: '+(prep.disabled_or_identity_settings||[]).join(', ')+'. Resize applied: '+String(prep.resize_applied)+'. Feature extraction executed: '+String(prep.feature_extraction_executed)+'.');
  if(enabled.length===1&&!prep.resize_applied&&prep.output_equals_display_oriented_input===false&&['gamma','denoise','percentile_stretch','clahe','exposure'].includes(enabled[0])&&!(enabled[0]==='denoise'&&prep.settings?.denoise==='Bilateral'))add(2,{title:'Contributing input preparation: '+enabled[0],parameters:prep.settings});
  else if(enabled.length)extensions.push('Per-operation visible contribution is not isolated in this preparation record; extra units are withheld.');
 }
 if((report.helmet||report.zebra)&&!steps.length)extensions.push('Internal quality diagnostics do not feed the detector decision. A selected video frame alone is not temporal processing. Missing preparation traces are not backfilled with guessed operations.');
 if(!steps.length&&!m&&!report.helmet&&!report.zebra)extensions.push('This archive lacks a contributing-operation trace; no extra units inferred from available controls.');
 return {units:Object.keys(groups).map(Number).sort((a,b)=>a-b),groups,omitted:trace.omitted_steps||[],extensions:[...new Set(extensions)],notice:'Mapped from this completed result only. Actual contributing step titles and parameters are listed; disabled/no-change and inspection-only alternatives are not counted.'};
}
function courseRunEvidenceHTML(run,feature){
 const e=unitsForRun(run,feature);
 return '<h2>Course units used (actual executed operations)</h2>'+(run?'<p>Selected result: '+escapeHTML(run.id)+' · '+escapeHTML(run.source||run.report?.source||'recorded source')+'</p>':'')+'<p>'+escapeHTML(e.notice)+'</p>'+(e.units.length?e.units.map(n=>'<section><h3>Unit '+n+' — '+COURSE_UNITS[n][0]+'</h3><p>'+escapeHTML(COURSE_UNITS[n][1])+'</p><ul>'+e.groups[n].map(step=>'<li>'+escapeHTML(step.title)+' · '+escapeHTML(JSON.stringify(step.parameters))+'</li>').join('')+'</ul></section>').join(''):'<p>Application logic; no direct numbered-unit match recorded for this selected result.</p>')+(e.omitted.length?'<details><summary>Recorded no-change boundaries omitted ('+e.omitted.length+')</summary><pre>'+escapeHTML(JSON.stringify(e.omitted,null,2))+'</pre></details>':'')+'<h3>Extensions beyond the listed syllabus</h3><ul>'+e.extensions.map(v=>'<li>'+escapeHTML(v)+'</li>').join('')+'</ul><small>Course-policy source: Section 3.1 Instruction Plan and Section 5 laboratory details. The private policy document is not bundled.</small>';
}
// Curriculum connections describe this workflow; recorded execution remains separate.
// References were checked against the supplied policy, instruction plan pp. 5–8
// and laboratory table p. 11. No policy PDF is served or bundled with the app.
function courseFeatureContext(run,feature){
 if(run){const r=run.report||{},m=r.motion;return {goal:(r.ivp_demo||run.demonstration)?.goal||null,mode:m?(m.uses_ai===true?'vehicles':'ivp'):null,kind:r.kind==='video'?'video':'image',saturation:r.settings?.saturation};}
 const control=id=>typeof $==='function'?$(id)?.value:undefined;
 return {goal:control('ivp-goal')||'regions',mode:control('motion-mode')||'vehicles',kind:typeof state!=='undefined'?state.media?.kind:null,saturation:typeof state!=='undefined'?state.settings?.saturation:undefined};
}
function courseFeatureProfile(feature,context={}){
 const rows=[],add=(unit,topic,reference,use)=>rows.push({unit,topic,reference,use});
 const gray=()=>add(1,'Digital images as matrices; gray-level representation','Lecture 1 · p. 5; Lab 1 · week 1 · p. 11','Convert RGB into a grayscale intensity image for the following analysis stages.');
 const smooth=()=>add(2,'Smoothing spatial filters; Gaussian filtering; histogram equalization','Lectures 6–8 · pp. 5–6; Lab 4 · week 4 · p. 11','Smooth the grayscale image and apply local contrast enhancement before the selected output. CLAHE is a local, contrast-limited extension of histogram equalization.');
 const morph=use=>add(4,'Dilation, erosion, opening and closing','Lecture 11 · p. 6; Lab 6 · week 7 · p. 11',use);
 const temporal=use=>add(8,'Video representation, temporal correlation and fundamentals of motion estimation','Lectures 25 & 27 · p. 8; Lab 10 · week 12 · p. 11',use);
 let title='',extension='';
 if(feature==='lab'){
  title={regions:'Connected region map',edges:'Edge map',colour:'Colour image'}[context.goal]||'Image processing';
  if(context.goal==='regions'||context.goal==='edges'){gray();smooth();}
  if(context.goal==='regions'){
   morph('Opening removes small mask structures; closing fills gaps before region labelling.');
   add(6,'Threshold segmentation, pixel connectivity and Otsu’s method','Lectures 17–18 · p. 7; Lab 8 · week 10 · p. 11','The selected threshold makes a binary mask, then connected components group its pixels into regions. Adaptive thresholding is explicitly listed in Lab 8.');
   extension='Component area filtering is a project extension of connectivity. These regions do not identify vehicles.';
  }else if(context.goal==='edges'){
   add(5,'Gradient-based edge detection; Canny edge detector','Lectures 15–16 · p. 7; Lab 5 · week 6 · p. 11','Canny uses the prepared intensity image and two hysteresis thresholds to produce the displayed edge map.');
   extension='An edge map locates intensity boundaries; it does not assign traffic-object labels.';
  }else if(context.goal==='colour'){
   add(2,'Contrast stretching, power-law transformation, spatial filtering and histogram equalization','Lectures 3, 5–8 · pp. 5–6; Labs 3–4 · weeks 3–4 · p. 11','Selected colour-pipeline settings apply gamma, contrast stretch, denoising, sharpening or luminance CLAHE. Disabled and unchanged operations are excluded from the recorded evidence below.');
   if(Number.isFinite(context.saturation)&&context.saturation!==1)add(7,'RGB / HSV colour models and model transformation','Lectures 21–23 · p. 7; Lab 9 · week 11 · p. 11','The enabled saturation adjustment converts RGB to HSV, changes saturation, and returns to RGB.');
   extension='CLAHE, bilateral filtering and unsharp masking are implementation variants; the policy names their broader enhancement families.';
  }else extension='This saved image has no selected-output trace. Its recorded operations, when available, are listed separately without guessing a processing path.';
 }else if(feature==='markings'){
  title='Road markings';
  add(2,'Neighbourhood processing; median filtering','Lectures 6–7 · p. 5; Lab 4 · week 4 · p. 11','Estimate a local brightness background with a median filter, then compare candidate paint with nearby road pixels.');
  morph('Clean the candidate mask with opening and closing before measuring connected fragments.');
  add(6,'Threshold segmentation and pixel connectivity','Lecture 17 · p. 7; Lab 8 · week 10 · p. 11','Combine appearance and local-contrast thresholds, then label connected candidate fragments. Shape and darker-flank checks are additional project logic.');
  add(7,'HSV colour model, model transformation and colour slicing','Lectures 22–23 · p. 7; Lab 9 · week 11 · p. 11','Use HSV brightness, saturation and hue to build white/yellow appearance masks in the selected road patch.');
  extension='Coverage, shape filtering and darker-flank checks measure appearance candidates. They do not establish crossing/stop-line geometry or traffic rules.';
 }else if(feature==='tracking'){
  title=context.mode==='ivp'?'Tracking & speed · classical IVP':'Tracking & speed · vehicle mode';
  if(context.mode==='ivp'){
   gray();add(2,'Gaussian smoothing','Lectures 6–7 · p. 5; Lab 4 · week 4 · p. 11','Smooth frames before the foreground model discovers moving regions.');
   morph('Remove small foreground structures and close mask gaps before discovering tracks.');
   add(6,'Foreground/background segmentation and pixel connectivity','Lectures 17–18 · p. 7','A MOG2 background model produces foreground labels; connected components supply candidate regions. MOG2 is the project’s implementation, not a claim to implement the policy’s EM binarization exercise.');
  }
  temporal(context.mode==='ivp'?'Follow texture points between frames with Lucas–Kanade optical flow and use timestamps to measure displacement per second.':'Use IVP camera registration and timestamped vehicle-track positions to measure motion over successive frames. Vehicle proposals come from the separate local detector.');
  extension='Optical flow and tracking extend the motion-estimation topic. Speed = displacement / elapsed time; perspective calibration and the optional speed-limit display are project logic. The named block-matching search algorithms are not implemented here.';
 }else if(['direction','red_line','restricted_red','speed_review','required_stop'].includes(feature)){
  const pages={direction:['Travel direction','Compare a vehicle’s observed displacement over time with the reviewed travel direction.','The direction corridor, vector comparison and wrong-direction candidate rules are project geometry and decision logic.'],red_line:['Line crossing during red','Follow positions across successive frames to find passage over the reviewed line.','Finite-line intersection and the user-declared red timeline are project logic; the app does not infer a signal state here.'],restricted_red:['Area entry during red','Follow vehicle positions over time to identify entry into the reviewed area.','Polygon inclusion, entry rules and the declared red timeline are project logic. Polygon occupancy is not Unit 6 image segmentation.'],speed_review:['Calibrated speed review','Measure timestamped displacement over an observation window before comparing supported speeds.','Homography calibration, displacement/time speed, uncertainty gates and speed-threshold review are project logic. A perspective transform is not the DFT, DCT or Hadamard transform taught in Unit 3.'],required_stop:['Required-stop review','Use successive observed positions and timestamps to examine approach, dwell and boundary crossing.','Approach geometry, calibration, dwell-time thresholds and configured stop context are project logic. They do not establish a legal verdict.']};
  [title,,extension]=pages[feature];temporal(pages[feature][1]);
  extension+=' Vehicle recognition uses a trained model outside the numbered classical IVP units; optical flow extends Unit 8 rather than implementing its block-matching searches.';
 }else if(feature==='zebra'){
  title='Crossing occupancy';
  if(context.kind==='video')temporal('Video mode follows observed vehicles over time and aggregates occupancy in the reviewed crossing polygon.');
  extension='Vehicle detection and polygon/box overlap are model-based recognition and project geometry, not image segmentation. A single image or selected frame does not claim temporal motion processing. Any contributing input enhancement is recorded separately.';
 }else if(feature==='helmet'){
  title='Helmet observations';extension='Helmet recognition and rider/head association use trained models and project geometry. They are not listed in the numbered IVP units. Only contributing input enhancement, when recorded, can establish a classical-IVP unit for a result; a selected video frame is still a single-image check.';
 }else if(feature==='detection'){
  title='Detection & rules';extension='Object recognition and traffic-rule assessments use trained models and application logic outside the numbered IVP units. Available filters or diagnostic masks alone do not prove they contributed to a detection; use the result’s recorded preparation evidence.';
 }else if(feature==='overview'){
  title='Scene overview';extension='The local vision-language model describes the original frame. This is an AI extension with no direct numbered-unit implementation claimed; it does not run the image-laboratory stages or measure motion.';
 }else if(feature==='library'){
  return {title:'Saved runs',rows,extension:'Open a result to see the unit and topic mapping for that result’s own feature. Browsing saved files does not execute an IVP algorithm.'};
 }else return {title:'Selected result',rows,extension:'This result does not identify one feature clearly. Only its recorded operations are shown; no unrelated page mapping is attached.'};
 return {title,rows,extension};
}
function courseGuideHTML(feature,context){
 const profile=courseFeatureProfile(feature,context);
 const cards=profile.rows.map(row=>'<article class="course-topic"><div class="course-topic-heading"><span class="course-unit-badge">Unit '+row.unit+'</span><h3>'+escapeHTML(COURSE_UNITS[row.unit][0])+'</h3></div><p class="course-topic-name">'+escapeHTML(row.topic)+'</p><p>'+escapeHTML(row.use)+'</p><small>'+escapeHTML(row.reference)+'</small></article>').join('');
 return '<div class="course-guide" data-course-feature="'+escapeHTML(feature||'unknown')+'"><div class="course-guide-heading"><div><span class="eyebrow">COURSE CONNECTION</span><h2>Units & topics · '+escapeHTML(profile.title)+'</h2></div><span class="course-guide-tag">Feature guide</span></div><p class="course-guide-intro">'+(profile.rows.length?'These syllabus topics support this feature’s implementation. The selected mode and output determine this list; completed-run evidence below confirms what actually contributed.':'This page’s core function has no direct numbered-unit algorithm to claim.')+'</p>'+(cards?'<div class="course-topic-grid">'+cards+'</div>':'')+'<p class="course-extension"><strong>Project-specific part.</strong> '+escapeHTML(profile.extension)+'</p><small class="course-source">Course policy: §3.1 Instruction Plan, pp. 5–8; §5 Laboratory details, p. 11. References use the PDF’s page numbers.</small></div>';
}
function featureUnitsHTML(run,feature){
 const guide=courseGuideHTML(feature,courseFeatureContext(run,feature));
 return guide+(run?'<details class="course-evidence"><summary>Recorded operations for this result</summary><div class="course-run-evidence">'+courseRunEvidenceHTML(run,feature)+'</div></details>':feature==='library'?'':'<p class="course-evidence-empty">Process this feature to attach its source-bound execution evidence. This guide is not a claim that a run has already executed.</p>');
}
function featureSignature(feature){
 const controls={tracking:['motion-mode','motion-view','motion-axis','motion-position','motion-use-roi','speed-enabled','speed-width','speed-length'],red_line:['incident-enabled','feature-context-confirmed','incident-condition','incident-intervals','incident-direction','incident-start','incident-end','motion-axis','motion-position'],restricted_red:['incident-enabled','feature-context-confirmed','incident-condition','incident-intervals','incident-zone-rect','incident-restricted'],speed_review:['motion-mode','motion-view','incident-enabled','feature-context-confirmed','incident-limit','speed-enabled','speed-width','speed-length','incident-calibration-confirmed','incident-real-time'],required_stop:['incident-enabled','feature-context-confirmed','incident-condition','incident-intervals','incident-approach','incident-direction','incident-start','incident-end','motion-axis','motion-position','speed-enabled','speed-width','speed-length','incident-calibration-confirmed','incident-real-time']};
 return JSON.stringify({speedDisplay:['tracking','speed_review'].includes(feature)&&$('speed-limit-controls')?[$('speed-limit-enabled').checked,$('speed-limit-value').value,speedLimitBinding]:null,redGeometry:typeof redPage==='function'&&redPage(feature)?redGeometry:null,controls:(controls[feature]||[]).map(id=>{const e=$(id);return [id,e?.value,e?.checked];}),settings:feature==='lab'?state.settings:null,goal:feature==='lab'?$('ivp-goal')?.value:null,roi:['markings','tracking'].includes(feature)?state.region:null,target:feature==='tracking'&&typeof targetRectangle!=='undefined'?targetRectangle:null,synthetic:IVP_INCIDENT_FEATURES.includes(feature)?$('incident-synthetic')?.checked:null,zebra:feature==='zebra'&&typeof zebraReviewed!=='undefined'?zebraReviewed:null,direction:feature==='direction'&&typeof directionReviewed!=='undefined'?directionReviewed:null,calibration:['tracking','speed_review','required_stop'].includes(feature)&&typeof calibrationPoints!=='undefined'?calibrationPoints:null});
}
function runFeature(run){return run.report?.helmet?'helmet':run.report?.zebra?'zebra':run.report?.motion?motionRunFeature(run):run.report?.kind==='ivp_marking_survey'?'markings':'lab';}
function mayPresentRun(run){
 const submitted=unitSubmissions.get(run.id);if(!submitted)return false;
 return submitted.feature===workbenchTask&&submitted.signature===featureSignature(submitted.feature)&&submitted.source===state.media?.id&&submitted.revision===state.media?.source_revision&&submitted.requestedStamp===$('timestamp')?.value;
}
function rememberFeatureRun(run){
 const submitted=unitSubmissions.get(run?.id),owner=submitted?.feature||run?.report?.info?.ui_feature;
 if(run?.status?.state!=='complete'){if(owner)unitRuns.delete(owner);updateFeatureUnits();return;}
 if(!mayPresentRun(run))return;
 const feature=runFeature(run),r=run.report,b=r.helmet?.source_binding||r.zebra?.source_binding||r.source_binding||r.ivp_demo?.source_binding;
 const revision=b?.source_revision||r.source_info?.source_revision;
 const media=b?.media_id||null,stamp=b?.timestamp_seconds;
 if(revision!==state.media?.source_revision||media&&media!==state.media?.id||stamp!==undefined&&Math.abs(stamp-(state.preview?.timestamp??-1))>1e-6)return;
 unitRuns.set(feature,{run,source:state.media.id,revision,stamp:state.preview?.timestamp,requestedStamp:$('timestamp')?.value,signature:featureSignature(feature)});updateFeatureUnits();
}
function resetFeatureUnits(){unitRuns.clear();if($('motion-result'))$('motion-result').hidden=true;if($('frame-export-result'))$('frame-export-result').hidden=true;updateFeatureUnits();}
function beginFeatureAttempt(){
 unitRuns.delete(workbenchTask);
 for(const id of ['ivp-demo-result','motion-result','frame-export-result']){
  const card=$(id);if(card&&!card.hidden)card.setAttribute('data-previous-result','true');
 }
 updateFeatureUnits();
}
function updateFeatureUnits(){
 if(!$('course-units'))return;const entry=unitRuns.get(workbenchTask);
 const valid=entry&&entry.source===state.media?.id&&entry.revision===state.media?.source_revision&&entry.stamp===state.preview?.timestamp&&entry.requestedStamp===$('timestamp')?.value&&entry.signature===featureSignature(workbenchTask);
 $('course-units').innerHTML=featureUnitsHTML(valid?entry.run:null,workbenchTask);
}

function setupFeatureUnits(){document.addEventListener('input',updateFeatureUnits);document.addEventListener('change',updateFeatureUnits);setupGuidedUX();if(typeof setupRedReview==='function')setupRedReview();setupSpeedLimit();}

function captureFeatureSubmission(){return {feature:workbenchTask,source:state.media?.id,revision:state.media?.source_revision,requestedStamp:$('timestamp')?.value,signature:featureSignature(workbenchTask)};}
function bindFeatureSubmission(run,context){unitSubmissions.set(run.id,context);while(unitSubmissions.size>100)unitSubmissions.delete(unitSubmissions.keys().next().value);}

function paintFeatureInput(){if(state.preview&&['helmet','zebra','direction',...IVP_INCIDENT_FEATURES].includes(workbenchTask)){ $('base-image').src=state.preview.original;$('base-image').alt='Original source for '+workbenchTasks[workbenchTask].title;}}

// Readiness guidance describes existing controls; it never supplies review consent.
const GUIDED_TASKS={
 lab:['Understand an image','See grayscale, edges or connected regions, one operation at a time.','crossing.jpg','ivp-generate'],
 markings:['Measure road markings','Select a road patch and measure paint-like appearance.','crossing.jpg','survey-run'],
 tracking:['Explore movement','Track moving regions with classical IVP. Start with pixels/second; calibration is optional.','india-city.mp4','motion-run'],
 zebra:['Check crossing occupancy','Review a crossing band, then check vehicle overlap with the local detector.','crossing.jpg','zebra-frame'],
 helmet:['Inspect helmets','Use the installed head detector. Weak or missing evidence remains unknown.','indian_riders.png','helmet-frame'],
 direction:['Review travel direction','Draw a road region and expected direction, then review tracked motion.','india-city.mp4','direction-run'],
 red_line:['Line crossing during red','Draw a line → declare the signal context → review crossings. No road dimensions needed.','synthetic-speed-v1/videos/green_cross_18.mp4','motion-run'],
 restricted_red:['Area entry during red','Draw an area → declare when it is restricted → review entries. No road dimensions needed.','synthetic-speed-v1/videos/green_cross_18.mp4','motion-run'],
 speed_review:['Review calibrated speed','Supply measured road dimensions, or explicitly apply a synthetic example’s virtual scale. The optional display limit is separate from incident review.','synthetic-speed-v1/videos/green_cross_18.mp4','motion-run'],
 required_stop:['Review a required stop','Configure the observed approach, stop boundary and calibrated motion.','synthetic-speed-v1/videos/green_cross_18.mp4','motion-run']
};
function guidedPlan(feature,f){
 const next=(stage,title,body,target,activate=false)=>({stage,title,body,target,activate});
 if(f.running)return next(4,'Processing is in progress',f.foreground?'The selected image is being processed. Your result and executed stages will appear below.':'Follow the progress below. You can cancel the active job; wait for it to finish before starting another.',f.foreground?'ivp-demo-status':'active-job');
 if(!f.media)return next(1,'Choose footage to begin','Upload a file or try the recommended example below. Nothing is analysed until you choose Run.','source-picker');
 if(f.videoRequired&&f.kind!=='video')return next(1,'This task needs a video','The selected file is a still image. Choose a video to measure motion over time.','source-picker');
 if(f.loading)return next(2,'Preparing your preview','Keep your settings. Setup becomes available when the preview is ready.','preview-state');
 if(!f.preview)return next(1,'Preview unavailable','The selected file has no usable preview yet. Use Retry this preview below to keep your setup, or choose another file.','source-picker');
 if(f.dependency)return next(2,'Required local detector unavailable',f.dependency+' You can still use the image laboratory without a model.','ivp-workspace',true);
 if(f.firstFrame)return next(2,'Return to the first frame','This geometry must be reviewed at the start of this video. Returning to frame zero does not confirm the review.','direction'===feature?'direction-first':'incident-frame-zero',true);
 if(feature==='zebra'&&!f.reviewed)return next(2,f.drawComplete?'Confirm the crossing band':'Mark the full crossing band',f.drawComplete?'Inspect the four corners, including gaps between stripes, then confirm your review.':f.drawing?'Click the next corner on the image. Include the full band, not each stripe.':'Click four corners around the full crossing. No speed calibration is needed.',f.drawComplete?'zebra-review':f.drawing?'frame':'zebra-mark',!f.drawing&&!f.drawComplete);
 if(feature==='direction'&&!f.reviewed)return next(2,f.drawComplete?'Confirm the region and arrow':'Mark the road and expected direction',f.drawComplete?'Inspect the region and arrow, then confirm that they express your intended setup.':f.drawing?'Finish four region corners, then click the arrow tail and tip on the image.':'Mark four road-region corners, followed by the expected-travel arrow. No speed calibration is needed.',f.drawComplete?'direction-review':f.drawing?'frame':'direction-mark',!f.drawing&&!f.drawComplete);
 if(feature==='markings'&&!f.region)return next(2,'Select the road patch','Drag a rectangle over visible road paint. Avoid vehicles and bright reflections.','survey-select',true);
 if(f.setupError)return next(2,'Complete the required setup',f.setupError,f.setupTarget||'motion-options');
 if(f.completed)return next(4,'Your result is ready','Inspect the saved result and its executed steps. Downloads belong to that run. Change a setting to prepare another run.',f.resultTarget||'view-job-result',!f.resultTarget);
 return next(3,'Ready to run',(f.unknown?'Context is set to unknown. Conclusions depending on that context will be withheld. ': '')+(f.synthetic?'This example uses declared virtual geometry. Its speeds and rule conditions are synthetic test evidence.':'Use the action below. Results and downloads stay with this task; uncertain observations remain labelled.'),GUIDED_TASKS[feature]?.[3]||'ivp-generate');
}
function guidedFacts(){
 const feature=workbenchTask,incident=IVP_INCIDENT_FEATURES.includes(feature),vehicle=incident||['helmet','zebra','direction'].includes(feature)||(feature==='tracking'&&$('motion-mode').value==='vehicles');
 const foreground=typeof demoGenerating!=='undefined'&&demoGenerating;
 const entry=unitRuns.get(feature),completed=entry&&mayPresentRun(entry.run)&&entry.stamp===state.preview?.timestamp;
 const f={media:!!state.media,kind:state.media?.kind,preview:!!state.preview,loading:workspaceRestoring||state.rendering||state.pending,videoRequired:isVideoFeature(feature),synthetic:state.media?.example?.synthetic===true,region:!!state.region,running:foreground||!!state.job&&!!$('cancel-job')&&!$('cancel-job').hidden,foreground,completed,runId:completed?entry.run.id:null,resultTarget:feature==='lab'?'ivp-demo-result':feature==='markings'?'survey-result':null};
 if(feature==='helmet'&&state.config?.helmet?.ready!==true)f.dependency=state.config?.helmet?.message||'The helmet model is not ready on this computer.';
 else if(vehicle&&state.config?.vehicle_tracking_ready!==true)f.dependency='The local vehicle detector is not ready on this computer.';
 f.firstFrame=(incident||feature==='direction')&&state.preview?.timestamp!==0;
 if(feature==='zebra'){f.reviewed=zebraMatches()&&!!zebraReviewed;f.drawComplete=zebraMatches()&&zebraPoints.length===4;f.drawing=zebraDrawing;}
 if(feature==='direction'){f.reviewed=directionMatches()&&!!directionReviewed;f.drawComplete=directionMatches()&&directionPoints.length===4&&directionArrow.length===2;f.drawing=directionDrawing;}
 if(incident&&state.media?.kind==='video'&&state.preview){
  if(['speed_review','required_stop'].includes(feature)){
   try{if(!speedCalibration())throw Error('Enable calibration, mark four road corners, and enter their measured width and length. For a synthetic example, explicitly apply its virtual setup.');}
   catch(e){f.setupError=e.message;f.setupTarget=state.media?.example?.virtual_setup&&!$('speed-enabled').checked?'synthetic-apply':!$('speed-enabled').checked?'speed-enabled':typeof calibrationPoints!=='undefined'&&calibrationPoints.length!==4?'speed-mark':!Number($('speed-width').value)?'speed-width':!Number($('speed-length').value)?'speed-length':'speed-mark';}
  }
  if(!f.setupError){try{featureIncidentConfig(feature);}catch(e){f.setupError=e.message;f.setupTarget=e.field||'incident-enabled';}}
  if(f.setupTarget==='synthetic-apply')f.setupError='This is a synthetic example. Open its virtual setup below, inspect the declared dimensions, and apply them explicitly. You must then review the corners and confirm timing yourself.';
  if(!f.setupError&&['speed_review','required_stop'].includes(feature)&&(!$('incident-calibration-confirmed').checked||!$('incident-real-time').checked)){f.setupError='Review and confirm the road dimensions and original real-time playback before interpreting metric speed.';f.setupTarget=!$('incident-calibration-confirmed').checked?'incident-calibration-confirmed':'incident-real-time';}
 }
 if(!f.setupError&&['tracking','speed_review'].includes(feature)&&$('speed-limit-enabled')?.checked){try{speedLimitRequest(speedCalibration());}catch(e){f.setupError=e.message;f.setupTarget=e.field||'speed-limit-controls';}}
 f.unknown=incident&&feature!=='speed_review'&&$('incident-condition').value==='unknown';
 return f;
}
function revealGuidedControl(id,activate=false){
 const element=$(id);if(!element)return;
 for(let p=element;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;
 element.scrollIntoView({behavior:'smooth',block:'center'});
 if(activate&&!element.disabled)element.click();else if(typeof element.focus==='function')element.focus({preventScroll:true});
}
function guidedErrorAdvice(message){
 const text=String(message);
 if(/may already have started|Check Saved runs before repeating/i.test(text))return {category:'Processing outcome unknown',fix:'The request may have been accepted. Check Saved runs and the current job status before starting another run. Your setup is retained; this action will not resubmit the job.',target:'history-button',action:'Check Saved runs',activate:true};
 if(/saved run cannot be opened/i.test(text))return {category:'Saved run unavailable',fix:'Status polling has stopped. Open Saved runs and choose an available record. No processing was restarted.',target:'history-button',action:'Open Saved runs',activate:true};
 if(/server|fetch|network|connection|upload interrupted/i.test(text))return {category:'Connection problem',fix:'Your current inputs are retained. Check that Traffic Lab is running on this computer, then retry the action. If the server was restarted, reload this page.',target:'source-picker'};
 if(/model|weights|detector|dependency|ultralytics/i.test(text))return {category:'Local dependency unavailable',fix:'Check the installation guide for the named local dependency. You can continue with classical image processing while it is unavailable.',target:'ivp-workspace'};
 if(/source.*chang|revision|reopen|no longer|stale/i.test(text))return {category:'Source changed',fix:'Reopen the intended file, inspect its new preview, and review geometry again before running.',target:'source-picker'};
 if(/codec|decode|corrupt|unreadable|cannot read|format|file|upload/i.test(text))return {category:'Could not read this input',fix:'Check the named file and try a supported JPG/PNG image or MP4 video. Keep the original; use another sample to confirm the app is working.',target:'source-picker'};
 if(/corner|calibrat|metre|width|length|km\/h/i.test(text))return {category:'Calibration needs attention',fix:'Review all four corners and enter the real width and length in metres. Use virtual dimensions only for an explicitly synthetic example.',target:'speed-enabled'};
 if(/review|confirm|region|rectangle|condition|threshold|frame zero|line|polygon|arrow/i.test(text))return {category:'Setup needs attention',fix:'Complete the highlighted task setup, then retry. Review confirmations must be made by you for the current source.',target:'guided-next'};
 return {category:'The action could not finish',fix:'Read the reported detail below. Your inputs are retained. Retry the same action after correcting the reported problem; choose another sample if it concerns this file.',target:'guided-next'};
}
function showGuidedError(message){
 if(typeof ivpPage!=='function'||!ivpPage()||!$('guided-error'))return;
 const advice=guidedErrorAdvice(message),root=$('guided-error');
 root.hidden=false;root.innerHTML='<strong>'+escapeHTML(advice.category)+'</strong><p>'+escapeHTML(message)+'</p><p>'+escapeHTML(advice.fix)+'</p><small>Task: '+escapeHTML(workbenchTasks[workbenchTask]?.title||workbenchTask)+' · Source: '+escapeHTML(state.media?.name||'none selected')+'</small><div><button class="quiet" id="guided-error-fix">'+escapeHTML(advice.action||'Go to the relevant step')+'</button>'+(advice.category==='Connection problem'?'<button class="quiet" id="guided-retry-connection">Reload connection</button>':'')+(state.media&&state.config&&/preview/i.test(message)&&advice.category!=='Processing outcome unknown'?'<button class="quiet" id="guided-retry-preview">Retry this preview</button>':'')+'<button class="quiet" id="guided-error-dismiss">Dismiss message</button></div>';
 $('guided-error-fix').onclick=()=>revealGuidedControl(advice.target,advice.activate||advice.target==='ivp-workspace');
 if($('guided-retry-connection'))$('guided-retry-connection').onclick=()=>window.location.reload();
 if($('guided-retry-preview'))$('guided-retry-preview').onclick=()=>{root.hidden=true;renderPreview();};
 $('guided-error-dismiss').onclick=()=>{root.hidden=true;};
}
function updateGuidedUX(){
 const root=$('guided-start');if(!root)return;
 root.hidden=!ivpPage()||workbenchTask==='library';if(root.hidden)return;
 const task=GUIDED_TASKS[workbenchTask];if(!task)return;
 $('guided-goal').value=workbenchTask;
 const f=guidedFacts(),plan=guidedPlan(workbenchTask,f),body=$('guided-readiness');
 const html='<span class="guided-step">Step '+plan.stage+' of 4 · '+(plan.stage===1?'Footage':plan.stage===2?'Setup':plan.stage===3?'Run':'Progress & results')+'</span><h3>'+escapeHTML(plan.title)+'</h3><p>'+escapeHTML(plan.body)+'</p><small>'+escapeHTML(task[1])+'</small>';
 if(body.innerHTML!==html)body.innerHTML=html;
 $('guided-source').textContent=state.media?'Selected: '+state.media.name:'No footage selected';
 $('guided-next').textContent=f.completed&&!f.running?'View results':plan.stage===4?'View progress':plan.target==='feature-context-confirmed'?'Inspect draft geometry':plan.target==='ivp-workspace'?'Use classical image lab':plan.stage===3?'Go to Run':plan.stage===1?'Choose footage':'Continue setup';
 $('guided-next').onclick=()=>{if(f.completed&&!f.running){openRun(f.runId);return;}revealGuidedControl(plan.target==='feature-context-confirmed'?'frame':plan.target,plan.activate);};
 $('guided-example').textContent=f.videoRequired||IVP_INCIDENT_FEATURES.includes(workbenchTask)?'Try a video example':'Try an image example';
 $('guided-example').disabled=workspaceRestoring||state.rendering||state.pending;
 $('guided-recheck').hidden=!f.dependency;
 const action=$(task[3]);
 if(action){action.setAttribute('aria-describedby','guided-readiness');if(IVP_INCIDENT_FEATURES.includes(workbenchTask)){const disabled=!(plan.stage===3||f.completed&&!f.running)||startingMotion;if(action.disabled!==disabled)action.disabled=disabled;}}
 const beside=$('guided-preview-next');if(beside){beside.hidden=!['zebra','direction','red_line','restricted_red','speed_review','required_stop'].includes(workbenchTask);const text=plan.stage===3?'Setup ready · go to Run':plan.title;if(beside.textContent!==text)beside.textContent=text;beside.onclick=()=>{if(f.completed&&!f.running){openRun(f.runId);return;}revealGuidedControl(plan.target,plan.activate);};}
}
function setupGuidedUX(){
 if(!$('guided-start'))return;
 $('guided-goal').innerHTML=Object.entries(GUIDED_TASKS).filter(([id])=>id!=='helmet').map(([id,t])=>'<option value="'+id+'">'+escapeHTML(t[0])+'</option>').join('');
 $('guided-goal').oninput=$('guided-goal').onchange=()=>{const goal=$('guided-goal').value;if(goal!==workbenchTask)navigateTool(goal);};
 $('guided-example').onclick=async()=>{const name=GUIDED_TASKS[workbenchTask][2];if(name.startsWith('synthetic-speed-v1/'))$('synthetic-choice').value=name;await sample(name);};
 $('guided-source-change').onclick=()=>revealGuidedControl('source-picker');
 $('guided-recheck').onclick=recheckGuidedDependencies;
 document.addEventListener('input',event=>{if(event.target?.id!=='guided-goal')updateGuidedUX();});document.addEventListener('change',updateGuidedUX);
 if(typeof MutationObserver!=='undefined'){
  const observer=new MutationObserver(records=>{if(records.every(r=>$('guided-start').contains(r.target)||$('guided-error').contains(r.target)))return;updateGuidedUX();});
  observer.observe(document.body,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['disabled','data-workbench']});
 }
 updateGuidedUX();
}
async function recheckGuidedDependencies(){
 if(!state.config||!state.settings||workspaceRestoring){window.location.reload();return;}
 $('guided-recheck').disabled=true;
 try{const config=await api('/api/config');state.config={...state.config,helmet:config.helmet,vehicle_tracking_ready:config.vehicle_tracking_ready,speed_display_v1:config.speed_display_v1===true};actions();updateGuidedUX();notify('Local dependency status refreshed. Your footage and setup are retained.');}
 catch(e){notify(e.message,true);}finally{$('guided-recheck').disabled=false;}
}

'use strict';
// UI-only setup for the two image-coordinate red-context reviews.
let redGeometry=null,redDrawing=false,redPoints=[],redLayout=null;
const redHomes=new Map();
function redPage(feature=workbenchTask){return typeof document!=='undefined'&&document.body?.dataset?.redReview==='true'&&ivpPage()&&['red_line','restricted_red'].includes(feature);}
function redBound(){return !!redGeometry&&redGeometry.feature===workbenchTask&&redGeometry.media===state.media?.id&&redGeometry.revision===state.media?.source_revision;}
function redError(message,field='red-draw'){const e=Error(message);e.field=field;return e;}
function redLineValid(points){return Array.isArray(points)&&points.length===2&&points.every(p=>Array.isArray(p)&&p.length===2&&p.every(v=>Number.isFinite(v)&&v>=0&&v<=1))&&Math.hypot(points[1][0]-points[0][0],points[1][1]-points[0][1])>=.02;}
function redRequestLine(feature,draft=false){
 if(!redPage(feature))return undefined;
 if(!redBound()){if(draft)return null;throw redError('Draw and inspect '+(feature==='red_line'?'the line':'the area')+' on this video’s first frame.');}
 if(feature==='red_line'&&redGeometry.manual)for(const id of ['incident-start','incident-end'])if(!$(id).value.trim())throw redError('Enter a percentage for both endpoints, or draw a new line.',id);
 if(feature==='red_line'&&!redLineValid(redGeometry.line))throw redError('Place two distinct endpoints at least 2% of the frame apart.');
 return feature==='red_line'?redGeometry.line:null;
}
function invalidateRedReview(clearRestriction=false){
 if(!redPage())return;
 $('feature-context-confirmed').checked=false;$('incident-enabled').checked=false;
 if(clearRestriction)$('incident-restricted').checked=false;
}
function clearRedReviewGeometry(){redGeometry=null;redDrawing=false;redPoints=[];if($('frame'))$('frame').classList.remove('red-drawing');}
function bindRedGeometry(line=null){
 redGeometry={feature:workbenchTask,media:state.media.id,revision:state.media.source_revision,line};
 invalidateRedReview(true);
 if(state.media.example?.synthetic===true)$('incident-synthetic').checked=true;
}
function redReady(){return redPage()&&state.media?.kind==='video'&&state.media.source_revision&&state.preview?.timestamp===0&&!state.pending&&!state.rendering&&!workspaceRestoring;}
function beginRedDrawing(){
 if(!redReady())throw redError('Show the first frame and wait for its preview before drawing.','incident-frame-zero');
 clearRedReviewGeometry();invalidateRedReview(true);redDrawing=true;
 $('frame').classList.add('red-drawing');$('frame').scrollIntoView({behavior:'smooth',block:'center'});refreshRedReview();
}
function addRedPoint(point){
 if(!redDrawing||!redReady()||!Array.isArray(point)||point.length!==2||point.some(v=>!Number.isFinite(v)||v<0||v>1))return;
 redPoints.push(point);
 if(redPoints.length===2){
  const [a,b]=redPoints;
  if(workbenchTask==='red_line'){
   if(!redLineValid(redPoints)){redPoints.pop();throw redError('The endpoints are too close. Click a second endpoint farther away.');}
   bindRedGeometry(redPoints.map(p=>p.slice()));
  }else{
   const rect=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];
   if(rect[2]-rect[0]<.02||rect[3]-rect[1]<.02){redPoints.pop();throw redError('The area must be at least 2% wide and high. Click a farther opposite corner.');}
   $('incident-zone-rect').value=rect.map(v=>(v*100).toFixed(3)).join(', ');bindRedGeometry();
  }
  redDrawing=false;$('frame').classList.remove('red-drawing');
 }
 refreshRedReview();
}
function knownVirtualSetup(media=state.media){
 const s=media?.example?.virtual_setup;
 return media?.kind==='video'&&!!media.source_revision&&media.example?.synthetic===true&&s?.kind==='fixed_translation_fixture_v1'&&s.reference_frame===0&&s.threshold_is_legal_limit===false&&s.width_m===16&&s.length_m===40&&JSON.stringify(s.points)==='[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]'?s:null;
}
function placeRedExample(){
 if(!redReady()||!knownVirtualSetup())throw redError('Open a catalog synthetic example and show its first frame. This placement does not apply to real footage.','red-example-open');
 clearRedReviewGeometry();
 if(workbenchTask==='red_line'){$('motion-axis').value='horizontal';$('motion-position').value='65';$('incident-start').value='15';$('incident-end').value='85';$('incident-direction').value='positive';bindRedGeometry([[.15,.65],[.85,.65]]);}
 else{$('incident-zone-rect').value='15, 65, 85, 78';bindRedGeometry();}
 $('incident-condition').value='unknown';$('incident-intervals').value='';refreshRedReview();
}
function redSetupChanged(event){
 if(!redPage())return;
 const id=event.target?.id,geometry=['motion-axis','motion-position','incident-start','incident-end','incident-zone-rect'];
 if(id==='feature-context-confirmed'){$('incident-enabled').checked=$('feature-context-confirmed').checked;refreshRedReview();return;}
 if(![...geometry,'incident-direction','incident-condition','incident-intervals','incident-restricted','incident-synthetic'].includes(id))return;
 invalidateRedReview(geometry.includes(id)||['incident-condition','incident-intervals'].includes(id));
 if(geometry.includes(id)){
  redDrawing=false;redPoints=[];$('frame').classList.remove('red-drawing');
  if(redReady()){
   const start=Number($('incident-start').value)/100,end=Number($('incident-end').value)/100,pos=Number($('motion-position').value)/100;
   const line=$('motion-axis').value==='horizontal'?[[start,pos],[end,pos]]:[[pos,start],[pos,end]];
   bindRedGeometry(workbenchTask==='red_line'?line:null);redGeometry.manual=true;
  }else clearRedReviewGeometry();
 }
 refreshRedReview();
}
function redAnnotation(c){
 let svg='';
 if(c.line){
  const [a,b]=c.line.map(p=>p.map(v=>v*100)),dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy),sign=c.direction==='negative'?-1:1;
  const mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2,nx=-dy/len*sign,ny=dx/len*sign;
  svg=`<defs><marker id="red-arrow-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#ffd166"/></marker></defs><line x1="${mx-nx*4}" y1="${my-ny*4}" x2="${mx+nx*6}" y2="${my+ny*6}" stroke="#ffd166" stroke-width=".6" marker-end="url(#red-arrow-head)" ${c.direction==='both'?'marker-start="url(#red-arrow-head)"':''}/><text x="${Math.max(2,Math.min(72,a[0]))}" y="${Math.max(4,a[1]-2)}" class="red-overlay-label">CROSS THIS LINE</text>`;
 }else if(c.zone)svg=`<text x="${c.zone[0]*100+1}" y="${Math.max(4,c.zone[1]*100-2)}" class="red-overlay-label">ENTER THIS AREA</text>`;
 if(redDrawing&&redPoints.length===1)svg+=`<circle cx="${redPoints[0][0]*100}" cy="${redPoints[0][1]*100}" r="1" fill="#ffd166"/>`;
 return svg;
}
function refreshRedReview(){
 updateVirtualDimensions();
 if(!redPage()||!$('red-review-steps'))return;
 const line=workbenchTask==='red_line',ready=redReady(),bound=redBound();
 $('red-draw').disabled=!ready;$('red-example-place').disabled=!ready||!knownVirtualSetup();
 $('red-example-open').disabled=!!(state.pending||state.rendering||workspaceRestoring);
 $('red-draw-status').textContent=redDrawing?(line?'Click endpoint ':'Click corner ')+(redPoints.length+1)+' of 2 on the preview.':bound?'Geometry placed. Inspect it, choose the applicable context, then confirm below.':'No geometry placed for this source. Draw it or use the labelled synthetic example.';
 const timeline=$('incident-intervals').value.trim(),condition=$('incident-condition').value;
 $('red-context-status').textContent=(state.media?.example?.synthetic?'SYNTHETIC TEST · ':'')+(timeline?'Using your supplied time intervals. Uncovered times are unknown.':'Your declared context: '+condition.toUpperCase()+'. Applies to the whole clip.')+' Signal state is not detected. '+(condition==='unknown'&&!timeline?'Red-dependent conclusions will be withheld.':'');
 $('feature-context-confirmed').disabled=!ready||!bound||redDrawing;
 paintIncidents();if(typeof updateFeatureUnits==='function')updateFeatureUnits();
}
function updateVirtualDimensions(){
 const el=$('virtual-dimensions-note');if(!el)return;el.hidden=!knownVirtualSetup();
 if(!el.hidden)el.textContent='This selected catalog sample declares a 16 m width across × 40 m length along the virtual road patch. Corners: top-left → top-right → bottom-right → bottom-left at (10%,10%), (90%,10%), (90%,90%), (10%,90%). These dimensions apply only to that patch, not the full video or differently drawn corners. Inspect before applying; no calibration or confirmation is supplied by this note.';
}
function moveRedControl(id,destination){
 const node=$(id).closest('label')||$(id);if(!redHomes.has(node)){const home=document.createComment('red-control-home');node.before(home);redHomes.set(node,home);}$(destination).appendChild(node);node.hidden=false;
}
function restoreRedControls(){
 for(const [node,home] of redHomes){home.after(node);home.remove();}redHomes.clear();
 $('incident-direction').options[0].textContent='Positive · down / left';$('incident-direction').options[1].textContent='Negative · up / right';$('incident-direction').options[2].textContent='Both directions';
 $('incident-condition').querySelector('[value="red"]').textContent='Red · this boundary applies to these vehicles';
}
function configureRedReview(){
 const active=ivpPage()&&['red_line','restricted_red'].includes(workbenchTask);if(!$('red-review-steps'))return;
 if(redLayout!==workbenchTask){clearRedReviewGeometry();redLayout=workbenchTask;}
 document.body.dataset.redReview=String(active);
 if(!active){for(const [node,home] of redHomes)home.after(node);redHomes.clear();$('feature-context-confirmed').disabled=false;return;}
 const line=workbenchTask==='red_line';
 for(const [id,to] of [['incident-direction','red-direction'],['incident-condition','red-condition'],['incident-intervals','red-timeline'],['incident-restricted','red-restriction'],['feature-context-confirmed','red-confirm'],['incident-synthetic','red-confirm'],['motion-axis','red-manual-line'],['motion-position','red-manual-line'],['incident-start','red-manual-line'],['incident-end','red-manual-line'],['incident-zone-rect','red-manual-zone']])moveRedControl(id,to);
 $('red-direction').hidden=!line;$('red-manual-line').hidden=!line;$('red-manual-zone').hidden=line;$('red-restriction').hidden=line;
 $('red-geometry-title').textContent=line?'1 · Place the crossing line':'1 · Mark the restricted area';
 $('red-geometry-help').textContent=line?'A vehicle must cross this finite line in the arrow’s direction. Click its two endpoints; the line can be slanted.':'A vehicle must enter this rectangle. Click two opposite corners around the area that is restricted during red.';
 $('red-draw').textContent=line?'Draw line on preview':'Draw area on preview';
 $('red-example-place').textContent=line?'Place example line':'Place example area';
 $('red-numeric-help').textContent=line?'Changing these values replaces the drawn line with a horizontal or vertical line. Percentages refer to this frame.':'Enter left, top, right, bottom as percentages of this frame. Changing them requires a new review.';
 $('incident-condition').querySelector('[value="red"]').textContent=line?'Red · this boundary applies to these vehicles':'Red · this area is restricted for these vehicles';
 $('motion-line-help').hidden=true;
 $('incident-direction').options[0].textContent='Along the shown arrow';$('incident-direction').options[1].textContent='Reverse the arrow';$('incident-direction').options[2].textContent='Either direction';
 refreshRedReview();
}
function setupRedReview(){
 $('red-draw').onclick=()=>{try{beginRedDrawing();}catch(e){notify(e.message,true);}};
 $('red-example-open').onclick=()=>sample('synthetic-speed-v1/videos/green_cross_18.mp4');
 $('red-example-place').onclick=()=>{try{placeRedExample();}catch(e){notify(e.message,true);}};
 $('red-first').onclick=()=>$('incident-frame-zero').click();
 document.addEventListener('input',redSetupChanged,true);document.addEventListener('change',redSetupChanged,true);
 $('frame').addEventListener('pointerdown',e=>{if(!redDrawing||!redPage())return;e.preventDefault();e.stopImmediatePropagation();const r=$('frame').getBoundingClientRect();try{addRedPoint([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);}catch(error){notify(error.message,true);refreshRedReview();}},true);
 $('frame').addEventListener('keydown',e=>{if(e.key==='Escape'&&redDrawing){redDrawing=false;redPoints=[];$('frame').classList.remove('red-drawing');refreshRedReview();}});
}

const SPEED_LIMIT_RESTART='This running service does not support the speed-limit display yet. Restart the existing local app normally after active work finishes, then reload this page. No limit can be applied by this service.';
let speedLimitBinding=null,speedLimitNotice='';
function speedLimitAvailable(){return typeof ivpPage==='function'&&ivpPage()&&!!$('speed-limit-controls')&&state.config?.speed_display_v1===true;}
function speedLimitContext(calibration){return JSON.stringify({media:state.media?.id,revision:state.media?.source_revision,width:state.media?.width,height:state.media?.height,analysis_max_width:960,calibration,mode:$('motion-mode').value,roi:workbenchTask==='tracking'&&$('motion-use-roi').checked?state.region:null});}
function clearSpeedLimit(message=''){
 if(!$('speed-limit-controls'))return;
 $('speed-limit-enabled').checked=false;$('speed-limit-value').value='';speedLimitBinding=null;speedLimitNotice=message;
}
function speedLimitRequest(calibration){
 const unit=calibration?'km/h':'px/s',on=$('speed-limit-enabled')?.checked===true;
 if(!on)return {enabled:false,limit:null,unit};
 const fail=(text,field='speed-limit-value')=>{const e=Error(text);e.field=field;throw e;};
 if(!speedLimitAvailable())fail(SPEED_LIMIT_RESTART,'speed-limit-controls');
 if(speedLimitBinding!==speedLimitContext(calibration))fail('Source, units or analysis setup changed. Re-enable the filter and enter the limit again.','speed-limit-enabled');
 const text=$('speed-limit-value').value.trim(),limit=Number(text);
 if(!text||!Number.isFinite(limit)||limit<=0)fail('Enter a positive finite speed limit in '+unit+'.');
 return {enabled:true,limit,unit};
}
function syncSpeedLimit(){
 if(!$('speed-limit-controls'))return;
 const visible=ivpPage()&&['tracking','speed_review'].includes(workbenchTask),available=speedLimitAvailable();
 $('speed-limit-controls').hidden=!visible;if(!visible)return;
 $('speed-limit-pixel-link').hidden=workbenchTask!=='speed_review';
 let calibration=null,problem='';try{calibration=speedCalibration();}catch(e){problem=e.message;}
 const context=problem?null:speedLimitContext(calibration);
 if(speedLimitBinding!==null&&speedLimitBinding!==context)clearSpeedLimit('Source, units or analysis setup changed. Enter a new limit; no conversion was made.');
 $('speed-limit-unit').textContent=calibration?'· '+(state.media?.example?.synthetic?'virtual km/h':'km/h'):'· analysis px/s';
 $('speed-limit-enabled').disabled=!available||!!problem||!state.media||state.media.kind!=='video';
 $('speed-limit-value').disabled=$('speed-limit-enabled').disabled||!$('speed-limit-enabled').checked;
 $('speed-limit-status').textContent=!available?SPEED_LIMIT_RESTART:problem||speedLimitNotice||($('speed-limit-enabled').checked?'Only accepted speeds strictly above this value get a track box and trail.':'Off: keep the usual tracking display.');
 $('speed-limit-space').textContent='px/s means camera-registered analysis pixels after an aspect-preserving resize to at most 960 px wide. Original and preview resolutions can differ. The exact analysis width and height are recorded with the saved result. No automatic conversion to km/h. '+($('motion-mode').value==='vehicles'?'This mode uses local vehicle candidates.':'This mode filters moving texture regions, not cars.');
}
function setupSpeedLimit(){
 if(!$('speed-limit-controls'))return;
 $('speed-limit-pixel-link').onclick=()=>navigateTool('tracking');
 $('speed-limit-enabled').addEventListener('change',()=>{
  speedLimitNotice='';if($('speed-limit-enabled').checked){try{speedLimitBinding=speedLimitContext(speedCalibration());}catch(e){clearSpeedLimit(e.message);}}
  else speedLimitBinding=null;
  syncSpeedLimit();updateFeatureUnits();updateGuidedUX();
 });
 document.addEventListener('input',event=>{if(event.target?.id==='speed-limit-value')speedLimitNotice='';if(['speed-limit-value','speed-width','speed-length','speed-enabled','motion-mode','motion-use-roi'].includes(event.target?.id))syncSpeedLimit();});
 document.addEventListener('change',event=>{if(['speed-enabled','motion-mode','motion-use-roi'].includes(event.target?.id))syncSpeedLimit();});
}
function speedDisplaySummary(display){
 if(!display)return '';
 const size=display.analysis_frame_size,dimensions=Array.isArray(size)&&size.length===2?size.join(' × ')+' px':'not recorded';
 const counts=display.sample_counts||{};
 return `<section class="speed-display-report"><h3>Speed-limit display</h3><p>${display.enabled?'Only accepted speeds strictly above '+escapeHTML(String(display.limit))+' '+escapeHTML(display.unit)+' were highlighted. Equal, below, pending and withheld samples have no track box or trail.':'Off: the usual tracking display was retained. No threshold comparison was performed.'}</p><p>${escapeHTML(display.scope)} · Analysis image: ${escapeHTML(dimensions)}. ${escapeHTML(display.coordinate_space)}</p><p>${display.enabled?'Comparison uses the same two-decimal speed saved in the trajectories CSV. '+Number(counts['strictly above limit']||0)+' above-limit track samples. ':''}All measurements remain exported; this is not a violation decision.</p></section>`;
}
