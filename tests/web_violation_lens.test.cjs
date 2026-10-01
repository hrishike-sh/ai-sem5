const test = require('node:test');
const assert = require('node:assert/strict');
const lens = require('../web/violation_lens.js');
const fs = require('node:fs');
const vm = require('node:vm');

function browserHarness(respond, savedID = null) {
  const elements = new Map(), requests = [], navigation = [];
  function element(id) {
    if (!elements.has(id)) elements.set(id, {id,hidden:false,disabled:false,checked:false,files:[],value:id==='rule-filter'?'all':'',textContent:'',innerHTML:'',listeners:{},classList:{toggle(){}},addEventListener(type, fn){this.listeners[type]=fn;},setAttribute(name,value){this[name]=value;},removeAttribute(name){delete this[name];},replaceChildren(){this.innerHTML='';},focus(){}});
    return elements.get(id);
  }
  const window = {document:{getElementById:element,querySelector(){return {content:'test-csrf-token'};},querySelectorAll(){return [];}},localStorage:{getItem(){return savedID;},setItem(){}},location:{assign(url){navigation.push(url);}},fetch:async(path,options={})=>{requests.push({path,options});return {ok:true,json:async()=>respond(path,options)};},setTimeout(){return 1;}};
  const context=vm.createContext({window,URL,setTimeout,clearTimeout,module:{exports:{}}});
  vm.runInContext(fs.readFileSync(require.resolve('../web/violation_lens.js'),'utf8'),context);
  return {element,requests,navigation,settle:()=>new Promise(resolve=>setImmediate(resolve))};
}

test('uploaded names, model facts, IDs and notes cannot create executable markup', () => {
  const attack = '\"><img src=x onerror=alert(1)><script>bad()</script>&\'';
  const html = lens.renderCard({id:attack,rule:attack,status:attack,reason:attack,track_id:attack,review:{note:'</textarea>'+attack},facts:[{text:attack,value:attack}],missing_facts:[{text:attack}]}, 0, {message:attack});
  assert.doesNotMatch(html, /<script>|<img src=x|<\/textarea>\"><img/);
  assert.match(html, /&lt;script&gt;bad\(\)&lt;\/script&gt;/);
  assert.match(html, /&lt;\/textarea&gt;/);
  assert.match(lens.escapeHTML(attack), /&quot;.*&amp;&#39;/);
});

test('only local server evidence and export URLs are accepted', () => {
  for (const url of ['javascript:alert(1)','https://example.com/a','//example.com/a','/\\example.com/a','/\n/evil','data:text/html,x',null]) assert.equal(lens.localURL(url),'');
  assert.equal(lens.localURL('/view/run/frame.png?x=1&y=2'),'/view/run/frame.png?x=1&y=2');
  assert.equal(lens.localURL('http://127.0.0.1:49123/token/frame.png','http://127.0.0.1:49123'),'http://127.0.0.1:49123/token/frame.png');
  for(const url of ['http://127.0.0.1:49124/token/frame.png','http://localhost:49123/token/frame.png','http://evil.test/token/frame.png','http://user@127.0.0.1:49123/token/frame.png'])assert.equal(lens.localURL(url,'http://127.0.0.1:49123'),'');
  assert.equal(lens.localURL('http://127.0.0.1:49123/token/frame.png'),'');
  assert.equal(lens.localURL('https://evil.test/a','https://evil.test'),'');
});

test('subscription sign-in accepts only the verified HTTPS issuer', () => {
  assert.equal(lens.authURL('https://auth.openai.com/api/accounts/authorize?state=one'),'https://auth.openai.com/api/accounts/authorize?state=one');
  for(const url of ['http://auth.openai.com/x','https://auth.openai.com.evil.test/x','https://evil.test/auth.openai.com','https://name@auth.openai.com/x','https://auth.openai.com:8888/x','javascript:bad()','/relative'])assert.equal(lens.authURL(url),'');
});

test('frame cards preserve actual timestamps and do not fabricate missing evidence', () => {
  const html=lens.renderFrames([{role:'during',id:'f1',timestamp:2.75,url:'/frame.png',available:true},{position:'after',timestamp:null,available:false,reason:'<not decoded>'}]);
  assert.match(html,/During<\/strong> · 2.75 s/);
  assert.match(html,/Before<\/strong> · Time unavailable/);
  assert.match(html,/&lt;not decoded&gt;/);
  assert.equal((html.match(/<img /g)||[]).length,1);
  assert.doesNotMatch(lens.renderFrames([{role:'during',available:true,url:'javascript:bad()',timestamp:3}]),/<img /);
  for (const value of [null,undefined,-1,Infinity,'3']) assert.equal(lens.timestamp(value),'Time unavailable');
  const served=lens.renderFrames([{role:'during',id:'local',timestamp:4,url:'http://127.0.0.1:49123/token/frame.png',available:true}],'http://127.0.0.1:49123');
  assert.match(served,/<img src="http:\/\/127.0.0.1:49123\/token\/frame.png"/);
});

test('review decisions require current evidence identity and stay separate from machine findings', () => {
  const card={id:'c1',rule:'helmet',status:'cannot_assess',recorded_verdict:'cannot_assess',review:{decision:'accept'},facts:[],missing_facts:[]};
  const view={run_id:'r1',fingerprint:'fp1',cards:[card]};
  assert.deepEqual(lens.reviewPayload(view,'c1','reject','unclear'),{id:'r1',card_id:'c1',fingerprint:'fp1',decision:'reject',note:'unclear'});
  assert.deepEqual(lens.reviewPayload(view,'c1','reset','old note'),{id:'r1',card_id:'c1',fingerprint:'fp1',decision:'reset',note:''});
  assert.throws(()=>lens.reviewPayload({...view,fingerprint:null},'c1','accept',''));
  assert.throws(()=>lens.reviewPayload(view,'other-card','accept',''));
  assert.throws(()=>lens.reviewPayload(view,'c1','guilty',''));
  const html=lens.renderCard(card,0,{});
  assert.match(html,/Evidence status: Cannot assess/);
  assert.match(html,/Recorded pipeline label: cannot_assess/);
  assert.match(html,/Human disposition: accepted/);
  assert.equal(card.status,'cannot_assess');
});

test('unknown job states stay recoverable and progress never invents completion', () => {
  for (const state of ['queued','running','encoding','unknown','cancelling']) assert.equal(lens.runActive({status:{state}}),true);
  for (const state of ['complete','cancelled','failed']) assert.equal(lens.runActive({status:{state}}),false);
  assert.equal(lens.runActive({}),true);
  for(const invalid of [undefined,null,NaN,Infinity,-4,200,'0.5',true])assert.equal(lens.progress(invalid),null);
  assert.equal(lens.progress(0),0);assert.equal(lens.progress(0.375),37.5);assert.equal(lens.progress(0.97),97);assert.equal(lens.progress(1),100);
});

test('worker progress renders queued zero, running fractions, completed one and unknown values correctly', async () => {
  for(const [state,value,label,barValue] of [['queued',0,'0%',0],['running',0.375,'38%',37.5],['encoding',0.97,'97%',97],['complete',1,'100%',100],['unknown',undefined,'Progress unavailable',undefined],['running',4,'Progress unavailable',undefined]]) {
    const harness=browserHarness(path=>{
      if(path==='/api/config')return {vehicle_tracking_ready:true,helmet:{ready:true}};
      if(path==='/api/run-library')return {runs:[]};
      if(path==='/api/violation-llm')return {available:false};
      if(path==='/api/run?id=progress-run')return {id:'progress-run',source:'clip.mp4',status:{state,progress:value}};
      if(path==='/api/violation-review?id=progress-run')return {run_id:'progress-run',fingerprint:'fp',cards:[]};
      throw Error('Unexpected request: '+path);
    },'progress-run');
    await harness.settle();
    assert.equal(harness.element('progress-label').textContent,label,state);
    assert.equal(harness.element('run-progress').value,barValue,state);
    assert.equal(harness.element('run-progress-wrap').hidden,false,state);
  }
});

test('helmet and direction candidates are prioritized without hiding uncertainty or other rules', () => {
  const cards=[{id:'a',rule:'speed',review:{decision:'reject'}},{id:'b',rule:'helmet',status:'cannot_assess'},{id:'c',rule:'wrong_way',review:{decision:'accept'}},{id:'d',rule:'red_light'}];
  assert.deepEqual(lens.visibleCards(cards,'all').map(c=>c.id),['b','c','a','d']);
  assert.deepEqual(lens.visibleCards(cards,'priority').map(c=>c.id),['b','c']);
  assert.deepEqual(lens.visibleCards(cards,'unreviewed').map(c=>c.id),['b','d']);
  assert.deepEqual(cards.map(c=>c.id),['a','b','c','d']);
});

test('explanations are unavailable by default and explicit consent is visible when available', () => {
  const card={id:'c',facts:[{id:'fact',text:'A recorded fact'}]};
  const unavailable=lens.renderCard(card,0,{available:false});
  assert.match(unavailable,/No new LLM explanation available. Deterministic facts remain/);
  assert.match(unavailable,/data-action="prepare" disabled/);
  const available=lens.renderCard(card,0,{available:true});
  assert.match(available,/Preview frames and facts for ChatGPT/);
  assert.match(available,/Send structured facts only to ChatGPT/);
  assert.doesNotMatch(available,/data-action="prepare" disabled/);
  assert.match(lens.renderCard({id:'empty',facts:[]},0,{available:true}),/data-action="prepare" disabled/);
});

test('full readiness blocks detection even if the vehicle detector alone is ready', () => {
  assert.equal(lens.readiness({vehicle_tracking_ready:true,helmet:{ready:true}}).ready,true);
  const status=lens.readiness({vehicle_tracking_ready:true,violation_readiness:{ready:false,blockers:[{id:'scene',message:'Install verified scene model.'}],warnings:[]}});
  assert.equal(status.ready,false);assert.deepEqual(status.blockers,['Install verified scene model.']);
});

test('ready scope warnings do not show missing-model installation or restart instructions', async () => {
  const harness=browserHarness(path=>path==='/api/config'?{violation_readiness:{ready:true,blockers:[],warnings:[{id:'scope',message:'Real-world accuracy has not been established.'}]}}:path==='/api/run-library'?{runs:[]}:{available:false});
  await harness.settle();
  assert.equal(harness.element('setup').hidden,false);
  assert.equal(harness.element('setup-title').textContent,'Detection available with limitations');
  assert.match(harness.element('setup-message').textContent,/accuracy has not been established/);
  assert.match(harness.element('setup-help').textContent,/Detection is available/);
  assert.doesNotMatch(harness.element('setup-help').textContent,/install|restart|weights/i);
});

test('actual missing-model blockers retain setup requirements', async () => {
  const harness=browserHarness(path=>path==='/api/config'?{violation_readiness:{ready:false,blockers:[{id:'model',message:'Install the verified vehicle model.'}],warnings:[]}}:path==='/api/run-library'?{runs:[]}:{available:false});
  await harness.settle();
  assert.equal(harness.element('setup').hidden,false);
  assert.equal(harness.element('setup-title').textContent,'Local model setup needed');
  assert.match(harness.element('setup-message').textContent,/Install the verified vehicle model/);
  assert.match(harness.element('setup-help').textContent,/resolve the required setup items/);
  assert.equal(harness.element('start').disabled,true);
});

test('prepared consent is bound to the exact card, evidence fingerprint and immutable preparation', () => {
  const view={run_id:'run',fingerprint:'fp'};
  const prepared={run_id:'run',card_id:'card',fingerprint:'fp',preparation_id:'prep1',mode:'images_and_facts',frames:[{id:'f1',available:true,url:'/image.jpg'}]};
  assert.equal(lens.validatePreparation(prepared,view,'card'),prepared);
  assert.deepEqual(lens.preparedPayload(prepared,view,'card'),{id:'run',card_id:'card',fingerprint:'fp',preparation_id:'prep1',mode:'images_and_facts',consent:true});
  for(const bad of [{...prepared,fingerprint:'old'},{...prepared,card_id:'other'},{...prepared,preparation_id:null},{...prepared,frames:[]},{...prepared,frames:[{id:'f1',available:true,url:'https://remote.test/img'}]}])assert.throws(()=>lens.validatePreparation(bad,view,'card'));
  assert.throws(()=>lens.preparedPayload(prepared,{...view,fingerprint:'new'},'card'));
});

test('prepared preview shows exact frames and facts safely and makes facts-only mode explicit', () => {
  const payload={mode:'images_and_facts',frames:[{id:'f1',role:'during',timestamp:2.25,url:'/exact.jpg',available:true}],facts:[{text:'<fact>',value:'<value>'}],missing_facts:[{text:'<missing>'}],limitations:['<limitation>']};
  const html=lens.renderPreparation(payload);
  assert.match(html,/src="\/exact.jpg"/);assert.match(html,/2.25 s/);assert.match(html,/data-action="send-prepared" disabled/);
  for(const text of ['fact','value','missing','limitation'])assert.match(html,new RegExp(`&lt;${text}&gt;`));
  const factsOnly=lens.renderPreparation({...payload,mode:'facts_only'});
  assert.match(factsOnly,/No images will be sent/);assert.doesNotMatch(factsOnly,/<img /);assert.match(factsOnly,/Send prepared structured facts only/);
});

test('fallible visual observations are separate and must cite prepared frame IDs', () => {
  const response={explanation:'Recorded facts remain unchanged.',model:'<model>',vision_mode:'selected_frames',visual_observations:[{frame_id:'f1',text:'<model observation>'}],visual_notice:'<notice>'};
  const html=lens.renderExplanation(response,{frames:[{id:'f1',available:true}]});
  assert.match(html,/Model observations of selected images/);assert.match(html,/These observations can be wrong/);assert.match(html,/&lt;model observation&gt;/);assert.match(html,/Frame f1/);
  const unknown=lens.renderExplanation(response,{frames:[{id:'other',available:true}]});
  assert.match(unknown,/could not be matched/);assert.doesNotMatch(unknown,/&lt;model observation&gt;/);
  assert.match(lens.renderExplanation({vision_mode:'facts_only',explanation:'Facts'}),/No images were sent/);
});

test('synthetic cards are explicitly identified as test evidence', () => {
  assert.match(lens.renderCard({id:'s',synthetic:true},0,{}),/SYNTHETIC TEST EVIDENCE/);
  assert.doesNotMatch(lens.renderCard({id:'r'},0,{}),/SYNTHETIC TEST EVIDENCE/);
});

test('page load performs read-only checks; ChatGPT authorization requires its explicit approval click', async () => {
  const harness=browserHarness(path=>path==='/api/config'?{vehicle_tracking_ready:true,helmet:{ready:true},defaults:{}}:path==='/api/run-library'?{runs:[]}:path==='/api/violation-auth/start'?{url:'https://auth.openai.com/api/accounts/authorize?state=test'}:{available:false});
  await harness.settle();
  assert.ok(harness.requests.length>=3);
  assert.ok(harness.requests.every(r=>!r.options.method));
  harness.element('connect-chatgpt').listeners.click();
  assert.equal(harness.element('auth-confirmation').hidden,false);
  assert.ok(harness.requests.every(r=>!r.options.method));
  await harness.element('approve-auth').listeners.click();
  const post=harness.requests.find(r=>r.path==='/api/violation-auth/start');
  assert.equal(post.options.method,'POST');
  assert.equal(post.options.headers['X-Traffic-Token'],'test-csrf-token');
  assert.deepEqual(JSON.parse(post.options.body),{approve:true});
  assert.deepEqual(harness.navigation,['https://auth.openai.com/api/accounts/authorize?state=test']);
  assert.ok(!harness.requests.some(r=>r.path==='/api/violation-explain'));
});

test('reload restores a completed run and evidence without resubmitting detection or sending facts', async () => {
  const harness=browserHarness(path=>{
    if(path==='/api/config')return {vehicle_tracking_ready:false,helmet:{ready:false}};
    if(path==='/api/run-library')return {runs:[]};
    if(path==='/api/violation-llm')return {available:false};
    if(path==='/api/run?id=saved')return {id:'saved',source:'<unsafe>.mp4',status:{state:'complete',progress:1}};
    if(path==='/api/violation-review?id=saved')return {run_id:'saved',fingerprint:'fp',synthetic:true,provenance_notice:'Synthetic test fixture',cards:[{id:'one',rule:'helmet',status:'cannot_assess',reason:'<unsafe reason>'}]};
    throw Error('Unexpected endpoint: '+path);
  },'saved');
  await harness.settle();
  assert.ok(harness.requests.every(r=>!r.options.method));
  assert.equal(harness.element('source-name').textContent,'<unsafe>.mp4');
  assert.match(harness.element('cards').innerHTML,/&lt;unsafe reason&gt;/);
  assert.equal(harness.element('provenance-notice').textContent,'Synthetic test fixture');
  assert.equal(harness.element('setup').hidden,false);
  assert.equal(harness.element('export-review').disabled,false);
  assert.equal(harness.element('start').disabled,true);
  assert.match(harness.element('next-action').textContent,/Saved run restored for review/);
  assert.match(harness.element('next-action').textContent,/upload the clip again/);
});

test('saving accept, reject and reset refreshes counts without rebuilding cards or losing evidence and notes', async () => {
  const fixture=decision=>({run_id:'saved',fingerprint:'fp',cards:[{id:'one',rule:'helmet',status:'needs_review',review:{decision,note:''}},{id:'two',rule:'wrong_way',review:{decision:null,note:''}}]});
  const harness=browserHarness((path,options)=>{
    if(path==='/api/config')return {vehicle_tracking_ready:true,helmet:{ready:true}};
    if(path==='/api/run-library')return {runs:[]};
    if(path==='/api/violation-llm')return {available:false};
    if(path==='/api/run?id=saved')return {id:'saved',source:'clip.mp4',status:{state:'complete',progress:1}};
    if(path==='/api/violation-review?id=saved')return fixture(null);
    if(path==='/api/violation-decision'){const {decision}=JSON.parse(options.body);return fixture(decision==='reset'?null:decision);}
    throw Error('Unexpected endpoint: '+path);
  },'saved');
  await harness.settle();
  assert.match(harness.element('review-summary').textContent,/0 reviewed/);
  let cardsHTML=harness.element('cards').innerHTML, rebuilds=0;
  Object.defineProperty(harness.element('cards'),'innerHTML',{get(){return cardsHTML;},set(value){cardsHTML=value;rebuilds++;}});
  const slots={'[data-slot="note"]':{value:'Selected note'},'[data-slot="disposition"]':{textContent:''},'[data-slot="feedback"]':{textContent:''},'[data-slot="frames"]':{innerHTML:'Loaded frame evidence'}};
  const button={disabled:false,dataset:{action:'accept'},closest(){return article;}};
  const article={dataset:{cardId:'one'},isConnected:true,querySelector(selector){return slots[selector];},querySelectorAll(){return [button];}};
  for(const [decision,count] of [['accept',1],['reject',1],['reset',0]]){
    button.dataset.action=decision;
    await harness.element('cards').listeners.click({target:{closest(){return button;}}});
    assert.match(harness.element('review-summary').textContent,new RegExp(`${count} reviewed`));
    assert.match(harness.element('review-summary').textContent,/2 shown/);
    assert.equal(rebuilds,0);
    assert.equal(slots['[data-slot="frames"]'].innerHTML,'Loaded frame evidence');
    if(decision!=='reset')assert.equal(slots['[data-slot="note"]'].value,'Selected note');
  }
  assert.equal(slots['[data-slot="note"]'].value,'');
});

test('preparing evidence never sends it, and image consent stays disabled until every exact preview loads', async () => {
  const view={run_id:'saved',fingerprint:'fp',cards:[{id:'one',rule:'helmet',facts:[{id:'fact',text:'Recorded fact'}],review:{decision:null}}]};
  const prepared={run_id:'saved',card_id:'one',fingerprint:'fp',preparation_id:'immutable-prep',mode:'images_and_facts',frames:[{id:'f1',role:'during',timestamp:2,url:'/exact.jpg',available:true}],facts:[{id:'fact',text:'Recorded fact'}],missing_facts:[],limitations:[]};
  const harness=browserHarness((path)=>{
    if(path==='/api/config')return {vehicle_tracking_ready:true,helmet:{ready:true}};
    if(path==='/api/run-library')return {runs:[]};
    if(path==='/api/violation-llm')return {available:true};
    if(path==='/api/run?id=saved')return {id:'saved',source:'clip.mp4',status:{state:'complete',progress:1}};
    if(path==='/api/violation-review?id=saved')return view;
    if(path==='/api/violation-prepare')return prepared;
    if(path==='/api/violation-explain')return {card_id:'one',explanation:'Recorded facts',vision_mode:'selected_frames',visual_observations:[{frame_id:'f1',text:'Model observation: unclear head visibility.'}]};
    throw Error('Unexpected endpoint: '+path);
  },'saved');
  await harness.settle();
  let preparedVisible=false;
  const images=[{complete:false,naturalWidth:0,listeners:{},addEventListener(type,fn){this.listeners[type]=fn;}}];
  const slots={'[data-slot="feedback"]':{textContent:''},'[data-slot="preview-status"]':{textContent:''},'[data-slot="explanation"]':{innerHTML:''},'[data-slot="preparation"]':{replaceChildren(){preparedVisible=false;},set innerHTML(value){this.html=value;preparedVisible=true;}}};
  const buttons={prepare:{disabled:false,dataset:{action:'prepare'},closest(){return article;}},send:{disabled:true,dataset:{action:'send-prepared'},closest(){return article;}}};
  const article={dataset:{cardId:'one'},isConnected:true,querySelector(selector){return selector==='[data-action="send-prepared"]'?buttons.send:slots[selector];},querySelectorAll(selector){return selector==='img[data-prepared-frame]'?images:preparedVisible?Object.values(buttons):[buttons.prepare];}};
  const click=button=>harness.element('cards').listeners.click({target:{closest(){return button;}}});
  await click(buttons.prepare);
  assert.ok(preparedVisible);assert.equal(buttons.send.disabled,true);
  assert.equal(harness.requests.filter(r=>r.path==='/api/violation-explain').length,0);
  await click(buttons.send);
  assert.equal(harness.requests.filter(r=>r.path==='/api/violation-explain').length,0);
  images[0].complete=true;images[0].naturalWidth=0;images[0].listeners.error();
  assert.equal(buttons.send.disabled,true);
  assert.match(slots['[data-slot="preview-status"]'].textContent,/could not be displayed/);
  await click(buttons.send);
  assert.equal(harness.requests.filter(r=>r.path==='/api/violation-explain').length,0);
  images[0].complete=true;images[0].naturalWidth=320;images[0].listeners.load();
  assert.equal(buttons.send.disabled,false);
  await click(buttons.send);
  const request=harness.requests.find(r=>r.path==='/api/violation-explain');
  assert.deepEqual(JSON.parse(request.options.body),{id:'saved',card_id:'one',fingerprint:'fp',preparation_id:'immutable-prep',mode:'images_and_facts',consent:true});
  assert.equal(request.options.headers['X-Traffic-Token'],'test-csrf-token');
  assert.match(slots['[data-slot="explanation"]'].innerHTML,/Model observations of selected images/);
  assert.equal(view.cards[0].review.decision,null);
  await harness.element('refresh-review').listeners.click();
  await harness.settle();
  const before=harness.requests.filter(r=>r.path==='/api/violation-explain').length;
  await click(buttons.send);
  assert.equal(harness.requests.filter(r=>r.path==='/api/violation-explain').length,before);
  assert.match(slots['[data-slot="feedback"]'].textContent,/Load and inspect all prepared image previews/);
});

test('empty real review shows zero supported candidates and the exact source/run identity', async () => {
  const harness=browserHarness(path=>{
    if(path==='/api/config')return {vehicle_tracking_ready:true,helmet:{ready:true}};
    if(path==='/api/run-library')return {runs:[]};
    if(path==='/api/violation-llm')return {available:false};
    if(path==='/api/run?id=real-run')return {id:'real-run',source:'permitted clip.mp4',status:{state:'complete'}};
    if(path==='/api/violation-review?id=real-run')return {run_id:'real-run',fingerprint:'fp',cards:[],omitted_nonapplicable_count:4};
  },'real-run');
  await harness.settle();
  assert.match(harness.element('review-summary').textContent,/0 supported candidates/);
  assert.match(harness.element('review-summary').textContent,/does not establish that no violation occurred/);
  assert.match(harness.element('review-summary').textContent,/4 explicitly non-applicable observations omitted/);
  assert.equal(harness.element('review-source').textContent,'Source: permitted clip.mp4 · Run: real-run');
  assert.equal(harness.element('start').disabled,true);
  assert.match(harness.element('next-action').textContent,/upload the clip again/);
});

test('saved image explanations reopen from their recorded descriptors without a connection or a new send', async () => {
  const result={mode:'images_and_facts',vision_mode:'selected_frames',explanation:'Saved canonical facts.',visual_observations:[{frame_id:'saved-frame',text:'Model observation: head is obscured.'}],images_sent:[{id:'saved-frame',timestamp:2.5,role:'subject_crop'}],visual_notice:'Model interpretations are not recorded findings.'};
  const harness=browserHarness(path=>{
    if(path==='/api/config')return {vehicle_tracking_ready:false,helmet:{ready:false}};
    if(path==='/api/run-library')return {runs:[]};
    if(path==='/api/violation-llm')return {available:false};
    if(path==='/api/run?id=saved')return {id:'saved',source:'clip.mp4',status:{state:'complete'}};
    if(path==='/api/violation-review?id=saved')return {run_id:'saved',fingerprint:'fp',cards:[{id:'card',facts:[{text:'Recorded fact'}]}],explanations:{card:result}};
    throw Error('Unexpected request: '+path);
  },'saved');
  await harness.settle();
  const html=harness.element('cards').innerHTML;
  assert.match(html,/Saved explanation/);assert.match(html,/No new ChatGPT request was sent/);
  assert.match(html,/head is obscured/);assert.match(html,/Frame saved-frame/);assert.match(html,/2.50 s/);
  assert.match(html,/data-action="prepare" disabled/);
  assert.ok(harness.requests.every(request=>!request.options.method));
});

test('facts-only button sends explicit facts-only consent without preparing or transmitting images', async () => {
  const harness=browserHarness(path=>{
    if(path==='/api/config')return {vehicle_tracking_ready:true,helmet:{ready:true}};
    if(path==='/api/run-library')return {runs:[]};
    if(path==='/api/violation-llm')return {available:true};
    if(path==='/api/run?id=saved')return {id:'saved',source:'clip.mp4',status:{state:'complete'}};
    if(path==='/api/violation-review?id=saved')return {run_id:'saved',fingerprint:'fp',cards:[{id:'one',facts:[{text:'Recorded fact'}]}]};
    if(path==='/api/violation-explain')return {card_id:'one',state:'explained',explanation:'Recorded facts.',vision_mode:'facts_only'};
    throw Error('Unexpected request: '+path);
  },'saved');
  await harness.settle();
  const slots={'[data-slot="feedback"]':{textContent:''},'[data-slot="explanation"]':{innerHTML:''}};
  const button={disabled:false,dataset:{action:'explain-facts'},closest(){return article;}};
  const article={dataset:{cardId:'one'},isConnected:true,querySelector(selector){return slots[selector];},querySelectorAll(){return [button];}};
  await harness.element('cards').listeners.click({target:{closest(){return button;}}});
  const posts=harness.requests.filter(request=>request.options.method==='POST');
  assert.equal(posts.length,1);assert.equal(posts[0].path,'/api/violation-explain');
  assert.deepEqual(JSON.parse(posts[0].options.body),{id:'saved',card_id:'one',fingerprint:'fp',mode:'facts_only',consent:true});
  assert.match(slots['[data-slot="explanation"]'].innerHTML,/No images were sent/);
});
