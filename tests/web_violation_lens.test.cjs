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
  assert.equal(lens.progress(undefined),null);assert.equal(lens.progress(NaN),null);assert.equal(lens.progress(-4),0);assert.equal(lens.progress(200),100);
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
  assert.match(unavailable,/No LLM explanation available. Deterministic facts remain/);
  assert.match(unavailable,/data-action="explain" disabled/);
  const available=lens.renderCard(card,0,{available:true});
  assert.match(available,/Send these facts using my ChatGPT plan/);
  assert.match(available,/Footage is not sent/);
  assert.doesNotMatch(available,/data-action="explain" disabled/);
  assert.match(lens.renderCard({id:'empty',facts:[]},0,{available:true}),/data-action="explain" disabled/);
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
    if(path==='/api/run?id=saved')return {id:'saved',source:'<unsafe>.mp4',status:{state:'complete',progress:100}};
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
});

test('saving accept, reject and reset refreshes counts without rebuilding cards or losing evidence and notes', async () => {
  const fixture=decision=>({run_id:'saved',fingerprint:'fp',cards:[{id:'one',rule:'helmet',status:'needs_review',review:{decision,note:''}},{id:'two',rule:'wrong_way',review:{decision:null,note:''}}]});
  const harness=browserHarness((path,options)=>{
    if(path==='/api/config')return {vehicle_tracking_ready:true,helmet:{ready:true}};
    if(path==='/api/run-library')return {runs:[]};
    if(path==='/api/violation-llm')return {available:false};
    if(path==='/api/run?id=saved')return {id:'saved',source:'clip.mp4',status:{state:'complete',progress:100}};
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
