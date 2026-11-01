const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path').join(__dirname, '../explorer/static/');
const source = fs.readFileSync(path + 'app.js', 'utf8');
const html = fs.readFileSync(path + 'index.html', 'utf8');
const researchSource = source.split('// ---------- local research ----------')[1].split('// ---------- claude:')[0];
const viewSource = source.slice(source.indexOf('function setView(v)'), source.indexOf('const layerSel ='));
assert.ok(researchSource && viewSource);

class Element {
  constructor(id) { this.id=id; this.value=''; this.checked=false; this.disabled=false; this.hidden=false; this.dataset={}; this.listeners={}; this.options=[]; this.textContent=''; this.innerHTML=''; this.title=''; this.classes=new Set(); this.classList={toggle:(key,on)=>{ if(on===undefined) on=!this.classes.has(key); if(on) this.classes.add(key); else this.classes.delete(key); return on; }}; }
  get valueAsNumber() { return this.value === '' ? NaN : Number(this.value); }
  addEventListener(name, fn) { this.listeners[name]=fn; }
  removeAttribute(name) { delete this[name]; }
}
function response(data, status=200) { return {ok:status>=200 && status<300,status,text:async()=>JSON.stringify(data)}; }
function goodOverview(extra={}) { return {model_key:'fixture-qwen',live_model_enabled:true,operations:{score:true,generate:true,jobs:false,presets:false},runtime:{backend:'fixture-only',inference_loaded:false},live_validation:{validated:false},limits:{max_new:256,timeout_seconds:300},...extra}; }
function goodScore(extra={}) { return {prompt_token_ids:[101,102],completion_token_ids:[17,18],per_token_logprobs:[-0.2,-0.3],total_logprob:-0.5,mean_logprob:-0.25,completed:true,finish_reason:'completed',eos_policy:'exclude',boundary_policy:'contextual-prefix-stable',counts:{prompt_tokens:2,output_tokens:0,requested_tokens:2,scored_tokens:2,forward_passes:2,evaluated_tokens:5,wall_seconds:0.02},identity:{checkpoint:'fixture-only'},cleanup:{interventions_cleared:true,cache_cleared:true},...extra}; }
function goodGeneration(extra={}) { return {prompt_token_ids:[101,102],output_token_ids:[17,18],text:'result',stop_token_id:0,finish_reason:'eos',completed:true,eos_policy:'stop-before-eog',counts:{prompt_tokens:2,output_tokens:2,requested_tokens:17,scored_tokens:0,forward_passes:3,evaluated_tokens:9,wall_seconds:0.03},identity:{checkpoint:'fixture-only'},cleanup:{interventions_cleared:true,cache_cleared:true},...extra}; }
function harness() {
  const elements = new Map();
  for (const id of html.matchAll(/\bid="([^"]+)"/g)) elements.set('#'+id[1],new Element(id[1]));
  for (const m of html.matchAll(/<input\b[^>]*\bid="([^"]+)"[^>]*>/g)) { const el=elements.get('#'+m[1]); el.value=m[0].match(/\bvalue="([^"]*)"/)?.[1]??''; el.max=m[0].match(/\bmax="([^"]*)"/)?.[1]??''; }
  for (const m of html.matchAll(/<select\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) { const el=elements.get('#'+m[1]); el.options=[...m[2].matchAll(/<option\b[^>]*\bvalue="([^"]*)"[^>]*>/g)].map(m=>({value:m[1],disabled:/\bdisabled\b/.test(m[0])})); el.value=el.options[0]?.value??''; }
  elements.set('#r-output pre',new Element('r-output-pre'));
  const $ = selector => { assert.ok(elements.has(selector),'missing real HTML selector '+selector); return elements.get(selector); };
  const requests=[], external={chat:'retain-me',provider:'codex',history:['external private prompt']}, externalBefore=JSON.stringify(external);
  const nav=['map','embed','research'].map(value=>{const el=new Element(value);el.dataset.view=value;return el;});
  const document={body:new Element('body'),querySelectorAll:selector=>{ assert.equal(selector,'#views button');return nav; }};
  const research={overview:null,probe:null,ctl:null,busy:false,refreshing:false,refreshVersion:0};
  const context=vm.createContext({$,document,console,JSON,Number,Object,Array,Error,AbortController,structuredClone,setTimeout,clearTimeout,
    index:{model_key:'fixture-qwen'},arch:{n_layers:32,inter:9216},modelName:'Fixture Qwen',research,
    state:{view:'map',focus:null,chat:{mode:'add',dose:{add:0.8,mul:2}}},agent:external,
    hideTip(){},updateInsets(){},loadEmbeddings(){throw new Error('unexpected embeddings fetch')},
    async fetch(url,opts={}) { requests.push({url,opts,body:opts.body===undefined?undefined:JSON.parse(opts.body)}); return await context.fetchImpl(url,opts); },
    fetchImpl:async()=>response(goodOverview())
  });
  vm.runInContext(source.match(/^const esc = .*$/m)[0]+'\n'+researchSource+'\n'+viewSource+'\nglobalThis.handlers={researchRequest,researchBody,refreshResearch,researchControls,submitResearch,renderResearchProbe,setView,researchOperationAvailable,checkResearchResult,researchResultHTML,updateResearchJob,researchPresetAction};',context,{filename:'actual-research-ui.js'});
  return {context,$,research,requests,external,externalBefore,h:context.handlers};
}

const checks=[];
function check(name,fn){checks.push([name,fn]);}
check('opening Research performs overview GET only; returning preserves external Agent',async()=>{
  const t=harness(); assert.equal(t.requests.length,0);
  t.h.setView('research'); await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(t.requests.map(r=>[r.url,r.opts.method??'GET']),[['/api/research/overview','GET']]);
  assert.equal(t.context.state.view,'research'); assert.equal(t.context.document.body.classes.has('research-open'),true);
  t.h.setView('map'); assert.equal(t.context.state.view,'map'); assert.equal(t.context.document.body.classes.has('research-open'),false);
  assert.equal(JSON.stringify(t.external),t.externalBefore);
});
check('score submit preserves exact prompt/completion spaces, newlines, EOS and single arm',async()=>{
  const t=harness(); t.research.overview=goodOverview(); t.$('#r-mode').value='score';
  t.$('#r-prompt').value='  prompt\n';t.$('#r-answers').value=' 12\nnext ';t.$('#r-eos').value='include';
  t.context.fetchImpl=async()=>response(goodScore({eos_policy:'include'}));
  await t.h.submitResearch({preventDefault(){}});
  assert.equal(t.requests.length,1);assert.equal(t.requests[0].url,'/api/research/score');
  assert.deepEqual(t.requests[0].body,{text:'  prompt\n',completion:' 12\nnext ',chat:true,prefix:'',neurons:[],eos_policy:'include',timeout_seconds:60});
  assert.equal(t.research.probe.state,'completed');
});
check('generation submit preserves exact prompt and bounds without uncontracted fields',async()=>{
  const t=harness();t.research.overview=goodOverview();t.$('#r-prompt').value=' x\n ';t.$('#r-max-tokens').value='17';t.$('#r-deadline').value='23';
  t.context.fetchImpl=async()=>response(goodGeneration());await t.h.submitResearch({preventDefault(){}});
  assert.deepEqual(t.requests[0].body,{messages:[{role:'user',content:' x\n '}],neurons:[],max_new:17,timeout_seconds:23});
  assert.equal(t.research.probe.state,'completed');assert.match(t.$('#r-output').innerHTML,/TARGET OUTPUT/);assert.match(t.$('#r-output').innerHTML,/result/);
});
check('invalid coordinates, nonfinite doses and integer bounds reject before POST',async()=>{
  const t=harness();t.research.overview=goodOverview();t.$('#r-prompt').value='prompt';t.$('#r-use-neuron').checked=true;
  for(const [id,bad,good,pattern] of [['#r-layer','32','0',/Layer/],['#r-neuron','-1','0',/Neuron/],['#r-dose','Infinity','0.8',/Dose/],['#r-max-tokens','1.5','64',/Token limit/],['#r-deadline','301','60',/Deadline/]]){t.$(id).value=bad;await assert.rejects(t.h.researchBody('generate'),pattern);t.$(id).value=good;}
  assert.equal(t.requests.length,0);
});
check('200 application error and 422 detail surface as failed probes',async()=>{
  for(const [payload,status,message] of [[{error:'memory admission refused'},200,'memory admission refused'],[{detail:'unsupported configuration'},422,'unsupported configuration']]){
    const t=harness();t.research.overview=goodOverview();t.$('#r-prompt').value='prompt';t.context.fetchImpl=async()=>response(payload,status);
    await t.h.submitResearch({preventDefault(){}});assert.equal(t.research.probe.state,'failed');assert.equal(t.$('#r-job-error').textContent,message);assert.equal(t.$('#r-job-error').hidden,false);
  }
});
check('legacy first_token_only score is rejected, not accepted as complete',async()=>{
  const t=harness();t.research.overview=goodOverview();t.$('#r-mode').value='score';t.$('#r-prompt').value='prompt';t.$('#r-answers').value=' 12';
  t.context.fetchImpl=async()=>response({targets:[{text:' 12',first_token_only:true,logprob:-0.1}]});await t.h.submitResearch({preventDefault(){}});
  assert.equal(t.research.probe.state,'failed');assert.match(t.$('#r-job-error').textContent,/Rejected first-token-only/);assert.notEqual(t.$('#r-job-state').textContent,'Completed');
});
check('browser cancellation remains requested/unconfirmed and resume disabled',async()=>{
  const t=harness();t.research.overview=goodOverview();t.$('#r-prompt').value='prompt';
  let called;const ready=new Promise(resolve=>called=resolve);
  t.context.fetchImpl=async(_url,opts)=>new Promise((_resolve,reject)=>{called();opts.signal.addEventListener('abort',()=>reject(new Error('aborted')),{once:true});});
  const pending=t.h.submitResearch({preventDefault(){}});await ready;t.$('#r-cancel').onclick();await pending;
  assert.equal(t.research.probe.state,'cancel_requested');assert.match(t.$('#r-job-state').textContent,/unconfirmed/);assert.match(t.$('#r-stop-reason').textContent,/unconfirmed/);assert.equal(t.$('#r-resume').disabled,true);assert.equal(t.research.busy,false);
});
check('missing/false/disabled/mismatched capabilities disable submit and execution',async()=>{
  for(const overview of [null,goodOverview({operations:{}}),goodOverview({operations:{generate:false,score:false}}),goodOverview({live_model_enabled:false}),goodOverview({model_key:'another-checkpoint'}),goodOverview({admission:{allowed:false,reason:'memory refused'}})]){
    const t=harness();t.research.overview=overview;t.h.researchControls();assert.equal(t.$('#r-submit').disabled,true);await assert.rejects(t.h.researchBody('generate'),/Refresh research tools/);
  }
});
check('job/researcher modes follow the integrated server capabilities',async()=>{
  const t=harness();t.research.overview=goodOverview({operations:{generate:true,score:true,jobs:true,presets:true,researcher:true}});t.h.researchControls();
  for(const value of ['experiment','researcher'])assert.equal(t.$('#r-mode').options.find(x=>x.value===value).disabled,false);
});
check('completed score with two-token answer but only shared first token scored is rejected',async()=>{
  const t=harness();t.research.overview=goodOverview();t.$('#r-mode').value='score';t.$('#r-prompt').value='prompt';t.$('#r-answers').value=' 12';
  const result=goodScore();result.per_token_logprobs=[-0.2];result.counts.scored_tokens=1;result.total_logprob=-0.2;result.mean_logprob=-0.2;
  t.context.fetchImpl=async()=>response(result);await t.h.submitResearch({preventDefault(){}});
  assert.equal(t.research.probe.state,'failed');assert.equal(t.research.probe.rejected,true);assert.equal(t.$('#r-output').innerHTML,'');assert.match(t.$('#r-job-error').textContent,/Every answer token/);
});
check('inconsistent sequence total, mean and per-token coverage are rejected',async()=>{
  for(const change of [{total_logprob:-9},{mean_logprob:-3},{per_token_logprobs:[-0.2]}]){
    const t=harness();t.research.overview=goodOverview();t.$('#r-mode').value='score';t.$('#r-prompt').value='prompt';t.$('#r-answers').value=' 12';
    t.context.fetchImpl=async()=>response(goodScore(change));await t.h.submitResearch({preventDefault(){}});
    assert.equal(t.research.probe.state,'failed');assert.equal(t.research.probe.rejected,true);assert.equal(t.$('#r-output').innerHTML,'');
  }
});
check('worker-acknowledged canceled partial score shows coverage but no complete-sequence headline',async()=>{
  const t=harness();t.research.overview=goodOverview();t.$('#r-mode').value='score';t.$('#r-prompt').value='prompt';t.$('#r-answers').value=' 12';
  const result=goodScore({completed:false,finish_reason:'canceled',per_token_logprobs:[-0.2],total_logprob:-0.2,mean_logprob:-0.2});result.counts.scored_tokens=1;
  t.context.fetchImpl=async()=>response(result);await t.h.submitResearch({preventDefault(){}});
  assert.equal(t.research.probe.state,'canceled');assert.match(t.$('#r-job-state').textContent,/worker returned/);
  assert.match(t.$('#r-output').innerHTML,/Partial answer score/);assert.match(t.$('#r-output').innerHTML,/1 \/ 2 tokens scored/);assert.match(t.$('#r-output').innerHTML,/Not scored/);assert.doesNotMatch(t.$('#r-output').innerHTML,/Sequence log probability|Mean per token/);
});
check('unknown resource fields render Unavailable, never invented zero totals',async()=>{
  const t=harness();t.research.overview=goodOverview();t.$('#r-prompt').value='prompt';
  const result=goodGeneration();result.counts.wall_seconds=null;delete result.counts.evaluated_tokens;delete result.counts.forward_passes;
  t.context.fetchImpl=async()=>response(result);await t.h.submitResearch({preventDefault(){}});
  assert.equal(t.research.probe.state,'completed');
  for(const label of ['Evaluated tokens','Forward passes','Runtime seconds'])assert.match(t.$('#r-resources').innerHTML,new RegExp('<span>'+label+'</span><strong>Unavailable</strong>'));
});
check('submitted provenance remains a snapshot after a later overview refresh',async()=>{
  const t=harness();t.research.overview=goodOverview({identity:{checkpoint:'initial-id'},calibration:{hash:'initial-cal'}});t.$('#r-prompt').value='prompt';
  t.context.fetchImpl=async()=>response(goodGeneration());await t.h.submitResearch({preventDefault(){}});
  t.research.overview.identity.checkpoint='mutated-old-object';t.research.overview.calibration.hash='mutated-old-cal';
  t.context.fetchImpl=async()=>response(goodOverview({identity:{checkpoint:'refreshed-id'},calibration:{hash:'refreshed-cal'}}));await t.h.refreshResearch();t.h.renderResearchProbe();
  const provenance=JSON.parse(t.$('#r-provenance').textContent);assert.equal(provenance.identity.checkpoint,'initial-id');assert.equal(provenance.calibration.hash,'initial-cal');assert.equal(t.research.overview.identity.checkpoint,'refreshed-id');
});

check('paired submission binds exact label, split, budgets and both configurations',async()=>{
  const t=harness();t.research.overview=goodOverview({operations:{generate:true,score:true,jobs:true,presets:true}});
  t.$('#r-mode').value='experiment';t.$('#r-prompt').value='task';t.$('#r-answers').value=' 9\n';
  t.$('#r-split').value='validation';t.$('#r-requests').value='4';t.$('#r-token-budget').value='8192';
  const job={id:'job-fixture',name:'Paired task',split:'validation',state:'completed',progress:{completed_arms:2,total_arms:2},costs:{requests:2},outcomes:{paired_items:1},cleanup_acknowledged:true};
  t.context.fetchImpl=async url=>response(url.includes('?')?{items:[]}:job);
  await t.h.submitResearch({preventDefault(){}});
  const posted=t.requests.find(r=>r.url==='/api/research/jobs'&&r.opts.method==='POST');
  assert.equal(posted.body.items[0].answer,' 9\n');assert.equal(posted.body.split,'validation');
  assert.deepEqual(posted.body.baseline,{pushes:[]});assert.deepEqual(posted.body.treatment,{pushes:[]});
  assert.deepEqual(posted.body.limits,{max_requests:4,max_tokens:8192,max_wall_seconds:60});
  assert.equal(t.research.probe.id,'job-fixture');assert.equal(t.$('#r-resume').disabled,true);
});
check('persisted cancellation and resume use the selected job, never replay an untracked request',async()=>{
  const t=harness();t.research.overview=goodOverview();
  t.research.probe={mode:'experiment',id:'job-fixture',result:{state:'failed'},overview:t.research.overview};
  t.context.fetchImpl=async url=>response({id:'job-fixture',state:url.endsWith('/cancel')?'canceled':'completed',progress:{completed_arms:0,total_arms:2},costs:{},outcomes:{}});
  await t.h.updateResearchJob('cancel');assert.equal(t.requests[0].url,'/api/research/jobs/job-fixture/cancel');
  assert.equal(t.$('#r-resume').disabled,false);
  await t.h.updateResearchJob('resume');assert.equal(t.requests[1].url,'/api/research/jobs/job-fixture/resume');
});
check('researcher creates one frozen protocol and sends development feedback separately',async()=>{
  const t=harness();t.research.overview=goodOverview({operations:{researcher:true,presets:true}});
  t.$('#r-mode').value='researcher';t.$('#r-prompt').value='Fixed brief';
  t.$('#r-menu').value=JSON.stringify([{id:'a',description:'Control',pushes:[]}]);
  t.$('#r-feedback').value=JSON.stringify([{split:'development',action_id:'a',correct:1,total:4}]);
  t.context.fetchImpl=async url=>response(url.includes('?')?{items:[]}:url.endsWith('/choose')?
    {id:'researcher-fixture',completed:true,action:{id:'a'},cost:{requests:1}}:{id:'researcher-fixture'});
  await t.h.submitResearch({preventDefault(){}});
  const create=t.requests.find(r=>r.url==='/api/research/researchers');
  assert.equal(create.body.brief,'Fixed brief');assert.equal(create.body.feedback,undefined);
  const choose=t.requests.find(r=>r.url.endsWith('/choose'));
  assert.deepEqual(choose.body,{feedback:[{split:'development',action_id:'a',correct:1,total:4}],timeout_seconds:60});
  assert.equal(t.$('#r-episode').value,'researcher-fixture');assert.equal(t.research.probe.mode,'researcher');
});
check('saved target presets are replayed before a direct arm and baseline clears selection',async()=>{
  const t=harness();t.research.overview=goodOverview();t.$('#r-prompt').value='task';t.$('#r-target-preset').value='preset-fixture';
  t.context.fetchImpl=async url=>response(url.endsWith('/replay')?{pushes:[[1,2,.5,'mul']]}:goodGeneration());
  const body=await t.h.researchBody('generate');assert.deepEqual(JSON.parse(JSON.stringify(body.neurons)),[{layer:1,n:2,dose:.5,mode:'mul'}]);
  t.context.fetchImpl=async url=>response(url.endsWith('/restore')?{pushes:[]}:{items:[]});
  await t.h.researchPresetAction('baseline');assert.equal(t.$('#r-target-preset').value,'');assert.equal(t.$('#r-use-neuron').checked,false);
});

check('explicit calibration sends one declared layer and refreshes metadata',async()=>{
  const t=harness();t.research.overview=goodOverview({operations:{generate:true,score:true,calibrate:true}});
  t.$('#r-layer').value='7';t.$('#r-deadline').value='90';
  t.context.fetchImpl=async url=>response(url.endsWith('/calibrate')?{completed:true,counts:{forward_passes:1}}:goodOverview());
  await t.$('#r-calibrate').onclick();
  assert.deepEqual(t.requests[0].body,{layers:[7],timeout_seconds:90});
  assert.equal(t.requests[0].url,'/api/research/calibrate');
  assert.ok(t.requests.some(r=>r.url==='/api/research/overview'));assert.equal(t.research.busy,false);
});
(async()=>{let failures=0;for(const [name,fn] of checks){try{await fn();console.log('PASS '+name);}catch(error){failures++;console.error('FAIL '+name+'\n'+error.stack);}}console.log(`${checks.length-failures}/${checks.length} behavior checks passed (synthetic DOM/runtime fixtures; no real browser or inference).`);process.exitCode=failures?1:0;})().catch(error=>{console.error(error);process.exitCode=1;});
