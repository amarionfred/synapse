// Selection scheduling over the actual client functions. Controlled network
// promises expose stale responses and deliberately slow metadata/labels.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname,'../explorer/static/app.js'),'utf8');
const part = (a,b) => source.slice(source.indexOf(a),source.indexOf(b,source.indexOf(a)));
const selection = part('let selectionController = null;', 'const isFocus =');
const loading = part('async function loadConnections(', '// sign = sign(g * u)');
const status = part('function renderStatus()', '// keeps polling');
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject};}
const tick=()=>new Promise(r=>setImmediate(r));
const link = n => ({up:[{layer:0,n,z:10,sign:1}],down:[{layer:31,n,z:12,sign:-1}]});
function fixture(tab='claude') {
  const elements=new Map();
  function element(id){if(!elements.has(id))elements.set(id,{textContent:'',innerHTML:'',value:'',style:{},dataset:{tab},classList:{add(){},remove(){},contains(){return false;}}});return elements.get(id);}
  const requests=[], draws=[], units=[], geometries=[], labels=[], lenses=[];
  const state={layer:29,unitMap:{},statusText:'idle'};
  const context=vm.createContext({AbortController,Map,Promise,console,state,agentSelecting:false,running:null,webController:null,webKey:null,curTab:()=>tab,
    $:element,document:{querySelector:()=>({dataset:{tab},style:{}})},history:{replaceState(){}},
    focusEl:element('focus'),map:{layerMode:null,select(l,n){this.selected=l==null?null:{layer:l,n};},setConnections(c){draws.push({selected:this.selected,con:c});},setWeb(){},setHoverConnections(){}},scatter:{select(){}},
    clearAgentMarks(){},flushName(){},setFocusCollapsed(){},setNameField(){},renderThread(){},syncTabs(){},renderList(){},renderAgentActions(){},renderExamples(){},renderDetails(){},renderConnections(){},refreshWeb(){},hideTip(){},setLayer:async l=>{state.layer=l;},setTab(){},hoverCache:new Map(),esc:x=>String(x),
    api(url,body,signal){const call={url,signal,d:deferred()};requests.push(call);
      if(url.startsWith('/api/unit?'))units.push(call);
      else if(url.startsWith('/api/connections?'))(url.includes('lens=0')?geometries:labels).push(call);
      else if(url.startsWith('/api/lens?'))lenses.push(call);
      else throw new Error(url);
      return call.d.promise;}
  });
  vm.runInContext(selection+'\n'+loading+'\n'+status+'\nglobalThis.check={openNeuron,loadConnections,loadConnectionLabels,clearSelection,renderStatus};',context);
  return {context,state,requests,draws,units,geometries,labels,lenses,elements,check:context.check};
}
(async()=>{
  let count=0;
  const t=fixture();
  const opening=t.check.openNeuron(29,6872);
  assert.match(t.requests[0].url,/connections.*lens=0/);
  assert.match(t.elements.get('#status').textContent,/Loading links/);
  t.geometries[0].d.resolve(link(7));await tick();
  assert.equal(t.state.con.up[0].n,7);
  assert.equal(t.draws.at(-1).selected.n,6872);
  assert.match(t.elements.get('#status').textContent,/2 links/);
  assert.equal(t.labels.length,0,'A hidden Details tab must not compute neighbor labels');
  assert.equal(t.lenses.length,0,'Root labels must not precede geometry or unit metadata');
  t.check.renderStatus();assert.match(t.elements.get('#status').textContent,/2 links/);
  t.units[0].d.resolve({name:''});await opening;await tick();
  assert.equal(t.lenses.length,1);
  count++;

  const d=fixture('details');
  const first=d.check.openNeuron(29,100);
  d.geometries[0].d.resolve(link(1));await tick();
  assert.equal(d.labels.length,1);assert.equal(d.state.con.up[0].n,1);
  assert.equal(d.draws.at(-1).con.up[0].n,1,'Geometry must draw while labels remain unresolved');
  const next=d.check.openNeuron(29,200);
  assert.equal(d.labels[0].signal.aborted,true);
  d.geometries[1].d.resolve(link(2));await tick();
  d.labels[0].d.resolve(link(99));await tick();
  assert.equal(d.state.con.up[0].n,2,'Late neighbor labels must not replace the current selection');
  d.units[0].d.resolve({});d.units[1].d.resolve({});await Promise.all([first,next]);
  count++;

  const r=fixture();
  const a=r.check.openNeuron(29,1), b=r.check.openNeuron(29,2), a2=r.check.openNeuron(29,1);
  r.geometries[2].d.resolve(link(3));await tick();
  r.geometries[0].d.resolve(link(99));r.geometries[1].d.resolve(link(98));await tick();
  assert.equal(r.state.con.up[0].n,3,'Delayed A→B→A responses must not overwrite the latest A');
  assert.equal(r.draws.at(-1).selected.n,1);
  r.units.forEach(c=>c.d.resolve({}));await Promise.all([a,b,a2]);
  r.check.clearSelection();assert.equal(r.state.con,null);assert.equal(r.elements.get('#status').textContent,'idle');
  count++;
  const mapSource=fs.readFileSync(path.join(__dirname,'../explorer/static/map3d.js'),'utf8');
  const selectSource=mapSource.slice(mapSource.indexOf('    select(layer, n, fly = true) {'),mapSource.indexOf('    setConnections(con) {'));
  const pans=[];
  const mapContext=vm.createContext({selected:null,connections:[],refreshHighlights(){},layerMode:null,
    canvas:{clientWidth:1600},insets:{left:400,right:600},camera:{},homeDist:200,curDist:()=>80,
    point:{x:.5,y:0,z:0},neuronPos(){return {project(){return mapContext.point}};},flyTo(p,d){pans.push(d)}});
  vm.runInContext('globalThis.mapCheck={'+selectSource+'};',mapContext);
  mapContext.mapCheck.select(29,1,false);
  assert.deepEqual(pans,[80],'A point under the right panel must pan into view without changing click zoom');
  mapContext.point={x:-.125,y:0,z:0}; // x=700, inside the 400..1000 gap.
  mapContext.mapCheck.select(29,2,false);assert.equal(pans.length,1,'Visible clicks must retain the camera');
  mapContext.mapCheck.select(29,2,true);assert.equal(pans.length,1);
  mapContext.point={x:-.8,y:0,z:0};
  mapContext.mapCheck.select(0,1,true);assert.equal(pans.at(-1),70,'Occluded navigation must never zoom out');
  count++;
  console.log(`${count} selection regressions passed: early lines, lazy labels, stale-response rejection and panel visibility.`);
})().catch(e=>{console.error(e);process.exitCode=1});
