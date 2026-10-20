import {BENCHMARK_NOTES} from './benchmark-notes.mjs';
import {protocolFor} from './protocols.mjs';
import {DATASETS,CRITERIA,PASS_CHECKS,PASS_CONTEXT,keyOf,select,aggregate,defaultComparators,TARGET_METRICS,targetValue} from './metrics.mjs';

const $=id=>document.getElementById(id);
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=1)=>v==null?'—':v.toLocaleString('en-US',{maximumFractionDigits:n,minimumFractionDigits:n});
const defaultState=()=>({dataset:'ppi',criterion:'ppi',condition:'best',metric:'clustered',targetMetric:'clustered',af3:true,log:true,sort:'rate',target:'',selected:'',targetModels:null,hidden:new Set()});
let state=defaultState(), data, current=[];
const mode=g=>DATASETS[state.dataset].modes[g.dataset];
const isSearch=g=>g.model.includes('search')||(g.model==='genie3'&&g.protocol==='native_pipeline');
const name=g=>data.models[g.model].name + (['bindcraft','bindcraft2'].includes(g.model)?` · ${g.protocol==='one_shot'?'single backbone':'full pipeline'}`:g.model==='genie3'&&g.protocol==='native_pipeline'?' · Search':'');
const recipe=g=>`${mode(g)} · ${g.protocol==='one_shot'?'one shot':'native pipeline'}`;
const color=g=>{
  const c=g.protocol==='native_pipeline'?({genie3:'#967022',bindcraft:'#873C3A',bindcraft2:'#945719'}[g.model]||data.models[g.model].color):data.models[g.model].color;
  if(document.documentElement.dataset.theme!=='dark')return c;
  const h=c.slice(1).match(/../g).map(x=>Math.round(parseInt(x,16)*.65+255*.35));
  return `rgb(${h.join(',')})`;
};

function controls(){
  $('datasets').innerHTML=Object.entries(DATASETS).map(([k,d])=>`<button data-dataset="${k}" aria-pressed="${state.dataset===k}">${d.short}<span>${d.targets}</span></button>`).join('');
  $('condition').innerHTML='<option value="best">Best per method</option><option value="all">All conditioning variants</option>'+Object.entries(DATASETS[state.dataset].modes).map(([k,n])=>`<option value="${k}">${n}</option>`).join('');
  const choices=state.dataset==='sm'?{sm_binder:'Small-molecule binder · RF2AA/RFD3',disco:'Original DISCO'}:{[DATASETS[state.dataset].criterion]:'Common pass criteria'};
  $('criterion').innerHTML=Object.entries(choices).map(([k,n])=>`<option value="${k}">${n}</option>`).join('');
  $('criterion').disabled=state.dataset!=='sm';
  for(const k of ['condition','criterion','sort'])$(k).value=state[k];
  $('af3').checked=state.af3;$('log').checked=state.log;
  document.querySelectorAll('[data-metric]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.metric===state.metric));
}

function render(){
  $('tooltip').hidden=true;
  const rows=select(data.rows,state,data.models);
  current=aggregate(rows,state.af3);
  const sortValue=g=>state.sort==='efficiency'?g[state.metric==='clustered'?'efficiency':'ordinaryEfficiency']:state.sort==='rate'?g[state.metric]:g[state.sort];
  current.sort((a,b)=>(sortValue(b)??-Infinity)-(sortValue(a)??-Infinity)||name(a).localeCompare(name(b)));
  const definition=state.metric==='clustered'?'clustered':'unclustered';
  const selection=state.condition==='best'?`Best of ${DATASETS[state.dataset].variants}, where both were run (benchmark-wide mean ${definition} pass rate).`:state.condition==='all'?'All tested conditioning variants shown separately.':`${DATASETS[state.dataset].modes[state.condition]} conditioning.`;
  $('scope').textContent=`${DATASETS[state.dataset].description} ${current.length} methods · ${new Set(rows.map(r=>r.target)).size} targets · 200 starts per target. ${selection}`;
  $('ranking-note').textContent=state.condition==='best'?'Best conditioning per method; all workflows included.':'All workflows included; conditioning variants remain separate.';
  $('ligand-criteria-note').hidden=state.dataset!=='sm';
  $('benchmark-notes').innerHTML=BENCHMARK_NOTES[state.dataset]+'<p>For models evaluated under both conditioning modes, “Best per method” selects the better completed whole-benchmark arm by the displayed mean pass rate, not a different arm for each target. Distinct workflows—such as single-backbone generation and full pipelines—remain separate methods. All originals, including native failures, remain in each arm’s denominator.</p>';
  $('selection-note').innerHTML=state.target?`Target: <b>${esc(data.labels[state.target]||state.target)}</b>`:state.hidden.size?`${state.hidden.size} configuration(s) hidden · <button id="show-all" class="quiet">Show all</button>`:'';
  $('show-all')?.addEventListener('click',()=>{state.hidden.clear();render()});
  $('criteria-chips').innerHTML=PASS_CHECKS[state.criterion].map(([label,value])=>`<span><i aria-hidden="true">✓</i> ${esc(label)} <b>${esc(value)}</b></span>`).join('');
  $('criteria-detail').textContent=PASS_CONTEXT[state.criterion];
  $('clear-target').hidden=!state.target;
  $('download').disabled=!rows.length;
  drawScatter(current);drawRanking(current);drawDetail();drawTargets();saveView();
}

function shape(g,x,y,size=6){
  const c=color(g), family=data.models[g.model].family, common=`fill="${c}" stroke="var(--panel)" stroke-width="1.5"`;
  let s=family==='hallucination'?`<rect x="${x-size}" y="${y-size}" width="${2*size}" height="${2*size}" rx="1" ${common}/>`:family==='refinement'?`<path d="M ${x} ${y-size-1} L ${x+size+1} ${y+size} L ${x-size-1} ${y+size} Z" ${common}/>`:`<circle cx="${x}" cy="${y}" r="${size}" ${common}/>`;
  if(isSearch(g))s+=`<path d="M ${x} ${y-size-5} L ${x+size+5} ${y} L ${x} ${y+size+5} L ${x-size-5} ${y} Z" fill="none" stroke="${c}" stroke-width="1.2"/>`;
  if(g.model.startsWith('rfd4'))s=`<circle cx="${x}" cy="${y}" r="${size+6}" fill="${c}" opacity=".1"/>`+s;
  return s;
}

function drawScatter(groups){
  const gs=groups.filter(g=>g.throughput!=null&&g.throughput>0);
  const unavailable=groups.length-gs.length;
  $('plot-note').textContent=`Recorded warm generation includes sequence design and internal folding/selection. Common AF3 ${state.af3?'included':'excluded'}.${unavailable?` ${unavailable} configuration lacks complete timing; its pass rate remains in the comparison.`:''}`;
  if(!gs.length){$('scatter').innerHTML='<p class="empty">No configurations with complete timing for these filters.</p>';return}
  const W=Math.max(390,Math.min(690,$('scatter').clientWidth)),H=405,L=60,R=28,T=34,B=57,w=W-L-R,h=H-T-B;
  let xmin=state.log?Math.pow(10,Math.floor(Math.log10(Math.min(...gs.map(g=>g.throughput))))):0;
  let xmax=Math.max(...gs.map(g=>g.throughput))*1.18;
  if(state.log)xmax=Math.pow(10,Math.ceil(Math.log10(xmax)*2)/2);
  const ymax=Math.min(105,Math.max(5,Math.ceil(Math.max(...gs.map(g=>g[state.metric]))*1.17/5)*5));
  const x=v=>L+(state.log?(Math.log10(v)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin)):v/xmax)*w;
  const y=v=>T+h-v/ymax*h;
  let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="chart-access-title"><title id="chart-access-title">${state.metric} pass rate versus designs per 10 GPU-minutes for ${DATASETS[state.dataset].name}</title>`;
  for(let i=0;i<=5;i++){const v=i*ymax/5;out+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" class="grid-line"/><text x="${L-11}" y="${y(v)+3}" text-anchor="end" class="tick">${fmt(v,v%1?1:0)}%</text>`}
  let ticks=[];
  if(state.log){for(let e=Math.floor(Math.log10(xmin));e<=Math.ceil(Math.log10(xmax));e++)for(const m of [1,2,5]){const v=m*10**e;if(v>=xmin&&v<=xmax)ticks.push(v)}}
  else ticks=Array.from({length:6},(_,i)=>i*xmax/5);
  let prev=-Infinity;
  for(const v of ticks){if(x(v)-prev<44)continue;prev=x(v);out+=`<line x1="${x(v)}" y1="${T}" x2="${x(v)}" y2="${T+h}" class="grid-line"/><text x="${x(v)}" y="${T+h+23}" text-anchor="middle" class="tick">${fmt(v,v<1?2:v<10?1:0)}</text>`}
  out+=`<line x1="${L}" y1="${T+h}" x2="${W-R}" y2="${T+h}" class="axis-line"/><text x="${L+w/2}" y="${H-8}" text-anchor="middle" class="axis-label">Designs per 10 GPU-minutes${state.log?' · log scale':''}</text><text transform="translate(16 ${T+h/2}) rotate(-90)" text-anchor="middle" class="axis-label">${state.metric==='clustered'?'Clustered':'Unclustered'} pass rate</text>`;
  const points=gs.map(g=>({g,x:x(g.throughput),y:y(g[state.metric])}));
  // Labels use collision-tested candidate positions; every point retains a tooltip.
  const boxes=points.map(p=>({x:p.x-9,y:p.y-9,w:18,h:18}));
  const collide=(a,b)=>a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y;
  const labelFor=g=>name(g).replace('RFD4-Proteina · ','RFD4 · ').replace(' · single backbone','').replace(' · Codesign','').replace(' · LigandMPNN',' · MPNN');
  points.sort((a,b)=>(b.g.key===state.selected)-(a.g.key===state.selected)||Number(b.g.model.startsWith('rfd4'))-Number(a.g.model.startsWith('rfd4')));
  for(const p of points){const g=p.g,selected=!state.selected||state.selected===g.key;out+=`<g class="dot" tabindex="0" role="button" data-point="${esc(g.key)}" aria-label="${esc(name(g)+', '+recipe(g)+', '+fmt(g[state.metric])+' percent')}" opacity="${selected?1:.28}">${shape(g,p.x,p.y,g.model.startsWith('rfd4')?7:6)}</g>`;
    if(state.selected&&state.selected!==g.key)continue;
    let label=labelFor(g);if(gs.filter(q=>q.model===g.model).length>1)label+=g.dataset==='ppi_all_atom'?' (AA)':g.dataset==='ppi_backbone'?' (BB)':g.dataset==='mcsa_unfixed'?' (partially fixed)':g.dataset.endsWith('_unfixed')?' (unfixed)':' (fixed)';
    const tw=label.length*5.1,th=14;
    const candidates=[[12,-19],[12,7],[-tw-12,-19],[-tw-12,7],[-tw/2,-28],[-tw/2,16]];
    for(const [dx,dy] of candidates){const box={x:p.x+dx,y:p.y+dy,w:tw,h:th};if(box.x<L||box.x+tw>W-R||box.y<5||box.y+th>T+h-4||boxes.some(b=>collide(box,b)))continue;boxes.push(box);out+=`<text x="${box.x}" y="${box.y+10}" class="point-label" font-weight="${g.model.startsWith('rfd4')?600:400}">${esc(label)}</text>`;break}
  }
  $('scatter').innerHTML=out+'</svg>';
  document.querySelectorAll('[data-point]').forEach(el=>{
    const g=groups.find(g=>g.key===el.dataset.point);
    el.addEventListener('pointerenter',e=>showTip(e,g));el.addEventListener('pointermove',positionTip);
    el.addEventListener('pointerleave',hideTip);el.addEventListener('focus',e=>showTip(e,g));el.addEventListener('blur',hideTip);
    const choose=()=>{state.selected=g.key===state.selected?'':g.key;render()};el.addEventListener('click',choose);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose()}});
  });
}

function drawRanking(groups){
$('ranking').innerHTML=groups.length?groups.map(g=>`<button class="rank-row ${state.selected===g.key?'selected':''}" data-key="${esc(g.key)}" aria-pressed="${state.selected===g.key}"><div><div class="method-name"><i style="background:${color(g)};border-radius:${data.models[g.model].family==='hallucination'?'1':'50'}%"></i>${esc(name(g))}</div><div class="method-note">${esc(recipe(g))} · ${g.solved}/${g.targets} solved</div><div class="rank-bar"><span style="width:${g[state.metric]}%;background:${color(g)}"></span></div></div><div class="rank-number">${fmt(g[state.metric])}%<small>${g.efficiency==null?'Timing unavailable':fmt(state.metric==='clustered'?g.efficiency:g.ordinaryEfficiency)+' / GPU-h'}</small></div></button>`).join(''):'<p class="empty">No matching methods in this snapshot. Try another conditioning mode or reset.</p>';
  document.querySelectorAll('[data-key]').forEach(el=>el.addEventListener('click',()=>{state.selected=state.selected===el.dataset.key?'':el.dataset.key;render()}));
}

function drawDetail(){
  const dialog=$('detail'),g=current.find(g=>g.key===state.selected);if(!g){if(dialog.open)dialog.close();return}
  const kpi=(value,label)=>`<div class="kpi"><strong>${value}</strong><span>${label}</span></div>`;
  const method=data.models[g.model].protocol_methods?.[g.protocol]||data.models[g.model].method;
  const protocol=protocolFor(g);
  dialog.innerHTML=`<div class="detail-head"><div><h3 id="protocol-title">${esc(name(g))}</h3><p>${esc(DATASETS[state.dataset].name)} · ${esc(recipe(g))}${state.target?' · '+esc(data.labels[state.target]||state.target):''}</p></div><button id="close-detail" class="quiet" aria-label="Close protocol details" autofocus>✕</button></div>
    <section class="protocol-section"><h4>Workflow</h4><p>${esc(protocol.procedure)}</p><a href="${esc(method.reference_url)}" target="_blank" rel="noopener">${esc(method.reference_label)} ↗</a></section>
    <section class="protocol-section"><h4>Benchmark changes</h4><p>${esc(protocol.adaptation)}</p></section>
    ${protocol.note?`<p class="protocol-note">${esc(protocol.note)}</p>`:''}
    <div class="kpis">${kpi(`${g.passes.toLocaleString()} / ${g.attempts.toLocaleString()}`,'Passing originals')}${kpi(g.clusters.toLocaleString(),'Distinct passing folds')}${kpi(`${g.solved} / ${g.targets}`,'Tasks with any pass')}${kpi(fmt(g.seconds)+' s',`GPU seconds / attempt${state.af3?' · with AF3':''}`)}${kpi(fmt(g.efficiency),'Mean unique passers / GPU-hour')}</div>
    <button id="hide-method" class="quiet">Hide this configuration</button>`;
  $('close-detail').onclick=()=>{state.selected='';render()};$('hide-method').onclick=()=>{state.hidden.add(g.key);state.selected='';render()};
  if(!dialog.open)dialog.showModal();
}

function targetGroups(){
  const rows=select(data.rows,{...state,target:''},data.models);
  return aggregate(rows,state.af3).sort((a,b)=>b[state.metric]-a[state.metric]);
}
function targetRows(){
  const q=$('target-search').value.trim().toLowerCase();
  return targetGroups().filter(g=>state.targetModels?.has(g.key)).flatMap(g=>g.rows)
    .filter(r=>(r.target+' '+(data.labels[r.target]||'')).toLowerCase().includes(q));
}
function drawTargets(){
  const metric=TARGET_METRICS[state.targetMetric],value=r=>targetValue(r,state.targetMetric,state.af3);
  document.querySelectorAll('[data-target-metric]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.targetMetric===state.targetMetric));
  $('target-metric-note').textContent=metric.percent?'Per target: '+(state.targetMetric==='raw'?'passing designs':'distinct passing folds')+' ÷ 200 attempts.':`Per target: ${state.targetMetric==='passes_hour'?'passing designs':'distinct passing folds'} ÷ GPU-hours for all 200 attempts; common AF3 ${state.af3?'included':'excluded'}.`;
  const all=targetGroups();
  // Start with a manageable comparison; every configuration is available below.
  if(state.targetModels===null)state.targetModels=defaultComparators(all,state.metric);
  $('target-models').innerHTML=all.map(g=>`<label><input type="checkbox" data-target-model="${esc(g.key)}" ${state.targetModels.has(g.key)?'checked':''}><i style="background:${color(g)}"></i><span>${esc(name(g))}<small>${esc(recipe(g))}</small></span></label>`).join('');
  const groups=all.filter(g=>state.targetModels.has(g.key)),rows=targetRows();
  const targets=[...new Set(rows.map(r=>r.target))].sort();
  $('target-selection-count').textContent=`· ${groups.length} of ${all.length}`;
  $('target-download').disabled=!rows.length;
  $('target-chart-note').textContent='Model checkboxes affect this chart and its CSV only. Scroll horizontally for more targets; select a bar for exact counts and protocol details. Missing results are marked —, not zero.';
  document.querySelectorAll('[data-target-model]').forEach(el=>el.onchange=()=>{
    el.checked?state.targetModels.add(el.dataset.targetModel):state.targetModels.delete(el.dataset.targetModel);drawTargets();
  });
  if(!groups.length||!targets.length){$('target-chart').innerHTML='<p class="empty">Select models and a matching target to compare.</p>';return}
  const H=390,L=64,R=20,T=24,B=86,h=H-T-B;
  const slot=Math.max(42,groups.length*6+14),W=Math.max($('target-chart').clientWidth,targets.length*slot+L+R);
  const step=(W-L-R)/targets.length,bw=Math.min(25,(step-14)/groups.length);
  const peak=Math.max(0,...rows.map(value).filter(v=>v!=null));
  const unit=metric.percent?5:10**Math.floor(Math.log10(peak||1));
  const max=metric.percent?Math.min(100,Math.max(5,Math.ceil(peak/5)*5)):Math.max(unit,Math.ceil(peak/unit)*unit);
  const y=v=>T+h-h*v/max;
  let svg=`<svg width="${W}" height="${H}" role="img" aria-label="${metric.label} by target"><text class="axis-label" x="${L}" y="13">${metric.axis}</text>`;
  const digits=max<1?Math.ceil(-Math.log10(max/5)):1;
  for(let i=0;i<=5;i++){const value=max*i/5;svg+=`<line x1="${L}" y1="${y(value)}" x2="${W-R}" y2="${y(value)}" class="grid-line"/><text x="${L-9}" y="${y(value)+4}" class="tick" text-anchor="end">${fmt(value,value%1?digits:0)}${metric.percent?'%':''}</text>`}
  targets.forEach((t,ti)=>{
    const mid=L+step*(ti+.5);
    svg+=`<text transform="translate(${mid} ${T+h+18}) rotate(-40)" text-anchor="end" class="target-tick">${esc(data.labels[t]||t)}</text>`;
    groups.forEach((g,gi)=>{
      const r=g.rows.find(r=>r.target===t),x=mid-groups.length*bw/2+gi*bw+1,width=bw-2;
      if(!r){svg+=`<text class="tick" x="${x+width/2}" y="${T+h-4}" text-anchor="middle">—</text>`;return}
      const v=value(r),label=v==null?'Timing unavailable':fmt(v)+(metric.percent?'%':'');
      svg+=`<g data-cell="${esc(g.key)}" data-target="${esc(t)}" data-value="${v??''}" role="button" tabindex="0" aria-label="${esc(name(g)+', '+recipe(g)+', '+t+', '+metric.label+' '+label)}"><rect x="${x}" y="${T}" width="${width}" height="${h}" fill="transparent"/>${v==null?`<text class="tick" x="${x+width/2}" y="${T+h-4}" text-anchor="middle">—</text>`:`<rect class="target-bar" x="${x}" y="${y(v)}" width="${width}" height="${h*v/max}" fill="${color(g)}"/>${v===0?`<line x1="${x}" x2="${x+width}" y1="${y(0)}" y2="${y(0)}" stroke="${color(g)}" stroke-width="2"/>`:''}`}</g>`;
    });
  });
  $('target-chart').innerHTML=svg+`<text class="axis-label" x="${L+(W-L-R)/2}" y="${H-4}" text-anchor="middle">Targets</text></svg>`;
  bindTargetEvents(groups);
}

function bindTargetEvents(groups){
  document.querySelectorAll('[data-cell]').forEach(el=>{
    const g=groups.find(g=>g.key===el.dataset.cell), r=g.rows.find(r=>r.target===el.dataset.target);
    const single=r?aggregate([r],state.af3)[0]:null;
    el.addEventListener('pointerenter',e=>{if(single)showTip(e,single,el.dataset.target)});el.addEventListener('pointermove',positionTip);el.addEventListener('pointerleave',hideTip);
    el.addEventListener('focus',e=>{if(single)showTip(e,single,el.dataset.target)});el.addEventListener('blur',hideTip);
    const choose=()=>{state.target=el.dataset.target;state.selected=el.dataset.cell;render()};el.onclick=choose;
    el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose()}};
  });
}

function showTip(e,g,target=''){
  $('tooltip').innerHTML=`<strong>${esc(name(g))}</strong><p>${esc(recipe(g))}${target?' · '+esc(data.labels[target]||target):''}</p><div class="tt-metrics">${target?`<span>${TARGET_METRICS[state.targetMetric].label}</span><b>${fmt(targetValue(g.rows[0],state.targetMetric,state.af3))}${TARGET_METRICS[state.targetMetric].percent?'%':''}</b>`:''}<span>Passing originals</span><b>${g.passes} / ${g.attempts}</b><span>Passing clusters</span><b>${g.clusters}</b><span>Unclustered / clustered</span><b>${fmt(g.raw)}% / ${fmt(g.clustered)}%</b><span>Designs / 10 GPU-min</span><b>${fmt(g.throughput)}</b><span>Unique passers / GPU-h</span><b>${fmt(g.efficiency)}</b></div>`;
  $('tooltip').hidden=false;positionTip(e);
}
function positionTip(e){const b=e.currentTarget?.getBoundingClientRect(),x=e.clientX??b?.right??0,y=e.clientY??b?.top??0,tip=$('tooltip');tip.style.left=Math.max(8,Math.min(x+14,innerWidth-tip.offsetWidth-10))+'px';tip.style.top=Math.max(8,Math.min(y+14,innerHeight-tip.offsetHeight-10))+'px'}
function hideTip(){$('tooltip').hidden=true}
function saveView(){const p=new URLSearchParams();for(const k of ['dataset','criterion','condition','metric','targetMetric','af3','log','sort','target'])p.set(k,state[k]);history.replaceState(null,'',location.pathname+'?'+p.toString())}
function restoreView(){const p=new URLSearchParams(location.search);if(p.get('dataset') in DATASETS)state.dataset=p.get('dataset');state.criterion=DATASETS[state.dataset].criterion;
  if(state.dataset==='sm'&&p.get('criterion')==='disco')state.criterion='disco';
  if(p.get('condition') in DATASETS[state.dataset].modes)state.condition=p.get('condition');
  // Old URLs exposed protocol/family filters; do not let them hide workflows.
  if(!p.has('protocol')&&!p.has('family')&&['all','best'].includes(p.get('condition')))state.condition=p.get('condition');
  if(Object.hasOwn(TARGET_METRICS,p.get('targetMetric')))state.targetMetric=p.get('targetMetric');
  if(p.get('metric')==='raw')state.metric='raw';for(const k of ['af3','log'])if(p.get(k)==='false')state[k]=false;
  if(['rate','efficiency','throughput','solved'].includes(p.get('sort')))state.sort=p.get('sort');
  if(data.rows.some(r=>r.dataset in DATASETS[state.dataset].modes&&r.target===p.get('target')))state.target=p.get('target');
}
function download(selectedRows=select(data.rows,state,data.models),targetChart=false){const rows=selectedRows.map(r=>({...r,view_include_af3:state.af3,view_metric:targetChart?state.targetMetric:state.metric,...(targetChart?{view_value:targetValue(r,state.targetMetric,state.af3),view_units:TARGET_METRICS[state.targetMetric].percent?'percent':'per GPU-hour'}:{}),view_condition:state.condition,view_selection:state.condition==='best'?`Best completed conditioning arm by whole-benchmark mean ${state.metric} pass rate`:'Explicit conditioning selection'}));if(!rows.length)return;
  saveCSV(rows,`benchmarks-${state.dataset}-${state.criterion}-${data.meta.snapshot}.csv`);
}
function saveCSV(rows,filename){
  const keys=[...new Set(rows.flatMap(Object.keys))],quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
  const csv=[keys.join(','),...rows.map(r=>keys.map(k=>quote(r[k])).join(','))].join('\r\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

try{
  const response=await fetch(new URL('./data.json',import.meta.url));if(!response.ok)throw Error(`Snapshot HTTP ${response.status}`);data=await response.json();
  restoreView();controls();render();
  $('download-all').onclick=e=>{e.preventDefault();saveCSV(data.rows,`benchmarks-all-${data.meta.snapshot}.csv`)};
  $('datasets').onclick=e=>{const b=e.target.closest('[data-dataset]');if(!b)return;state={...defaultState(),dataset:b.dataset.dataset,criterion:DATASETS[b.dataset.dataset].criterion,af3:state.af3,metric:state.metric,targetMetric:state.targetMetric,log:state.log};$('target-search').value='';controls();render();$('ranking').scrollTop=0};
  for(const k of ['condition','criterion','sort'])$(k).onchange=()=>{state[k]=$(k).value;state.selected='';if(k!=='sort')state.targetModels=null;render();$('ranking').scrollTop=0};
  for(const k of ['af3','log'])$(k).onchange=()=>{state[k]=$(k).checked;render()};
  document.querySelectorAll('[data-metric]').forEach(b=>b.onclick=()=>{state.metric=b.dataset.metric;state.targetModels=null;controls();render()});
  document.querySelectorAll('[data-target-metric]').forEach(b=>b.onclick=()=>{state.targetMetric=b.dataset.targetMetric;hideTip();drawTargets();saveView()});
  $('target-search').oninput=drawTargets;$('clear-target').onclick=()=>{state.target='';render()};
  $('target-select-all').onclick=()=>{state.targetModels=new Set(targetGroups().map(g=>g.key));drawTargets()};
  $('target-select-none').onclick=()=>{state.targetModels=new Set();drawTargets()};
  $('target-download').onclick=()=>download(targetRows(),true);
  $('detail').addEventListener('cancel',e=>{e.preventDefault();state.selected='';render()});
  $('reset').onclick=()=>{state=defaultState();$('target-search').value='';controls();render()};$('download').onclick=()=>download();
  window.addEventListener('site-theme-change',render);
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>drawScatter(current),100)});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){hideTip();state.selected='';render()}});
}catch(error){$('scatter').innerHTML=`<p class="empty">Unable to load the benchmark snapshot. ${esc(error.message)}<br>Serve this directory over HTTP; direct file previews cannot fetch the data.</p>`;console.error(error)}
