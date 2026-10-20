// Pure measurement functions shared by the browser and node tests.
export const DATASETS = {
  ppi: {name:'Protein–protein binding', short:'PPI', targets:24, criterion:'ppi', modes:{ppi_backbone:'Backbone',ppi_all_atom:'All-atom'}, description:'Union of cases from the Proteina-Complex and RFdiffusion3 papers.', variants:'backbone-only or all-atom target conditioning'},
  sm: {name:'Small-molecule binding', short:'Small molecules', targets:77, criterion:'sm_binder', modes:{studio_figure_2c_unfixed:'Unfixed ligand',studio_figure_2c_fixed:'Fixed ligand'}, description:'77 small-molecule tasks from DISCO Figure 2C (STUDIO subset).', variants:'fixed or unfixed ligand coordinates'},
  ame: {name:'Enzyme active-site scaffolding', short:'Enzymes', targets:37, criterion:'ame', modes:{mcsa_fixed:'Fully fixed ligand',mcsa_unfixed:'Partially fixed ligand'}, description:'Atomic Motif Enzyme (AME) Benchmark: 37 of 41 tasks.', variants:'partially or fully fixed ligand coordinates'},
  dna: {name:'DNA binding', short:'DNA', targets:3, criterion:'dna', modes:{dna_unfixed:'Unfixed DNA',dna_fixed:'Fixed DNA'}, description:'Three DNA targets; 120–150-residue protein binders.', variants:'fixed or unfixed DNA coordinates'},
};
export const CRITERIA = {
  ppi:'AF3 binder pTM ≥ 0.80; minimum interface PAE ≤ 1.5 Å; target-aligned binder backbone RMSD ≤ 2.5 Å. Required target mapping and conditioning must be valid.',
  sm_binder:'AF3 interface PAE ≤ 1.5 Å; chain-of-interest ipTM ≥ 0.80; protein backbone RMSD ≤ 1.5 Å; protein-aligned ligand RMSD ≤ 5 Å.',
  disco:'Protein backbone RMSD ≤ 2 Å and maximum protein-aligned ligand-centroid displacement ≤ 2 Å. No PAE or ipTM gate. Not mathematically nested within the small-molecule binder criterion.',
  ame:'Protein backbone RMSD ≤ 2 Å; required motif all-atom RMSD ≤ 1.5 Å; AF3 minimum interface PAE strictly < 1.5 Å. Native unmatched or sequence-failed motifs fail. Clashes are diagnostic only.',
  dna:'AF3 minimum interface PAE ≤ 1.5 Å and DNA-aligned binder backbone RMSD ≤ 8 Å. All target DNA atoms are used for alignment in both modes.',
};
export const PASS_CHECKS = {
  ppi:[['Binder pTM','≥ 0.80'],['Minimum interface PAE','≤ 1.5 Å'],['Binder backbone RMSD','≤ 2.5 Å'],['Target constraints','Valid']],
  sm_binder:[['Minimum interface PAE','≤ 1.5 Å'],['Binder ipTM','≥ 0.80'],['Protein backbone RMSD','≤ 1.5 Å'],['Ligand RMSD','≤ 5 Å']],
  disco:[['Protein backbone RMSD','≤ 2 Å'],['Max ligand-centroid displacement','≤ 2 Å']],
  ame:[['Protein backbone RMSD','≤ 2 Å'],['Catalytic motif RMSD','≤ 1.5 Å'],['Minimum interface PAE','< 1.5 Å'],['Required motifs','Mapped & sequence-valid']],
  dna:[['Minimum interface PAE','≤ 1.5 Å'],['Binder backbone RMSD','≤ 8 Å']],
};
export const PASS_CONTEXT = {
  ppi:'Align the design and AF3 refold on the target backbone, then measure binder backbone RMSD. Hotspot mutations or target substitutions within 5 Å of the binder in design or refold fail; distal substitutions may pass. pTM is displayed on a 0–1 scale (80 on the recorded 0–100 scale).',
  sm_binder:'Align design and AF3 refold on the protein backbone; score that backbone and the ligand in the same alignment. ipTM is the chain-of-interest value. Every listed gate must pass; the original DISCO criterion can be selected separately.',
  disco:'Align on the protein backbone; measure backbone RMSD and the largest ligand-centroid displacement. There is no PAE or ipTM gate. This criterion and the small-molecule binder criterion are not mathematically nested.',
  ame:'Protein backbone RMSD uses a backbone alignment. Motif RMSD separately aligns mapped conditioned protein atoms to the input references, respecting coordinate groups. Missing required motif matches or sequence violations fail; missing metrics are not zero. PAE must be strictly below 1.5 Å. Clashes are diagnostic, not a pass gate.',
  dna:'Align design and AF3 refold on all target DNA atoms, then score the binder backbone. The same alignment and thresholds apply to fixed and unfixed DNA.',
};
export const keyOf = r => [r.model,r.protocol,r.dataset].join('|');
export const mean = a => a.length ? a.reduce((s,x)=>s+x,0)/a.length : null;
export function cost(r, af3) {
  if(r.generation_gpu_s == null || (af3 && r.af3_gpu_s == null)) return null;
  return r.generation_gpu_s + (af3 ? r.af3_gpu_s : 0);
}
export const TARGET_METRICS={
  raw:{label:'Pass rate',axis:'Pass rate (%)',percent:true},
  clustered:{label:'Clustered pass rate',axis:'Clustered pass rate (%)',percent:true},
  passes_hour:{label:'Passes / GPU-hour',axis:'Passes / GPU-hour',percent:false},
  unique_hour:{label:'Unique passes / GPU-hour',axis:'Unique passes / GPU-hour',percent:false},
};
export function targetValue(r, metric, af3){
  if(metric==='raw')return 100*r.passes/r.attempts;
  if(metric==='clustered')return 100*r.clusters/r.attempts;
  const seconds=cost(r,af3);
  if(!Number.isFinite(seconds)||seconds<=0)return null;
  return 3600*(metric==='passes_hour'?r.passes:r.clusters)/seconds;
}
export function select(rows, s) {
  let selected=rows.filter(r=>r.dataset in DATASETS[s.dataset].modes && r.criterion===s.criterion);
  if(s.condition==='best') {
    // Choose a complete benchmark arm before any target drilldown. Never pool
    // protocols or choose a different conditioning winner for each target.
    const metric=s.metric==='raw'?'raw':'clustered', other=metric==='raw'?'clustered':'raw';
    const groups=aggregate(selected,false).sort((a,b)=>b[metric]-a[metric]||b[other]-a[other]||a.key.localeCompare(b.key));
    const winners=new Set(), seen=new Set();
    for(const g of groups){const identity=g.model+'|'+g.protocol;if(!seen.has(identity)){seen.add(identity);winners.add(g.key)}}
    selected=selected.filter(r=>winners.has(keyOf(r)));
  } else if(s.condition!=='all')selected=selected.filter(r=>r.dataset===s.condition);
  return selected.filter(r=>(!s.target||r.target===s.target)&&!s.hidden.has(keyOf(r)));
}

export function defaultComparators(groups, metric='clustered') {
  const sorted=[...groups].sort((a,b)=>b[metric]-a[metric]||a.key.localeCompare(b.key));
  const chosen=[];
  const high=sorted.find(g=>g.model==='rfd4_proteina_high');
  const search=sorted.find(g=>g.model.startsWith('rfd4')&&g.model.includes('search'));
  if(search)chosen.push(search);if(high)chosen.push(high);
  const models=new Set();
  for(const g of sorted){if(g.model.startsWith('rfd4')||models.has(g.model))continue;chosen.push(g);models.add(g.model);if(models.size===3)break}
  return new Set(chosen.map(g=>g.key));
}
export function aggregate(rows, af3) {
  const groups = new Map();
  rows.forEach(r => {const k=keyOf(r);if(!groups.has(k)) groups.set(k,[]);groups.get(k).push(r)});
  return [...groups].map(([key,rs]) => {
    const costs=rs.map(r=>cost(r,af3)), complete=costs.every(c=>c!=null&&c>0);
    const total=complete?costs.reduce((a,b)=>a+b,0):null, attempts=rs.reduce((s,r)=>s+r.attempts,0);
    return {key,model:rs[0].model,protocol:rs[0].protocol,dataset:rs[0].dataset,rows:rs,
      targets:rs.length,attempts,passes:rs.reduce((s,r)=>s+r.passes,0),clusters:rs.reduce((s,r)=>s+r.clusters,0),
      raw:mean(rs.map(r=>100*r.passes/r.attempts)),clustered:mean(rs.map(r=>100*r.clusters/r.attempts)),
      solved:rs.filter(r=>r.passes>0).length,solved5:rs.filter(r=>r.passes/r.attempts>=.05).length,
      throughput:complete?600*attempts/total:null,
      seconds:complete?total/attempts:null,
      efficiency:complete?mean(rs.map((r,i)=>3600*r.clusters/costs[i])):null,
      ordinaryEfficiency:complete?mean(rs.map((r,i)=>3600*r.passes/costs[i])):null,
      timed:costs.filter(c=>c!=null&&c>0).length,
    };
  });
}
