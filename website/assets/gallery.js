/* Original paper coordinates; Northern Nights styling adapted for live WebGL.
   No inferred hydrogen bonds are drawn. The source CIF/PDB files are unchanged. */
(() => {
  const spectrum=['#186C70','#3B5A8D','#463D73','#8D52AD'];
  const elements={C:'#667078',N:'#416FA9',O:'#B94B47',S:'#B99637',P:'#C7782D'};
  const scenes=[
    {id:'nanopore',file:'p131_nanopore/source.pdb',format:'pdb',protein:{},
      helices:[[2,7],[13,28],[35,48],[51,63],[74,83],[85,98],[100,112],[116,132],[134,151],[153,165],[169,178],[181,193],[197,212],[216,225],[228,240],[244,259],[263,271],[273,281],[285,297]]},
    {id:'ligand',file:'acetyl_coa_barrel/source.cif',format:'cif',protein:{chain:'B'},ligand:{chain:'A'},camera:[0.24333,-0.95727,0.15631,0.07005,0.17808,0.98152,-0.96741,-0.22789,0.11039]},
    {id:'dna',file:'dna_ggtccgattccac_step40_af3/source.cif',format:'cif',protein:{chain:'C'},dna:{chain:['A','B']},camera:[-0.50382,-0.59426,-0.62691,-0.72936,-0.09619,0.67734,-0.46281,0.79850,-0.38496],
      helices:[[2,21],[25,33],[36,49],[55,76],[79,101],[107,125]]},
    {id:'nanobody',file:'nanobodies/ebpa_nanobody_design/source.cif',format:'cif',protein:{chain:'H'},target:{chain:'T'},camera:[0.46031,0.86756,-0.18828,0.88022,-0.47359,-0.03023,-0.11540,-0.15182,-0.98165]}
  ];
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const entries=[];
  function cameraQuaternion(m){
    const trace=m[0]+m[4]+m[8];let x,y,z,w,s;
    if(trace>0){s=Math.sqrt(trace+1)*2;w=s/4;x=(m[7]-m[5])/s;y=(m[2]-m[6])/s;z=(m[3]-m[1])/s}
    else if(m[0]>m[4]&&m[0]>m[8]){s=Math.sqrt(1+m[0]-m[4]-m[8])*2;w=(m[7]-m[5])/s;x=s/4;y=(m[1]+m[3])/s;z=(m[2]+m[6])/s}
    else if(m[4]>m[8]){s=Math.sqrt(1+m[4]-m[0]-m[8])*2;w=(m[2]-m[6])/s;x=(m[1]+m[3])/s;y=s/4;z=(m[5]+m[7])/s}
    else{s=Math.sqrt(1+m[8]-m[0]-m[4])*2;w=(m[3]-m[1])/s;x=(m[2]+m[6])/s;y=(m[5]+m[7])/s;z=s/4}
    const n=Math.hypot(x,y,z,w);return [x/n,y/n,z/n,w/n];
  }
  function gradient(atom,lo,hi){
    const t=Math.max(0,Math.min(2.99999,3*(atom.resi-lo)/Math.max(1,hi-lo)));
    const i=Math.floor(t),f=t-i,a=parseInt(spectrum[i].slice(1),16),b=parseInt(spectrum[i+1].slice(1),16);
    return [16,8,0].reduce((color,shift)=>color|(Math.round(((a>>shift)&255)*(1-f)+((b>>shift)&255)*f)<<shift),0);
  }
  function motion(entry){
    entry.spinning=!reduced.matches&&!document.hidden&&entry.visible&&!entry.paused;
    entry.viewer.spin(entry.spinning?'y':false,0.18);
  }
  const observer=new IntersectionObserver(changes=>{
    for(const change of changes){const entry=entries.find(e=>e.node===change.target);if(entry){entry.visible=change.isIntersecting;motion(entry)}}
  },{threshold:0.05});
  async function setup(scene){
    const node=document.getElementById(`molecule-${scene.id}`);
    try{
      const response=await fetch(`assets/${scene.file}.gz`);if(!response.ok)throw Error(`Structure ${response.status}`);
      const text=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).text();
      const viewer=$3Dmol.createViewer(node,{backgroundAlpha:0,antialias:true,disableFog:true});
      viewer.setBackgroundColor(getComputedStyle(document.documentElement).getPropertyValue('--bg').trim(),0);
      viewer.setProjection('orthographic');
      const model=viewer.addModel(text,scene.format,{doAssembly:false});
      const protein=model.selectedAtoms(scene.protein).filter(a=>!a.hetflag);
      if(!protein.length)throw Error('No protein atoms found');
      if(scene.helices)for(const atom of protein)atom.ss=scene.helices.some(([a,b])=>atom.resi>=a&&atom.resi<=b)?'h':'c';
      const residues=protein.map(a=>a.resi),lo=Math.min(...residues),hi=Math.max(...residues);
      viewer.setStyle({},{});
      viewer.setStyle(scene.protein,{cartoon:{colorfunc:atom=>gradient(atom,lo,hi),thickness:0.35,arrows:true}});
      if(scene.ligand){
        viewer.setStyle(scene.ligand,{stick:{radius:0.22,colorscheme:elements},sphere:{scale:0.22,colorscheme:elements}});
        viewer.addStyle({chain:'B',resi:[38,60,69,83,85,92],not:{atom:['N','CA','C','O']}},{stick:{radius:0.13,colorscheme:{...elements,C:'#C4963B'}}});
      }
      if(scene.dna)viewer.setStyle(scene.dna,{cartoon:{color:'#3B5A8D',thickness:0.3},stick:{radius:0.1,colorscheme:{...elements,C:'#3B5A8D'}}});
      if(scene.target){
        viewer.setStyle(scene.target,{cartoon:{color:'#8B969D',opacity:0.5}});
        await viewer.addSurface($3Dmol.SurfaceType.MS,{color:'#DADADA',opacity:1},scene.target);
      }
      viewer.zoomTo();viewer.zoom(1.85);
      if(scene.camera){const view=viewer.getView();view.splice(4,4,...cameraQuaternion(scene.camera));viewer.setView(view)}
      viewer.render();
      node.dataset.ready='true';node.dataset.atoms=String(model.selectedAtoms({}).length);
      const entry={node,viewer,visible:true,paused:false};entries.push(entry);observer.observe(node);motion(entry);
      // Space pauses motion without adding visible viewer controls.
      node.addEventListener('keydown',event=>{if(event.code==='Space'){event.preventDefault();entry.paused=!entry.paused;motion(entry)}});
      node.addEventListener('pointerdown',()=>{entry.paused=true;motion(entry)});
      node.addEventListener('pointerup',()=>{entry.paused=false;motion(entry)});
      new ResizeObserver(()=>{viewer.resize();viewer.render()}).observe(node);
    }catch(error){node.dataset.error='true';console.error(`Gallery ${scene.id}:`,error)}
  }
  document.addEventListener('visibilitychange',()=>entries.forEach(motion));
  window.addEventListener('site-theme-change',()=>entries.forEach(e=>{e.viewer.setBackgroundColor(getComputedStyle(document.documentElement).getPropertyValue('--bg').trim(),0);e.viewer.render()}));
  reduced.addEventListener('change',()=>entries.forEach(motion));
  window.paperGallery=entries;
  scenes.forEach(setup);
})();
