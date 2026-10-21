// Concise presentation of the recorded protocols; source records remain unchanged.
export function protocolFor(g) {
  const native=g.protocol==='native_pipeline', ppi=g.dataset.startsWith('ppi_');
  if(g.model.startsWith('rfd4')) {
    const search=g.model.includes('search');
    const level=g.model.includes('low')?'Low':g.model.includes('medium')?'Medium':'High';
    return {
      procedure:search?`RFD4-Proteina’s native search generates sequence–structure candidates and uses ${ppi?'AF2-Multimer':'OF3'} for internal evaluation and candidate selection.`:`RFD4-Proteina jointly generates protein sequence and structure using the recorded ${level.toLowerCase()}-thinking recycling schedule and checkpoint.`,
      adaptation:(search?'Evaluate one internally selected sequence–structure pair per start. Generation cost includes all search proposals and internal evaluations.':'Evaluate one codesigned sequence–structure pair per start, without additional sequence design or selection.')+(g.dataset.startsWith('mcsa_')?' Unmatched required catalytic motifs fail.':''),
    };
  }
  switch(g.model) {
    case 'bindcraft': return native?{
      procedure:'BindCraft optimization with continuation gates, relaxation, soluble ProteinMPNN redesign, and AF2 validation. Propose up to 20 sequences, stopping after two accepted candidates.',
      adaptation:'For each of 200 starts, select the available candidate with highest average interface pTM, regardless of filter acceptance, and its highest-pLDDT validation structure. Use the saved trajectory when no ranked candidate exists. Do not replace rejected starts.',
    }:{
      procedure:'BindCraft optimization with the 75/45/5/15 schedule, followed by one soluble ProteinMPNN sequence at temperature 0.1.',
      adaptation:'Disable all three continuation gates. Stop after sequence design, without validation, relaxation, or selection. Keep interface residues fixed during redesign; omit sidechains incompatible with the redesigned sequence.',
    };
    case 'bindcraft2': return native?{
      procedure:'BindCraft2 optimization with continuation gates, ten MPNN candidates, and three-recycle validation. Stop after three accepted candidates; relaxation is disabled.',
      adaptation:'For each of 200 starts, select the available candidate with highest i_pDAE, regardless of filter acceptance. Use the saved trajectory when no ranked candidate exists. Reset candidate history between starts; do not replace rejected starts.',
    }:{
      procedure:'BindCraft2 optimization with the 50/25/45/5/15 schedule, followed by one soluble-MPNN redesign at temperature 0.1.',
      adaptation:'Disable continuation gates and stop after sequence design, without independent validation, relaxation, or selection. Apply BindCraft2’s interface and redesign constraints; omit sidechains incompatible with the redesigned sequence.',
    };
    case 'genie3': return native?{
      procedure:'Genie3’s native 100-step beam search uses width 4 and scoring every 25 steps; candidates receive CA-MPNN sequences and AF2-Multimer evaluation.',
      adaptation:'Condition on target CA coordinates and original hotspot sidechains, without extended contacts. Each start includes 64 internal evaluations; select lowest mean interface PAE and retain the generated CA backbone. PPI RMSD uses binder CA atoms.',
      note:'Generation timing uses one padding-optimized warm run per target, applied to 200 starts, including internal MPNN and AF2. Common AF3 costs are recorded separately.',
    }:{
      procedure:'Genie3’s native generative diffusion runs for 100 steps, followed by one CA-MPNN sequence at temperature 0.1.',
      adaptation:'Beam search and ColabFold are omitted. Conditioning uses target CA coordinates and original hotspot sidechains, without extended contacts. Generated structures remain CA-only; PPI RMSD is measured on binder CA atoms.',
    };
    case 'halludesign': return {
      procedure:'Ten Protenix cycles with 150 refinement steps, two MPNN proposals, and five ranked predictions per cycle. The final cycle skips coordinate refinement.',
      adaptation:ppi?'Condition on target sequences and original hotspots, without target-coordinate templates. Enable interaction optimization and selection from cycle 0. Select the final cycle’s packed structure and MPNN sequence with highest evaluation pLDDT. Do not replace rejected starts.':'Condition on all target DNA sequences without coordinate templates, using 120–150-residue binders. Select the final cycle’s packed structure and MPNN sequence with highest evaluation pLDDT. Do not replace rejected starts.',
    };
    case 'pipelines': return g.dataset==='dna_fixed'?{
      procedure:'Follow the DNA-binder design procedure of Sehgal et al., using AF3-predicted target duplexes, a six-base-pair ORI window, donor/acceptor conditioning, and 2.5 Å ORI jitter.',
      adaptation:'Generate 200 attempts per target at protein lengths of 120–150 residues, followed by one LigandMPNN sequence per backbone. Evaluate the resulting designs with the common AF3 criteria.',
      reference_label:'Sehgal et al. · Generative design of sequence specific DNA binding proteins',
      reference_url:'https://doi.org/10.64898/2026.04.27.720408',
    }:{
      procedure:`RFdiffusion3 generation with 200 steps and classifier-free guidance disabled, followed by one LigandMPNN sequence at temperature 0.1.${g.dataset.startsWith('dna_')?' Use the DNA-finetuned checkpoint.':''}`,
      adaptation:ppi?'Omit terminal OXT constraints. Generate one sequence per backbone, without downstream folding or screening.':g.dataset.startsWith('dna_')?`DNA coordinates are ${g.dataset==='dna_fixed'?'fixed':'unfixed'}. Generate one sequence per backbone, without downstream folding or screening.`:g.dataset.startsWith('mcsa_')?'Omit downstream folding and screening. Retain reference-conformer preprocessing; unmatched required motifs fail. M0110 retains native bond and charge losses.':'Use source-derived ligand inputs; retain native reference-conformer preprocessing. Omit downstream folding and screening.',
    };
    case 'boltzgen': return {
      procedure:'BoltzGen design followed by inverse folding, with 500 and 200 steps respectively and three recycles each. Half the starts use diverse settings; half use adherence settings.',
      adaptation:ppi?'Project atom hotspots to residues on all-atom targets. Generate one sequence per start; omit terminal folding, filtering, and ranking.':g.dataset.startsWith('dna_')?`DNA coordinates are ${g.dataset==='dna_fixed'?'fixed':'unfixed'}. Generate one sequence per start; omit terminal folding, filtering, and ranking.`:'Use source-derived ligands, including native heme bond-type projections. Generate one sequence per start; omit terminal folding, filtering, and ranking.',
    };
    case 'proteina_complexa': return {
      procedure:`Generate one joint sequence/structure draw per original start with 400 steps using ${ppi?'the PPI V1 checkpoint':g.dataset.startsWith('mcsa_')?'the AME V2/LoRA32 checkpoint':'the dedicated V1/LoRA32 ligand checkpoint'}.`,
      adaptation:ppi?'Convert all-atom targets to atom37/template features and project atom hotspots to residue-level constraints. Use the generated sequence directly, without an additional sequence designer or output selection.':g.dataset.startsWith('mcsa_')?'Convert ligand and motif conditioning to template features. Unmatched required motifs count as failures without external rematching or coordinate repair. Use the generated sequence without additional design or selection.':'Use the dedicated fixed-ligand route with ligand/template features derived from annotated targets. Use the jointly generated sequence directly, without an additional sequence designer or output selection.',
    };
    case 'disco': return {
      procedure:'Jointly generate sequence and structure with DISCO’s diverse/max configuration, using 200 diffusion steps and four recycles.',
      adaptation:'Derive SDF/JSON from annotated CIFs in place of repository benchmark molecules. Hydrogen and bond-order losses follow the author preprocessing. Use the generated sequence without additional design or output selection.',
    };
    case 'protein_hunter': return {
      procedure:'Seven refinement cycles from an all-X sequence, with 200 prediction steps, three recycles, and MPNN temperature 0.1.',
      adaptation:ppi?'Use SolubleMPNN and original residue hotspots with CA, CB, and frame guidance; sidechains are not fixed. Select the highest-ipTM cycle among candidates with at most 20% alanine. Rejected starts count as failures and are not replaced.':'Use LigandMPNN with ligand or DNA inputs derived from shared CIFs. Select the highest-ipTM cycle among candidates with at most 20% alanine. Rejections and chemistry or structure-writing failures count as failed starts, without replacements.',
    };
    case 'esmfold2': return {
      procedure:'Binder-design inversion with 150 optimization steps, two models, and four intrinsic critics.',
      adaptation:'Use predesignated critic 0. Condition on target sequences and residue hotspots, without target-coordinate conditioning. Extend to all target chains; whole-complex ipTM includes target–target contributions. No additional MPNN, terminal refolding, or AF3 selection.',
    };
    case 'rfdiffusion': return {
      procedure:'Run 50-step Complex_base diffusion with CA/frame noise scales set to zero, followed by one ProteinMPNN sequence at temperature 0.1.',
      adaptation:'Map target chains and contigs explicitly to preserve target copies, gaps, and original hotspots. Omit Rosetta, refolding, and screening; generated outputs contain backbone coordinates with implementation-specific atom omissions.',
    };
    case 'rfdiffusion2': return {
      procedure:'Generate scaffolds with the AME RFD_140 checkpoint and 100 flow steps, followed by one LigandMPNN sequence at temperature 0.1 with context and packing.',
      adaptation:'Replace the example’s sequence panel with one sequence; omit refolding and selection. Guidepost matching follows the author implementation. M0110 input omits two ligand bonds while keeping atoms and coordinates, changing its physical topology.',
    };
    case 'odesign': return {
      procedure:'Native ODesign scaffold generation followed by one LigandMPNN sequence at temperature 0.1, fixing source-conditioned sequence positions.',
      adaptation:g.dataset.startsWith('mcsa_')?'Enzyme inputs include additional motif context beyond the specified tip atoms.':'Fix ligand or DNA target coordinates, as labeled. Generate one redesigned sequence per scaffold.',
      note:'Generation timing comes from the original scaffold-plus-MPNN run; the later corrected MPNN pass is not added again.',
    };
    case 'alphaprotein_novo': return {
      procedure:'AlphaProtein-novo’s native 1,000-step generation jointly produces the protein sequence and structure, retaining the codesigned sequence.',
      adaptation:'Evaluate the codesigned sequence without additional sequence design or selection. This arm shares scaffolds with the LigandMPNN arm.',
      note:g.dataset.startsWith('mcsa_')?'Native APN1 composite-ligand handling changes the interface interpretation for M0110.':'',
    };
    case 'alphaprotein_novo_ligandmpnn': return {
      procedure:'AlphaProtein-novo’s 1,000-step scaffold generation followed by one LigandMPNN sequence at temperature 0.1 (checkpoint 030; atom context on, sidechain context and packing off).',
      adaptation:'Replace the codesigned sequence with one LigandMPNN sequence, fixing conditioned residues. Scaffolds are shared with the codesign arm, not independent samples.',
      note:'CPU MPNN contributes no GPU time; shared scaffold generation remains charged.'+(g.dataset.startsWith('mcsa_')?' Native APN1 composite-ligand handling changes the interface interpretation for M0110.':''),
    };
    default: throw Error(`Missing concise protocol: ${g.model}/${g.protocol}`);
  }
}
