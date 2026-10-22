nextflow.enable.dsl = 2

process GENERATE {
    tag "shard-${shard_id}"
    label 'generation'
    cpus params.generation_cpus.toInteger()
    memory '32 GB'
    maxForks 2
    accelerator 1, type: params.generation_accelerator
    container params.container_image
    publishDir "${params.outdir}/shards", mode: 'copy'

    input:
    val shard_id
    path input_cif, stageAs: 'input/*'
    path model_bundle, stageAs: 'model'
    path settings, stageAs: 'settings.json'
    path runner, stageAs: 'workflow.py'

    output:
    path "shard-${shard_id}", emit: results

    script:
    def request = groovy.json.JsonOutput.toJson([
        input_cif: input_cif.toString(),
        model_bundle: model_bundle.toString(),
        settings: settings.toString(),
        shard_id: shard_id,
        num_shards: params.num_shards.toInteger(),
        container_image: params.container_image
    ]).bytes.encodeBase64().toString()
    """
    python3 workflow.py generate '${request}'
    """

    stub:
    def request = groovy.json.JsonOutput.toJson([
        input_cif: input_cif.toString(),
        model_bundle: model_bundle.toString(),
        settings: settings.toString(),
        shard_id: shard_id,
        num_shards: params.num_shards.toInteger(),
        container_image: params.container_image
    ]).bytes.encodeBase64().toString()
    """
    python3 workflow.py plan '${request}'
    """
}

process SUMMARIZE {
    container params.container_image
    cpus 1
    memory '2 GB'
    publishDir params.outdir, mode: 'copy'

    input:
    path shards
    path runner, stageAs: 'workflow.py'

    output:
    path 'summary', emit: report

    script:
    """
    python3 workflow.py summarize . ${params.num_shards.toInteger()}
    """
}

workflow {
    if (!params.input_cif || !params.model_bundle || !params.settings || !params.container_image) {
        error 'Set --input_cif, --model_bundle, --settings and --container_image; see examples/healthomics/README.md.'
    }
    if (!params.num_shards.toString().isInteger() || params.num_shards.toInteger() < 1) {
        error '--num_shards must be a positive integer.'
    }
    if (!params.generation_cpus.toString().isInteger() || params.generation_cpus.toInteger() < 1) {
        error '--generation_cpus must be a positive integer.'
    }
    if (!(params.generation_accelerator in ['nvidia-l40s', 'nvidia-l4'])) {
        error '--generation_accelerator must be nvidia-l40s or nvidia-l4.'
    }
    def input_cif = file(params.input_cif, checkIfExists: true)
    def model_bundle = file(params.model_bundle, checkIfExists: true)
    def settings = file(params.settings, checkIfExists: true)
    def runner = file("${projectDir}/bin/workflow.py", checkIfExists: true)
    def shard_ids = channel.fromList((0..<params.num_shards.toInteger()).toList())

    GENERATE(shard_ids, input_cif, model_bundle, settings, runner)
    SUMMARIZE(GENERATE.out.results.collect(), runner)
}
