/*__JXL_VARDCT_BLOCK_CONTEXT__*/

struct ProbeParams {
    tables: HfBlockContextTables,
    order_channel: u32,
    order_id: u32,
    qf: u32,
    lf_x: u32,
    lf_y: u32,
    lf_b: u32,
    _reserved0: u32,
    _reserved1: u32,
};

@group(0) @binding(0) var<storage, read> modular_metadata: array<u32>;
@group(0) @binding(1) var<storage, read> params_input: array<ProbeParams>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn probe_block_context(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let params = params_input[index];
    result[index] = hf_block_context(
        params.tables,
        params.order_channel,
        params.order_id,
        params.qf,
        vec3<i32>(
            bitcast<i32>(params.lf_x),
            bitcast<i32>(params.lf_y),
            bitcast<i32>(params.lf_b),
        ),
    );
}
