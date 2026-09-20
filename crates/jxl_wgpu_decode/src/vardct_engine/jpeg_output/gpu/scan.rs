use super::super::{runtime::Plan, scan::PlannedScan};
use super::*;

type Step = (usize, [u32; 4], u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum ScanPhase {
    Count,
    Emit,
    Pack,
}
#[derive(Debug)]
pub(in super::super) struct Scan {
    tables: GpuBufferLease,
    tasks: GpuBufferLease,
    work: GpuBufferLease,
    raw: GpuBufferLease,
    pub output: GpuBufferLease,
    params: [u32; 12],
    pub phase: ScanPhase,
    pub byte_len: u32,
    pub operation: Operation,
}
fn scratch_words(mut count: u32) -> crate::Result<u32> {
    let mut words = 0u64;
    loop {
        let groups = count.div_ceil(64);
        words = add(words, u64::from(groups) * 2)?;
        if groups <= 1 {
            return word(words);
        }
        count = groups;
    }
}
fn prefix(
    count: u32,
    source: u32,
    destination: u32,
    scratch: u32,
    max: bool,
    steps: &mut Vec<Step>,
) {
    let groups = count.div_ceil(64);
    steps.push((
        if max { 12 } else { 7 },
        [source, destination, count, scratch],
        groups,
    ));
    if groups > 1 {
        prefix(
            groups,
            scratch,
            scratch + groups,
            scratch + groups * 2,
            max,
            steps,
        );
        steps.push((
            if max { 13 } else { 8 },
            [scratch + groups, destination, count, 0],
            groups,
        ));
    }
}
fn count_steps(blocks: u32, scratch: u32) -> Vec<Step> {
    let groups = blocks.div_ceil(64);
    let progressive = 4 + blocks * 5;
    let mut steps = vec![(9, [0; 4], groups), (10, [0; 4], groups)];
    prefix(
        blocks,
        progressive + blocks,
        progressive + blocks * 2,
        scratch,
        true,
        &mut steps,
    );
    steps.extend([(11, [0; 4], groups), (0, [0; 4], groups)]);
    prefix(blocks, 4, 4 + blocks * 3, scratch, false, &mut steps);
    steps.push((1, [0; 4], groups));
    prefix(
        blocks,
        4 + blocks * 4,
        4 + blocks,
        scratch,
        false,
        &mut steps,
    );
    steps.push((2, [0; 4], groups));
    steps
}
struct Bindings<'a> {
    coefficients: &'a GpuJpegCoefficients,
    padding: &'a GpuBufferLease,
    buffers: [&'a GpuBufferLease; 5],
    params: &'a [u32; 12],
}
fn encode(
    allocator: &mut Allocator<'_>,
    mut commands: wgpu::CommandEncoder,
    bindings: Bindings<'_>,
    steps: &[Step],
    mut held: Vec<GpuBufferLease>,
) -> crate::Result<Operation> {
    let resources = allocator.resources;
    let Bindings {
        coefficients,
        padding,
        buffers,
        params,
    } = bindings;
    let uniform = allocator.uniform(params)?;
    let [tables, tasks, work, raw, output] = buffers;
    for &(phase, words, groups) in steps {
        let scan_uniform = allocator.uniform(&words)?;
        resources.dispatch(
            &mut commands,
            &resources.pipelines.entropy[phase],
            &resources.pipelines.entropy_layout,
            &[
                &uniform,
                coefficients.buffer(),
                tables,
                tasks,
                work,
                raw,
                output,
                &scan_uniform,
                padding,
            ],
            groups,
        )?;
        held.push(scan_uniform);
    }
    held.extend([
        uniform,
        coefficients.buffer().clone(),
        padding.clone(),
        tables.clone(),
        tasks.clone(),
        work.clone(),
        raw.clone(),
        output.clone(),
    ]);
    let staging = allocator.staging()?;
    submit(resources, commands, work, staging, held, coefficients)
}
impl Scan {
    pub(in super::super) fn start(
        resources: &Resources,
        coefficients: &GpuJpegCoefficients,
        padding: &GpuBufferLease,
        plan: &Plan,
        scan_index: usize,
    ) -> crate::Result<Self> {
        let PlannedScan {
            parameters,
            tasks,
            tables,
        } = &plan.scans.scans[scan_index];
        let blocks = word(tasks.len() as u64)?;
        let persistent = word(4 + u64::from(blocks) * 9)?;
        let work_words = word(add(
            u64::from(persistent),
            u64::from(scratch_words(blocks)?),
        )?)?;
        let steps = count_steps(blocks, persistent);
        let table_bytes = storage_bytes(add(tables.len() as u64 * 4, plan.padding.len() as u64)?)?;
        let task_bytes = tasks.len() as u64 * 48;
        let total = add(
            add(table_bytes, task_bytes)?,
            u64::from(work_words) * 4 + 8 + 48 + 16 + steps.len() as u64 * 16,
        )?;
        let mut allocator = Allocator::new(resources, total)?;
        let tables = allocator.buffer(
            table_bytes,
            STORAGE,
            &[bytemuck::cast_slice(tables), &plan.padding],
        )?;
        let tasks = allocator.buffer(task_bytes, STORAGE, &[bytemuck::cast_slice(tasks)])?;
        let work = allocator.buffer(u64::from(work_words) * 4, STORAGE, &[])?;
        let raw = allocator.buffer(4, STORAGE, &[])?;
        let output = allocator.buffer(4, STORAGE, &[])?;
        let raw_limit = plan
            .limits
            .max_raw_scan_bytes
            .min(u64::from(u32::MAX / 32) * 4);
        let output_limit = plan
            .limits
            .max_output_bytes
            .min(u64::from(u32::MAX / 4) * 4);
        let params = [
            blocks,
            word(coefficients.layout().storage_bytes() / 4)?,
            word(raw_limit.div_ceil(4))?,
            word(output_limit.div_ceil(4))?,
            plan.padding_bits,
            u32::from(plan.has_padding)
                | (u32::from(scan_index + 1 == plan.scans.scans.len()) << 1),
            4 + blocks * 5,
            *parameters,
            resources.width(),
            0,
            0,
            0,
        ];
        let commands = resources
            .backend
            .device()
            .create_command_encoder(&Default::default());
        let operation = encode(
            &mut allocator,
            commands,
            Bindings {
                coefficients,
                padding,
                buffers: [&tables, &tasks, &work, &raw, &output],
                params: &params,
            },
            &steps,
            Vec::new(),
        )?;
        Ok(Self {
            tables,
            tasks,
            work,
            raw,
            output,
            params,
            phase: ScanPhase::Count,
            byte_len: 0,
            operation,
        })
    }
    /// Advance only after consuming a successful map result. No callback schedules further work.
    pub(in super::super) fn advance(
        mut self,
        resources: &Resources,
        coefficients: &GpuJpegCoefficients,
        padding: &GpuBufferLease,
        plan: &Plan,
        status: [u32; 4],
    ) -> crate::Result<Self> {
        if status[1] == 0 || status[1] & 7 != 0 || status[2] != status[1] / 8 {
            return Err(Error::GpuStatus {
                stage: "raw count",
                status,
            }
            .into());
        }
        check(
            "raw scan bytes",
            u64::from(status[2]),
            plan.limits.max_raw_scan_bytes,
        )?;
        let persistent = 4 + self.params[0] * 9;
        let mut commands = resources
            .backend
            .device()
            .create_command_encoder(&Default::default());
        let (new_bytes, steps) = match self.phase {
            ScanPhase::Count => {
                self.params[2] = status[2].div_ceil(4);
                let bytes = word(u64::from(self.params[2]) * 4)?;
                let scratch = word(u64::from(persistent) + u64::from(bytes) * 2)?;
                let work_words = word(add(u64::from(scratch), u64::from(scratch_words(bytes)?))?)?;
                let mut steps = vec![
                    (3, [0; 4], self.params[0].div_ceil(64)),
                    (4, [0; 4], bytes.div_ceil(64)),
                ];
                prefix(
                    bytes,
                    persistent,
                    persistent + bytes,
                    scratch,
                    false,
                    &mut steps,
                );
                steps.push((5, [0; 4], 1));
                (u64::from(work_words) * 4 + u64::from(bytes), steps)
            }
            ScanPhase::Emit => {
                if status[3] < status[2] || u64::from(status[3]) > u64::from(status[2]) * 2 {
                    return Err(Error::GpuStatus {
                        stage: "escaped count",
                        status,
                    }
                    .into());
                }
                check(
                    "escaped scan bytes",
                    u64::from(status[3]),
                    plan.limits.max_output_bytes,
                )?;
                self.byte_len = status[3];
                self.params[3] = status[3].div_ceil(4);
                (
                    u64::from(self.params[3]) * 4,
                    vec![(6, [0; 4], status[2].div_ceil(64)), (14, [0; 4], 1)],
                )
            }
            ScanPhase::Pack => return Err(Error::Invalid("completed scan advanced").into()),
        };
        let mut allocator = Allocator::new(
            resources,
            add(new_bytes, 48 + 16 + steps.len() as u64 * 16)?,
        )?;
        let mut held = Vec::new();
        if self.phase == ScanPhase::Count {
            let raw_bytes = u64::from(self.params[2]) * 4;
            let work = allocator.buffer(new_bytes - raw_bytes, STORAGE, &[])?;
            commands.copy_buffer_to_buffer(
                self.work.as_wgpu_buffer(),
                0,
                work.as_wgpu_buffer(),
                0,
                u64::from(persistent) * 4,
            );
            held.push(std::mem::replace(&mut self.work, work));
            self.raw = allocator.buffer(raw_bytes, STORAGE, &[])?;
            self.phase = ScanPhase::Emit;
        } else {
            self.output = allocator.buffer(new_bytes, STORAGE, &[])?;
            self.phase = ScanPhase::Pack;
        }
        self.operation = encode(
            &mut allocator,
            commands,
            Bindings {
                coefficients,
                padding,
                buffers: [
                    &self.tables,
                    &self.tasks,
                    &self.work,
                    &self.raw,
                    &self.output,
                ],
                params: &self.params,
            },
            &steps,
            held,
        )?;
        Ok(self)
    }
}
