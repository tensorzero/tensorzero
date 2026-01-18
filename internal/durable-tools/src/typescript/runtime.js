// Runtime bridge that exposes ToolContext methods to TypeScript tools.
// This file is bundled with the deno extension and provides the global `ctx` object.

const core = Deno.core;

globalThis.ctx = {
    // Identity
    taskId() {
        return core.ops.op_task_id();
    },

    episodeId() {
        return core.ops.op_episode_id();
    },

    // Tool operations
    async callTool(name, llmParams, sideInfo) {
        return await core.ops.op_call_tool(name, llmParams, sideInfo);
    },

    async spawnTool(name, llmParams, sideInfo) {
        return await core.ops.op_spawn_tool(name, llmParams, sideInfo);
    },

    async joinTool(handleId) {
        return await core.ops.op_join_tool(handleId);
    },

    // Inference
    async inference(params) {
        return await core.ops.op_inference(params);
    },

    // Durable primitives
    async rand() {
        return await core.ops.op_rand();
    },

    async now() {
        return await core.ops.op_now();
    },

    async uuid7() {
        return await core.ops.op_uuid7();
    },

    async sleepFor(name, durationMs) {
        return await core.ops.op_sleep_for(name, BigInt(durationMs));
    },

    // Events
    async awaitEvent(eventName, timeoutMs) {
        return await core.ops.op_await_event(eventName, timeoutMs ?? null);
    },

    async emitEvent(eventName, payload) {
        return await core.ops.op_emit_event(eventName, payload);
    },
};

// Simple console implementation that writes to Deno.core.print
globalThis.console = {
    log(...args) {
        core.print(args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ') + '\n', false);
    },
    error(...args) {
        core.print(args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ') + '\n', true);
    },
    warn(...args) {
        core.print('[WARN] ' + args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ') + '\n', true);
    },
    info(...args) {
        core.print('[INFO] ' + args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ') + '\n', false);
    },
    debug(...args) {
        core.print('[DEBUG] ' + args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ') + '\n', false);
    },
};
