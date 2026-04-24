// RLM (Recursive Language Model) JavaScript bridge functions.
//
// Injected into every JsRuntime via the rlm_ext extension at restore time.
// These live on the outer globalThis (accessible to Deno.core.ops) but are
// NOT visible from inside the SES Compartment -- user code only sees the
// curated endowments passed to __t0_create_rlm_compartment.

// Async: send a prompt to a child RLM loop (or single-shot LLM at max depth).
globalThis.__t0_llm_query = async function(prompt) {
  return await Deno.core.ops.op_llm_query(prompt);
};

// Async: send multiple prompts concurrently and collect results in order.
globalThis.__t0_llm_query_batched = async function(prompts) {
  return await Deno.core.ops.op_llm_query_batched(prompts);
};

// Sync: mark a value as the final answer and end the RLM loop.
globalThis.__t0_FINAL = function(value) {
  var str = typeof value === 'string' ? value : JSON.stringify(value);
  Deno.core.ops.op_set_final(str);
};

globalThis.__t0_tool_call_dispatch = function(name, params) {
  return Deno.core.ops.op_tool_call_dispatch(name, params);
};

globalThis.__t0_tool_call_join = function(handle) {
  return Deno.core.ops.op_tool_call_join(handle);
};

// Override console.log to capture output via our op.
globalThis.__t0_console = {
  log: function() {
    var args = [];
    for (var i = 0; i < arguments.length; i++) {
      args.push(String(arguments[i]));
    }
    Deno.core.ops.op_console_log(args.join(' '));
  },
};
