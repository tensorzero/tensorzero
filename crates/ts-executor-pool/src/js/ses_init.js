// SES lockdown + compartment factory.
//
// Evaluated on creation inside the V8 snapshot. Freezes all ECMAScript
// standard intrinsics (Object.prototype, Array.prototype, etc.) so that
// user code cannot mutate them. Host objects like Deno.core.ops are NOT
// frozen, so deno_core extension ops still work at restore time.

lockdown({
  // Hide stack traces
  errorTaming: "safe",
  // Allow the code to execute `console.log()` (which we use for output)
  consoleTaming: "unsafe",
  // We need to override stuff via assignment
  overrideTaming: "moderate",
  // Disable `eval()` globally
  evalTaming: "noEval",
  // Remove SES-specific stack frames
  stackFiltering: "concise",
});

// Factory to create a Compartment with hardened endowments.
// Called from Rust during create_rlm_runtime.
globalThis.__t0_create_rlm_compartment = function (endowments) {
  harden(endowments);
  return new Compartment(endowments);
};
harden(globalThis.__t0_create_rlm_compartment);
