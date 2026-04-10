// ts_checker.js — TypeScript checker helper, evaluated into a V8 snapshot at
// build time (see build.rs) after typescript.js loads the `ts` global.
//
// Exports: __t0_check_and_transpile(code, ambientDeclarations) -> JSON string
//   - On success: { ok: true, js: "<transpiled JavaScript>" }
//   - On failure: { ok: false, diagnostics: "<formatted error messages>" }
//
// Uses a minimal virtual compiler host with two files:
//   - "ambient.d.ts": the ambient type declarations (rlm_ambient.d.ts + tool types)
//   - "input.ts": the user code to typecheck
//
// Key compiler options:
//   - noLib: true — skips the full TS stdlib for ~8x faster typechecking
//   - strict: true — full type safety
//   - target: ESNext, module: None — modern JS output, no module system
//   - Diagnostics are filtered to input.ts only (ambient.d.ts errors are ignored)

globalThis.__t0_check_and_transpile = function (code, ambientDeclarations) {
  // Make the checked source a module so top-level await is allowed, while
  // still transpiling the original code so runtime JS stays script-shaped.
  const typecheckInput = code + "\nexport {};";
  // Virtual file system
  const files = {
    "ambient.d.ts": ambientDeclarations,
    "input.ts": typecheckInput,
  };

  // Minimal compiler host
  const compilerOptions = {
    noEmit: true,
    noLib: true,
    strict: true,
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Node,
    skipLibCheck: true,
  };

  const host = {
    getSourceFile(fileName, languageVersion) {
      if (files[fileName] !== undefined) {
        return ts.createSourceFile(
          fileName,
          files[fileName],
          languageVersion
        );
      }
      return undefined;
    },
    getDefaultLibFileName() {
      return "";
    },
    writeFile() {},
    getCurrentDirectory() {
      return "/";
    },
    getCanonicalFileName(fileName) {
      return fileName;
    },
    useCaseSensitiveFileNames() {
      return true;
    },
    getNewLine() {
      return "\n";
    },
    fileExists(fileName) {
      return files[fileName] !== undefined;
    },
    readFile(fileName) {
      return files[fileName];
    },
    getDirectories() {
      return [];
    },
  };

  // Create program and get diagnostics for input.ts only
  const program = ts.createProgram(
    ["ambient.d.ts", "input.ts"],
    compilerOptions,
    host
  );

  const allDiagnostics = ts.getPreEmitDiagnostics(program);
  // Filter to diagnostics from input.ts only (ignore ambient.d.ts noise)
  const inputDiagnostics = allDiagnostics.filter(
    (d) => !d.file || d.file.fileName === "input.ts"
  );

  if (inputDiagnostics.length > 0) {
    const formatted = ts.formatDiagnosticsWithColorAndContext(
      inputDiagnostics,
      {
        getCanonicalFileName: (f) => f,
        getCurrentDirectory: () => "/",
        getNewLine: () => "\n",
      }
    );
    return JSON.stringify({ ok: false, diagnostics: formatted });
  }

  // Typecheck passed — transpile to strip types
  const transpiled = ts.transpileModule(code, {
    compilerOptions: {
      target: ts.ScriptTarget.ESNext,
      module: ts.ModuleKind.None,
    },
  });

  return JSON.stringify({ ok: true, js: transpiled.outputText });
};
