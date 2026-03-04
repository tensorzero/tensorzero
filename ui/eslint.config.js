import tseslint from "typescript-eslint";
import oxlint from "eslint-plugin-oxlint";

// Slim ESLint config: only rules that oxlint cannot handle yet.
// All standard rules (recommended, react, react-hooks, TS, etc.) are in .oxlintrc.json.
// When oxlint adds no-restricted-syntax and type-aware linting stabilizes,
// this file can be deleted entirely.
export default [
  {
    ignores: [
      "**/node_modules/**",
      "**/build/**",
      "**/.react-router/**",
      "**/.venv/**",
      "**/playwright-report/**",
      "**/test-results/**",
      "eslint.config.js",
      "**/.storybook/**",
    ],
  },
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      parserOptions: {
        project: "./tsconfig.json",
        tsconfigRootDir: import.meta.dirname,
      },
    },
    linterOptions: {
      // Rules like react-hooks/exhaustive-deps and no-console are now enforced by oxlint,
      // but code still has eslint-disable comments for them. Don't report these as errors.
      reportUnusedDisableDirectives: "off",
    },
    rules: {
      // Type-aware: oxlint's tsgolint is alpha and incompatible with our tsconfig (baseUrl)
      "@typescript-eslint/switch-exhaustiveness-check": "error",

      // AST selector rules: oxlint does not implement no-restricted-syntax
      "no-restricted-syntax": [
        "error",
        {
          selector:
            ":matches(Literal, TemplateElement)[value.raw=/tensorzero_ui_fixtures/]",
          message: "The string 'tensorzero_ui_fixtures' is not allowed.",
        },
        {
          selector:
            "MemberExpression[object.name='config'][property.name='functions']",
          message:
            "Direct access to config.functions is not allowed. Use the useFunctionConfig(functionName) hook instead.",
        },
        {
          selector:
            "MemberExpression[object.property.name='functions'][property.type='Literal']",
          message:
            "Direct access to config.functions[functionName] is not allowed. Use the useFunctionConfig(functionName) hook instead.",
        },
        {
          selector:
            "MemberExpression[object.property.name='functions'][property.type='Identifier']",
          message:
            "Direct access to config.functions[functionName] is not allowed. Use the useFunctionConfig(functionName) hook instead.",
        },
        {
          selector:
            "ImportDeclaration[source.value='tensorzero-node'][importKind='type']",
          message:
            "Do not import types directly from 'tensorzero-node'. Use 'import type { ... } from \"~/types/tensorzero\"' instead to avoid bundling the native client in browser code.",
        },
        {
          selector:
            "ImportDeclaration[source.value='tensorzero-node'] ImportSpecifier[importKind='type']",
          message:
            "Do not import types directly from 'tensorzero-node'. Use 'import type { ... } from \"~/types/tensorzero\"' instead to avoid bundling the native client in browser code.",
        },
        {
          selector:
            "CallExpression[callee.object.name='crypto'][callee.property.name='randomUUID']",
          message:
            "Do not use crypto.randomUUID(). Use `import { v7 as uuid } from 'uuid'` and call `uuid()` instead for UUIDv7.",
        },
      ],
    },
  },
  // Disable all ESLint rules that oxlint already covers
  ...oxlint.configs["flat/recommended"],
  ...oxlint.configs["flat/typescript"],
];
