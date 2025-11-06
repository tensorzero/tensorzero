import pluginJs from "@eslint/js";
import tseslint from "typescript-eslint";
import pluginReact from "eslint-plugin-react";
import pluginReactHooks from "eslint-plugin-react-hooks";

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
  {
    files: ["**/*.{js,mjs,cjs,ts,jsx,tsx}"],
    plugins: {
      react: pluginReact,
    },
    languageOptions: {
      globals: {
        document: true,
        window: true,
        process: true,
        require: true,
        module: true,
        __dirname: true,
        console: true,
      },
      parserOptions: {
        project: "./tsconfig.json",
        tsconfigRootDir: import.meta.dirname,
        ecmaFeatures: {
          jsx: true,
        },
      },
    },
    settings: {
      react: {
        version: "detect",
      },
    },
    rules: {
      "react/jsx-uses-react": "error",
      "react/jsx-uses-vars": "error",
      "react/react-in-jsx-scope": "off",
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
      ],
    },
  },
  pluginReactHooks.configs["recommended-latest"],
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      "@typescript-eslint/switch-exhaustiveness-check": "error",
    },
  },
];
