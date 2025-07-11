import pluginJs from "@eslint/js";
import tseslint from "typescript-eslint";
import pluginReact from "eslint-plugin-react";
import pluginReactHooks from "eslint-plugin-react-hooks";

/** @type {import('@typescript-eslint/utils').TSESLint.FlatConfig.ConfigFile} */
export default [
  {
    ignores: [
      "**/minijinja/pkg/",
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
      ],
    },
  },
  pluginReactHooks.configs["recommended-latest"],
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      "@typescript-eslint/switch-exhaustiveness-check": "warn", // TODO Change to error after incremental refactoring
    },
  },

  // TODO… Add stricter linting
  // ...tseslint.configs.recommendedTypeChecked,
  // ...tseslint.configs.stylisticTypeChecked,
];
