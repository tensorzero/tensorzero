import pluginJs from "@eslint/js";
import tseslint from "typescript-eslint";
import pluginReact from "eslint-plugin-react";

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
          selector: "Literal[value='tensorzero_ui_fixtures']",
          message: "The string 'tensorzero_ui_fixtures' is not allowed.",
        },
      ],
    },
  },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
];
