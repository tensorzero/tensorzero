import pluginJs from "@eslint/js";
import tseslint from "typescript-eslint";

export default [
  {
    ignores: ["**/node_modules/**", "**/dist/**"],
  },
  {
    files: ["**/*.{js,mjs,cjs,ts}"],
    languageOptions: {
      globals: {
        process: true,
        require: true,
        module: true,
        __dirname: true,
        console: true,
      },
    },
    rules: {
      "no-console": ["warn", { allow: ["warn", "error"] }],
      "no-unused-vars": "warn",
    },
  },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
];
