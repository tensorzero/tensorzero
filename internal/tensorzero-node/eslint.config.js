import pluginJs from "@eslint/js";
import tseslint from "typescript-eslint";

export default [
  {
    ignores: [
      "**/node_modules/**",
      "**/dist/**",
      "**/npm/**",
      "**/*.node",
      "eslint.config.js",
    ],
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
        Buffer: true,
        global: true,
      },
    },
  },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
];
