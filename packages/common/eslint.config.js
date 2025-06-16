// @ts-check
import pluginJs from "@eslint/js";
import tseslint from "typescript-eslint";
import globals from "globals";

export default [
  {
    files: ["**/*.{js,mjs,cjs,ts}"],
    languageOptions: {
      globals: { ...globals.browser },
    },
  },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
];
