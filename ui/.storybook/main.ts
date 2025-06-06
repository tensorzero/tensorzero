import type { StorybookConfig } from "@storybook/react-vite";

const config: StorybookConfig = {
  stories: [
    "../**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../**/*.storybook.mdx",
    // Examples:
    "../.storybook/examples/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../.storybook/examples/**/*.storybook.mdx",
  ],
  addons: [
    "@storybook/addon-onboarding",
    "@chromatic-com/storybook",
    "@storybook/addon-vitest",
    "@storybook/addon-docs",
  ],
  framework: {
    name: "@storybook/react-vite",
    options: {},
  },
};
export default config;
