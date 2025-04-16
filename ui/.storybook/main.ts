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
    "@storybook/addon-essentials",
    "@storybook/addon-onboarding",
    "@chromatic-com/storybook",
    "@storybook/experimental-addon-test",
  ],
  framework: {
    name: "@storybook/react-vite",
    options: {},
  },
};
export default config;
