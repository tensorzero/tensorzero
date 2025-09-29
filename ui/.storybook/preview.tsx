import type { Decorator } from "@storybook/react";
import type { Preview } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";

import "../app/tailwind.css";

const resetBrowserStorageDecorator: Decorator = (Story) => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  return <Story />;
};

const preview: Preview = {
  decorators: [withRouter, resetBrowserStorageDecorator],
  parameters: {
    layout: "centered",
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
};

export default preview;
