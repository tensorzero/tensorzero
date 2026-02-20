import type { Decorator } from "@storybook/react-vite";
import type { Preview } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import { TooltipProvider } from "../app/components/ui/tooltip";
import { EntitySheetProvider } from "../app/context/entity-sheet";

import "../app/tailwind.css";

const resetBrowserStorageDecorator: Decorator = (Story) => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  return <Story />;
};

const tooltipProviderDecorator: Decorator = (Story) => (
  <TooltipProvider>
    <Story />
  </TooltipProvider>
);

const entitySheetProviderDecorator: Decorator = (Story) => (
  <EntitySheetProvider>
    <Story />
  </EntitySheetProvider>
);

const preview: Preview = {
  decorators: [
    withRouter,
    resetBrowserStorageDecorator,
    tooltipProviderDecorator,
    entitySheetProviderDecorator,
  ],
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
