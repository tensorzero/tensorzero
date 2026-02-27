import { useEffect } from "react";
import type { Decorator } from "@storybook/react-vite";
import type { Preview } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import { TooltipProvider } from "../app/components/ui/tooltip";
import { MockEntitySheetProvider } from "../app/context/entity-sheet";

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
  <MockEntitySheetProvider>
    <Story />
  </MockEntitySheetProvider>
);

const themeDecorator: Decorator = (Story, context) => {
  const theme = context.globals["theme"] ?? "light";
  const isDark = theme === "dark";

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", isDark);
    root.style.colorScheme = isDark ? "dark" : "light";
    document.body.style.backgroundColor = isDark
      ? "hsl(0 0% 9%)"
      : "hsl(0 0% 100%)";
  }, [isDark]);

  return <Story />;
};

const preview: Preview = {
  globalTypes: {
    theme: {
      description: "Toggle light/dark mode",
      toolbar: {
        title: "Theme",
        icon: "mirror",
        items: [
          { value: "light", title: "Light", icon: "sun" },
          { value: "dark", title: "Dark", icon: "moon" },
        ],
        dynamicTitle: true,
      },
    },
  },
  initialGlobals: {
    theme: "light",
  },
  decorators: [
    withRouter,
    resetBrowserStorageDecorator,
    themeDecorator,
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
