import { useEffect } from "react";
import type { Decorator } from "@storybook/react-vite";
import type { Preview } from "@storybook/react-vite";
import { withRouter } from "storybook-addon-remix-react-router";
import { TooltipProvider } from "../app/components/ui/tooltip";
import { MockEntitySheetProvider } from "../app/context/entity-sheet";
import { Theme, ThemeProvider, useTheme } from "../app/context/theme";

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

/** Syncs the Storybook toolbar theme global → ThemeProvider context. */
function ThemeSync({ theme }: { theme: string }) {
  const { setTheme } = useTheme();
  useEffect(() => {
    setTheme(theme === "dark" ? Theme.Dark : Theme.Light);
  }, [theme, setTheme]);
  return null;
}

const themeDecorator: Decorator = (Story, context) => {
  const theme = context.globals["theme"] ?? "light";

  return (
    <ThemeProvider>
      <ThemeSync theme={theme} />
      <Story />
    </ThemeProvider>
  );
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
