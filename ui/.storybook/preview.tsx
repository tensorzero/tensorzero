import type { Decorator } from "@storybook/react-vite";
import type { Preview } from "@storybook/react-vite";
import { useMemo } from "react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { AppProviders } from "../app/providers/app-providers";

import "../app/tailwind.css";

const resetBrowserStorageDecorator: Decorator = (Story) => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  return <Story />;
};

const appProviderDecorator: Decorator = (Story) => {
  const router = useMemo(
    () =>
      createMemoryRouter(
        [
          {
            path: "*",
            element: (
              <AppProviders>
                <Story />
              </AppProviders>
            ),
          },
        ],
        { initialEntries: ["/"] },
      ),
    [Story],
  );
  return <RouterProvider router={router} />;
};

const preview: Preview = {
  decorators: [resetBrowserStorageDecorator, appProviderDecorator],
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
