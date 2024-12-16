/**
 * By default, Remix will handle hydrating your app on the client for you.
 * You are free to delete this file if you'd like to, but if you ever want it revealed again, you can run `npx remix reveal` ✨
 * For more information, see https://remix.run/file-conventions/entry.client
 */

import { HydratedRouter } from "react-router/dom";
import { startTransition, StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";

startTransition(() => {
  const root = document.getElementById("root");
  if (!root) {
    throw new Error("Root element not found");
  }
  hydrateRoot(
    root,
    <StrictMode>
      <HydratedRouter />
    </StrictMode>,
  );
});
