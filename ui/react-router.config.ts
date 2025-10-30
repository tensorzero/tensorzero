import type { Config } from "@react-router/dev/config";

export default {
  // Config options...
  // Server-side render by default, to enable SPA mode set this to `false`
  ssr: true,

  // This should fix a bug in React Router that causes the dev server to crash
  // on the first page load after clearing node_modules. This will likey be the
  // default behavior in a future version of React Router, so no need to remove
  // this unless it causes bigger issues.
  // https://github.com/remix-run/react-router/issues/12786#issuecomment-2634033513
  future: {
    unstable_optimizeDeps: true,
    v8_middleware: true,
  },
} satisfies Config;
