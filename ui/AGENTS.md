- The UI uses React Router 7, Tailwind, Node, and pnpm.
- UI routes (pages & API) are defined in `./app/routes.ts`.
- Prefer `logger` from `~/utils/logger` over `console.error`, `console.warn`, `console.log`, `console.debug`.
- Prefer using React Router's `useFetcher` over direct calls to `fetch`. [Reference](https://reactrouter.com/api/hooks/useFetcher)
- After modifying UI code, run from the `ui/` directory: `pnpm run format`, `pnpm run lint`, `pnpm run typecheck`. All commands must pass.

## URL Conventions

- **Frontend routes** (user-facing URLs): Use **hyphens** (e.g., `/api-keys`, `/workflow-evaluations`, `/supervised-fine-tuning`)
- **Backend API routes** (gateway API): Use **underscores** (e.g., `/internal/workflow_evaluations`, `/api/curated_inferences`)
- This convention aligns with web standards (hyphens for URLs) while maintaining consistency with the gateway API (underscores, like Stripe/OpenAI).
