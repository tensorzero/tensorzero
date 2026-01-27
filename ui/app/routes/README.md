# UI Routes

Routes are defined in `routes.ts`.

## URL Conventions

- Frontend routes (user-facing URLs): Use hyphens (e.g., `/api-keys`, `/workflow-evaluations`, `/supervised-fine-tuning`)
- Backend API routes: Use underscores
  - RR7 API routes (React Router resource routes): `/api/workflow_evaluations/search_runs`
  - Gateway API routes: `/internal/workflow_evaluations`, `/api/curated_inferences`

This convention aligns frontend URLs with web standards (hyphens) while backend routes use underscores for consistency with the TensorZero gateway API.
