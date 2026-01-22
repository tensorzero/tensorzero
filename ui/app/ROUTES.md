# UI Routes

Routes are defined in `routes.ts`.

## URL Conventions

- Frontend routes (user-facing URLs): Use hyphens (e.g., `/api-keys`, `/workflow-evaluations`, `/supervised-fine-tuning`)
- RR7 API routes (React Router resource routes): Use underscores (e.g., `/api/workflow_evaluations/search_runs`)

This convention aligns frontend URLs with web standards (hyphens) while backend/API routes use underscores for consistency with the gateway API.
