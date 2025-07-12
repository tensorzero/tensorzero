import { cva } from "class-variance-authority";

// TODO Rename to function section or something
export const playgroundGrid = cva("", {
  variants: {
    row: {
      // Header/title
      header: "row-[header]",

      // Common metadata
      metadata: "row-[metadata]",

      // Prompt section
      promptSectionHeader: "row-[prompt-section-header]",
      system: "row-[system]",
      assistant: "row-[assistant]",
      user: "row-[user]",
      outputSchema: "row-[output-schema]",

      // Advanced parameters section
      footer: "row-[footer]",
    },
  },
});

// TODO Set a max-height for each of these
export const PLAYGROUND_GRID_ROWS = `
[header]                min-content
[metadata]              min-content

[prompt-section-header] min-content
[system]                min-content
[assistant]             min-content
[user]                  min-content
[output-schema]         min-content

[footer]                1fr
`;
