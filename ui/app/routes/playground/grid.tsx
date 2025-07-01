export enum GridRow {
  // Function/variant section
  Header = "header",
  Metadata = "metadata",
  SystemPrompt = "system",
  AssistantPrompt = "assistant",
  UserPrompt = "user",
  OutputSchema = "output-schema",
  Footer = "footer", // TODO remove?

  // Datapoints section
  Datapoints = "datapoints",
}

export const GRID_TEMPLATE_ROWS = `
[${GridRow.Header}]          min-content
[${GridRow.Metadata}]        min-content
[${GridRow.SystemPrompt}]    min-content
[${GridRow.AssistantPrompt}] min-content
[${GridRow.UserPrompt}]      min-content
[${GridRow.OutputSchema}]    min-content
[${GridRow.Footer}]          min-content
[${GridRow.Datapoints}]      1fr
`;

// TODO this won't work!!!
/** Helper to generate the Tailwind class to assign an element to a particular row in the grid layout */
export const setRow = (row: GridRow) => `row-[${row}]`;

// TODO Tailwind dynamic classes won't be included!
/**
 * row-[header]
 * row-[metadata]
 * row-[system]
 * row-[assistant]
 * row-[user]
 * row-[output-schema]
 * row-[footer]
 * row-[datapoints]
 */
