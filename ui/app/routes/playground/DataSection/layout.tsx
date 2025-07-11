import { cva } from "class-variance-authority";

// TODO Make this grid internal to each cell.

/** Row within the datapoint grid (which is itself a row) */
export const datapointGrid = cva("", {
  variants: {
    /** Each datapoint row has its subgrid: columns inherit from the parent grid, but rows are defined independently. */
    row: {
      header: "row-[header]",
      input: "row-[input]",
      output: "row-[output]",
    },
  },
});

export const DATAPOINT_GRID_TEMPLATE_ROWS = `
[header]  min-content
[input]   min-content
[output]  min-content
`;
