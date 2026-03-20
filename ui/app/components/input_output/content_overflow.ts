export type ContentOverflow =
  | { type: "expandable"; maxHeight: number }
  | { type: "scroll"; maxHeight: number };
