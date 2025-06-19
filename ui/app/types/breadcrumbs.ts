import type { UIMatch } from "react-router";

declare module "react-router" {
  interface RouteHandle {
    /** Append 0, 1 or more breadcrumbs given current segment */
    crumb?: (match: UIMatch) => string[];
  }
}
