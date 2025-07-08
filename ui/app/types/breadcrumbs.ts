import type { UIMatch } from "react-router";

declare module "react-router" {
  interface RouteHandle {
    /** Should it hide breadcrumbs for this route? */
    hideBreadcrumbs?: boolean;
    /** Append 0, 1 or more breadcrumbs given current segment. Used to hierarchically generate page title. */
    crumb?: (match: UIMatch) => string[];
  }
}
