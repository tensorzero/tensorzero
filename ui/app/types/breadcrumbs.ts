import type { UIMatch } from "react-router";

export interface CrumbItem {
  label: string;
  isIdentifier?: boolean;
  noLink?: boolean;
}

declare module "react-router" {
  interface RouteHandle {
    /** Should it hide breadcrumbs for this route? */
    hideBreadcrumbs?: boolean;
    /** Append 0, 1 or more breadcrumbs given current segment. Used to hierarchically generate page title. */
    crumb?: (match: UIMatch) => (string | CrumbItem)[];
  }
}
