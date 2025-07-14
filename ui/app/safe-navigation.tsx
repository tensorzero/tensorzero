/* eslint-disable no-restricted-imports */

import { useCallback } from "react";
import {
  href,
  useNavigate as useOriginalNavigate,
  Link as OriginalLink,
  type Register,
  type NavigateOptions,
  type To as OriginalTo,
} from "react-router";

/**
 * Complex, internal React Router types
 */

type Pages = Register["pages"];
type Equal<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends <T>() => T extends Y ? 1 : 2
    ? true
    : false;
type ToArgs<Params extends Record<string, string | undefined>> =
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  Equal<Params, {}> extends true
    ? []
    : Partial<Params> extends Params
      ? [Params] | []
      : [Params];
type Args = {
  [K in keyof Pages]: ToArgs<Pages[K]["params"]>;
};

/** Paths with no dynamic parameters (or optional dynamic parameters) */
export type SimplePaths = {
  [K in keyof Args]: [] extends Args[K] ? K : never;
}[keyof Args];

/**
 * Type-checked paths and dynamic path parameters for navigation.
 * (Using React Router's [href](https://reactrouter.com/api/utils/href) function).
 *
 * #### Examples
 *
 * **Simple paths with no parameters:**
 * "/observability/inferences"
 *
 * **Parameterized paths with parameters as tuple**:
 * ["/datasets/:dataset_name", { dataset_name: "my-dataset" }]
 *
 * **Complex paths with search and hash:**
 * {
 *   path: ["/datasets/:dataset_name", { dataset_name: "my-dataset" }],
 *   search: "?offset=10",
 *   hash: "#section",
 * }
 *
 * **Full HTTPS URLs:**
 * "https://example.com/"
 */
export type To<Path extends keyof Args = keyof Args> =
  | `https://${string}`
  | SimplePaths
  | [path: Path, ...args: Args[Path]]
  | {
      pathname?: SimplePaths | [path: Path, ...args: Args[Path]];
      search?: string;
      hash?: string;
    }
  | URLSearchParams;

type LinkProps<Path extends keyof Args> = Omit<
  React.ComponentProps<typeof OriginalLink>,
  "to"
> &
  (
    | {
        to: To<Path>;
      }
    | {
        /**
         * Direct `href` for anchor element. Prefer the `to` prop.
         * This is useful to override type-checking for `data:` URLs, object URLs, and other composite URLs.
         */
        unsafeTo: string;
      }
  );

const isFullUrl = (href: string): href is `https://${string}` =>
  href.startsWith("https://");

function encodeParams<T extends Record<string, string | undefined>>(
  params: T,
): T {
  const encoded: Record<string, string | undefined> = {};
  for (const [key, value] of Object.entries(params)) {
    encoded[key] = value !== undefined ? encodeURIComponent(value) : undefined;
  }
  return encoded as T;
}

const buildHref = <Path extends keyof Pages>(to: To<Path>): OriginalTo => {
  let pathname: string | undefined;
  let search: string | undefined;
  let hash: string | undefined;

  // Use React Router's `href` function to safely interpolate dynamic path parameters

  // Handle full "https://" URLs and simple paths with no dynamic params
  if (typeof to === "string") {
    pathname = isFullUrl(to) ? to : href(to);
  }
  // Handle tuple of path and dynamic path parameters to interpolate
  else if (Array.isArray(to)) {
    const [path, ...args] = to;
    const encodedArgs = args.map(encodeParams) as typeof args;
    pathname = href(path, ...encodedArgs);
  }
  // Handle search parameters on the same page
  else if (to instanceof URLSearchParams) {
    search = to.toString();
  }
  // Handle object with dynamic path, search, and hash
  else {
    if (typeof to.pathname === "string") {
      pathname = href(to.pathname);
    } else if (to.pathname) {
      const [path, ...args] = to.pathname;
      const encodedArgs = args.map(encodeParams) as typeof args;
      pathname = href(path, ...encodedArgs);
    }
    search = to.search;
    hash = to.hash;
  }

  return {
    pathname,
    search,
    hash,
  };
};

export function Link<Path extends keyof Pages>(props: LinkProps<Path>) {
  return (
    <OriginalLink
      {...props}
      to={"unsafeTo" in props ? props.unsafeTo : buildHref(props.to)}
    />
  );
}

export function useNavigate() {
  const navigate = useOriginalNavigate();

  return useCallback(
    <Path extends keyof Args>(to: To<Path>, options?: NavigateOptions) => {
      return navigate(buildHref(to), options);
    },
    [navigate],
  );
}
