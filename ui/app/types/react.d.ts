import "react";

declare module "react" {
  interface CSSProperties {
    // Allows TypeScript to understand custom CSS properties passed to component
    // style props.
    [key: `--${string}`]: string | number;
  }
}
