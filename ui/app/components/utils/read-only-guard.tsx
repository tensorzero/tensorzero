"use client";

import { Slot } from "radix-ui";
import { useReadOnly } from "~/context/read-only";

export interface ReadOnlyGuardProps
  extends React.ComponentPropsWithRef<"button"> {
  asChild?: boolean;
}

export function ReadOnlyGuard({ asChild, ...props }: ReadOnlyGuardProps) {
  const Component = asChild ? Slot.Root : "button";
  const isReadOnly = useReadOnly();
  return <Component {...props} disabled={isReadOnly || props.disabled} />;
}
