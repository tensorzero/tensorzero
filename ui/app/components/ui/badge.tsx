import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "~/utils/common";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-hidden focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        destructive:
          "border-transparent bg-destructive text-destructive-foreground",
        outline: "text-foreground",
        warning: "bg-yellow-600 text-white",
      },
      shadow: {
        true: "shadow-sm",
        false: "",
      },
      hover: {
        true: "hover:bg-primary/80",
        false: "",
      },
    },
    defaultVariants: {
      variant: "default",
      shadow: false,
      hover: false,
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, shadow, hover, ...props }: BadgeProps) {
  return (
    <div
      className={cn(badgeVariants({ variant, shadow, hover }), className)}
      {...props}
    />
  );
}

export { Badge, badgeVariants };
