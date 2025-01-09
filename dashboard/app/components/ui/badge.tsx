import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "~/utils/common";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        destructive:
          "border-transparent bg-destructive text-destructive-foreground",
        outline: "text-foreground",
      },
      hover: {
        true: {
          default: "hover:bg-primary/80",
          secondary: "hover:bg-secondary/80",
          destructive: "hover:bg-destructive/80",
          outline: "",
        },
        false: "",
      },
      shadow: {
        true: {
          default: "shadow",
          secondary: "",
          destructive: "shadow",
          outline: "",
        },
        false: "",
      },
    },
    defaultVariants: {
      variant: "default",
      hover: false,
      shadow: false,
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {
  hover?: boolean;
  shadow?: boolean;
}

function Badge({ className, variant, hover, shadow, ...props }: BadgeProps) {
  return (
    <div
      className={cn(badgeVariants({ variant, hover, shadow }), className)}
      {...props}
    />
  );
}

export { Badge, badgeVariants };
