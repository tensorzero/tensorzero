import * as React from "react";
import { Slot } from "radix-ui";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "~/utils/common";
import type { IconProps } from "../icons/Icons";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-hidden focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 select-none",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow-sm hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline:
          "border border-input text-fg bg-bg hover:bg-bg-hover hover:text-accent-foreground",
        secondary: "bg-bg-hover text-secondary-foreground hover:bg-bg-hover/80",
        ghost: "hover:bg-accent text-fg hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-sm",
        lg: "h-10 rounded-md px-8",
        icon: "h-9 w-9",
        iconSm: "h-8 w-8",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export type ButtonVariant = VariantProps<typeof buttonVariants>["variant"];
export type ButtonSize = VariantProps<typeof buttonVariants>["size"];

interface ButtonContextValue {
  variant: ButtonVariant;
  size: ButtonSize;
}

const ButtonContext = React.createContext<ButtonContextValue | null>(null);
ButtonContext.displayName = "ButtonContext";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  /**
   * When using `asChild` only a single child is allowed. Using `slotLeft` or
   * `slotRight` allows rendering elements adjacent to the child.
   */
  slotLeft?: React.ReactNode;
  slotRight?: React.ReactNode;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = "default",
      size = "default",
      asChild = false,
      children,
      slotLeft,
      slotRight,
      ...props
    },
    ref,
  ) => {
    const Comp = asChild ? Slot.Root : "button";
    const child = asChild ? (
      <Slot.Slottable>{children}</Slot.Slottable>
    ) : (
      children
    );

    return (
      <Comp
        className={cn(
          "cursor-pointer",
          buttonVariants({ variant, size, className }),
        )}
        ref={ref}
        {...props}
      >
        <ButtonContext value={{ variant, size }}>
          {slotLeft}
          {child}
          {slotRight}
        </ButtonContext>
      </Comp>
    );
  },
);
Button.displayName = "Button";

const buttonIconVariants = cva(null, {
  variants: {
    variant: {
      default: "text-current",
      muted: "text-fg-muted",
      tertiary: "text-fg-tertiary",
    },
    size: {
      default: "h-4 w-4",
      sm: "h-3 w-3",
      lg: "h-5 w-5",
      icon: "h-4 h-4",
      iconSm: "h-3 w-3",
    },
  },
  defaultVariants: {
    variant: "default",
    size: "default",
  },
});

interface ButtonIconProps
  extends Omit<IconProps, "size">,
    Omit<VariantProps<typeof buttonIconVariants>, "size"> {
  as: React.ElementType<IconProps>;
}

export function ButtonIcon({
  as: Comp,
  className,
  variant,
  ...props
}: ButtonIconProps) {
  const { size = "default" } = React.use(ButtonContext) || {};
  return (
    <Comp
      {...props}
      aria-hidden
      className={cn(buttonIconVariants({ variant, size, className }))}
    />
  );
}

export { Button, buttonVariants };
