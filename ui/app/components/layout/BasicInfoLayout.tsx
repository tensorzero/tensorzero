import { cn } from "~/utils/common";

export const BasicInfoLayout: React.FC<React.ComponentProps<"div">> = ({
  children,
  className,
  ...props
}) => (
  <div className={cn("flex flex-col gap-4 md:gap-2", className)} {...props}>
    {children}
  </div>
);

export const BasicInfoItem: React.FC<React.ComponentProps<"div">> = ({
  children,
  className,
  ...props
}) => (
  <div
    className={cn("flex flex-col gap-0.5 md:flex-row", className)}
    {...props}
  >
    {children}
  </div>
);

export const BasicInfoItemTitle: React.FC<React.ComponentProps<"div">> = ({
  children,
  className,
  ...props
}) => (
  <div
    className={cn(
      "text-fg-secondary w-full flex-shrink-0 text-left text-sm md:w-32 md:py-1",
      className,
    )}
    {...props}
  >
    {children}
  </div>
);

export const BasicInfoItemContent: React.FC<React.ComponentProps<"div">> = ({
  children,
  className,
  ...props
}) => (
  <div
    className={cn(
      "text-fg-primary flex flex-wrap gap-x-4 gap-y-0.5 truncate md:gap-1 md:py-1",
      className,
    )}
    {...props}
  >
    {children}
  </div>
);
