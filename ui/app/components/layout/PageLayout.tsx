import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type { ReactNode } from "react";
import { forwardRef } from "react";
import { cn } from "~/utils/common";

// PageLayout component
interface PageLayoutProps {
  children: ReactNode;
  className?: string;
}

const PageLayout = forwardRef<HTMLDivElement, PageLayoutProps>(
  ({ children, className }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "container mx-auto flex flex-col gap-12 px-8 pt-16 pb-20",
          className,
        )}
      >
        {children}
      </div>
    );
  },
);
PageLayout.displayName = "PageLayout";

// PageHeader component
interface PageHeaderProps {
  label?: string;
  heading?: string;
  name?: string;
  count?: number;
  lateral?: string;
  className?: string;
  icon?: ReactNode;
  iconBg?: string;
  children?: ReactNode;
}

const PageHeader = forwardRef<HTMLDivElement, PageHeaderProps>(
  (
    {
      heading,
      label,
      name,
      count,
      lateral,
      className,
      icon,
      iconBg = "bg-none",
      children,
    },
    ref,
  ) => {
    return (
      <div ref={ref} className={cn("flex flex-col", className)}>
        <div className="flex flex-col gap-2">
          {label !== undefined && (
            <p className="text-fg-secondary flex items-center gap-1.5 text-sm font-normal">
              {icon && (
                <div
                  className={`${iconBg} flex size-5 items-center justify-center rounded-sm`}
                >
                  {icon}
                </div>
              )}
              {label}
            </p>
          )}
          <div className="flex items-baseline gap-2">
            {heading !== undefined && (
              <h1 className="text-2xl font-medium">{heading}</h1>
            )}
            {name !== undefined && (
              <span className="font-mono text-2xl leading-none font-medium">
                {name}
              </span>
            )}
            {lateral !== undefined && (
              <p className="text-fg-tertiary text-sm font-normal">{lateral}</p>
            )}
            {count !== undefined && (
              <h1 className="text-fg-muted text-2xl font-medium">
                {count.toLocaleString()}
              </h1>
            )}
          </div>
        </div>
        {children && <div className="mt-8 flex flex-col gap-8">{children}</div>}
      </div>
    );
  },
);
PageHeader.displayName = "PageHeader";

// SectionsGroup component
interface SectionsGroupProps {
  children: ReactNode;
  className?: string;
}

const SectionsGroup = forwardRef<HTMLDivElement, SectionsGroupProps>(
  ({ children, className }, ref) => {
    return (
      <div ref={ref} className={`flex flex-col gap-12 ${className || ""}`}>
        {children}
      </div>
    );
  },
);
SectionsGroup.displayName = "SectionsGroup";

// SectionLayout component
interface SectionLayoutProps {
  children: ReactNode;
  className?: string;
}

const SectionLayout = forwardRef<HTMLDivElement, SectionLayoutProps>(
  ({ children, className }, ref) => {
    return (
      <div ref={ref} className={`flex flex-col gap-4 ${className || ""}`}>
        {children}
      </div>
    );
  },
);
SectionLayout.displayName = "SectionLayout";

// SectionHeader component
interface SectionHeaderProps {
  heading: string;
  count?: number;
  badge?: {
    name: string;
    tooltip: string;
  };
  className?: string;
}

const SectionHeader = forwardRef<HTMLHeadingElement, SectionHeaderProps>(
  ({ heading, count, badge, className }, ref) => {
    return (
      <h2
        ref={ref}
        className={`flex items-center gap-2 text-xl font-medium ${className || ""}`}
      >
        {heading}

        {count !== undefined && (
          <span className="text-fg-muted text-xl font-medium">
            {count.toLocaleString()}
          </span>
        )}

        {badge && (
          <TooltipProvider>
            <Tooltip delayDuration={0}>
              <TooltipTrigger asChild>
                <Badge
                  variant="outline"
                  className="ml-1 px-2 py-0.5 text-xs font-medium"
                >
                  {badge.name}
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">{badge.tooltip}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </h2>
    );
  },
);
SectionHeader.displayName = "SectionHeader";

export { PageHeader, SectionHeader, SectionLayout, SectionsGroup, PageLayout };
