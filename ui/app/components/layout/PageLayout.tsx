import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type { ReactNode } from "react";
import { forwardRef } from "react";

// PageHeader component
interface PageHeaderProps {
  label?: string;
  heading?: string;
  name?: string;
  count?: number;
  lateral?: string;
  className?: string;
}

const PageHeader = forwardRef<HTMLDivElement, PageHeaderProps>(
  ({ heading, label, name, count, lateral, className }, ref) => {
    return (
      <div ref={ref} className={`${className || ""}`}>
        {label !== undefined && (
          <p className="text-sm font-normal text-foreground-muted">{label}</p>
        )}
        <div className="flex items-baseline gap-2">
          {heading !== undefined && (
            <h4 className="text-2xl font-medium">{heading}</h4>
          )}
          {name !== undefined && (
            <span className="rounded-lg bg-background-tertiary px-1.5 py-1 font-mono text-2xl font-semibold leading-none">
              {name}
            </span>
          )}
          {lateral !== undefined && (
            <p className="text-sm font-normal text-foreground-muted">
              {lateral}
            </p>
          )}
          {count !== undefined && (
            <h4 className="text-2xl font-medium text-foreground-muted">
              {count.toLocaleString()}
            </h4>
          )}
        </div>
      </div>
    );
  },
);
PageHeader.displayName = "PageHeader";

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
      <h3
        ref={ref}
        className={`flex items-center gap-2 text-xl font-medium ${className || ""}`}
      >
        {heading}

        {count !== undefined && (
          <span className="text-xl font-medium text-foreground-muted">
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
      </h3>
    );
  },
);
SectionHeader.displayName = "SectionHeader";

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
        className={`flex flex-col gap-8 pb-20 pt-16 ${className || ""}`}
      >
        {children}
      </div>
    );
  },
);
PageLayout.displayName = "PageLayout";

export { PageHeader, SectionHeader, SectionLayout, SectionsGroup, PageLayout };
