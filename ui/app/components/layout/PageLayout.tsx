import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type { ReactNode } from "react";

// PageLayout component
interface PageLayoutProps {
  children: ReactNode;
}

const PageLayout = ({ children }: PageLayoutProps) => {
  return (
    <div className="container mx-auto flex flex-col gap-12 px-8 pb-20 pt-16">
      {children}
    </div>
  );
};

// PageHeader component
interface PageHeaderProps {
  label?: string;
  heading?: string;
  name?: string;
  count?: number;
  lateral?: string;
  icon?: ReactNode;
  iconBg?: string;
  children?: ReactNode;
}

const PageHeader = ({
  heading,
  label,
  name,
  count,
  lateral,
  icon,
  iconBg = "bg-none",
  children,
}: PageHeaderProps) => {
  return (
    <div className="flex flex-col">
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
            <span className="font-mono text-2xl font-medium leading-none">
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
};

// SectionsGroup component
interface SectionsGroupProps {
  children: ReactNode;
}

const SectionsGroup = ({ children }: SectionsGroupProps) => {
  return <div className="flex flex-col gap-12">{children}</div>;
};

// SectionLayout component
interface SectionLayoutProps {
  children: ReactNode;
}

const SectionLayout = ({ children }: SectionLayoutProps) => {
  return <div className="flex flex-col gap-4">{children}</div>;
};

// SectionHeader component
interface SectionHeaderProps {
  heading: string;
  count?: number;
  badge?: {
    name: string;
    tooltip: string;
  };
}

const SectionHeader = ({ heading, count, badge }: SectionHeaderProps) => {
  return (
    <h2 className="flex items-center gap-2 text-xl font-medium">
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
};

export { PageHeader, SectionHeader, SectionLayout, SectionsGroup, PageLayout };
