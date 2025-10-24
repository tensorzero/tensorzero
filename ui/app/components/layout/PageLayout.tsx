import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type { ReactNode } from "react";
import { cn } from "~/utils/common";

const PageLayout: React.FC<React.ComponentProps<"div">> = ({
  children,
  className,
  ...props
}) => (
  <div
    className={cn(
      "container mx-auto flex flex-col gap-12 px-8 pt-16 pb-20",
      className,
    )}
    {...props}
  >
    {children}
  </div>
);

interface PageHeaderProps {
  label?: string;
  heading?: string;
  name?: string;
  count?: number | bigint;
  icon?: ReactNode;
  iconBg?: string;
  children?: ReactNode;
  tag?: ReactNode;
}

const PageHeader: React.FC<PageHeaderProps> = ({
  heading,
  label,
  name,
  count,
  icon,
  iconBg = "bg-none",
  children,
  tag,
}: PageHeaderProps) => {
  return (
    <div className="flex flex-col">
      <div className="flex flex-col gap-2">
        {label !== undefined && (
          <div className="text-fg-secondary flex items-center gap-1.5 text-sm font-normal">
            {icon && (
              <span
                className={`${iconBg} flex size-5 items-center justify-center rounded-sm`}
              >
                {icon}
              </span>
            )}

            {label}
          </div>
        )}
        <div className="flex items-center gap-2">
          {heading !== undefined && (
            <h1 className="text-2xl font-medium">{heading}</h1>
          )}
          {name !== undefined && (
            <span className="font-mono text-2xl leading-none font-medium">
              {name}
            </span>
          )}
          {count !== undefined && (
            <h1 className="text-fg-muted text-2xl font-medium">
              {count.toLocaleString()}
            </h1>
          )}

          {tag}
        </div>
      </div>

      {/* TODO Use wrapper for this instead - feels strange here */}
      {children && <div className="mt-8 flex flex-col gap-8">{children}</div>}
    </div>
  );
};

const SectionsGroup: React.FC<React.ComponentProps<"section">> = ({
  children,
  className,
  ...props
}) => (
  <section className={cn("flex flex-col gap-12", className)} {...props}>
    {children}
  </section>
);

const SectionLayout: React.FC<React.ComponentProps<"section">> = ({
  children,
  className,
  ...props
}) => (
  <section className={cn("flex flex-col gap-4", className)} {...props}>
    {children}
  </section>
);

interface SectionHeaderProps extends React.PropsWithChildren {
  heading: string;
  count?: number;
  badge?: {
    name: string;
    tooltip: string;
  };
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  heading,
  count,
  badge,
  children,
}) => (
  <h2 className="flex items-center gap-2 text-xl font-medium">
    {heading}

    {count !== undefined && (
      <span className="text-fg-muted text-xl font-medium">
        {count.toLocaleString()}
      </span>
    )}

    {badge && (
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
    )}

    {children}
  </h2>
);

export { PageHeader, SectionHeader, SectionLayout, SectionsGroup, PageLayout };
