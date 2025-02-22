interface PageHeaderProps {
  label?: string;
  headline: string;
  count?: number;
  lateral?: string;
}

export function PageHeader({
  headline,
  label,
  count,
  lateral,
}: PageHeaderProps) {
  return (
    <div className="pb-8 pt-16">
      {label !== undefined && (
        <p className="text-sm font-normal text-foreground-muted">{label}</p>
      )}
      <div className="flex items-baseline gap-2">
        <h4 className="text-2xl font-medium">{headline}</h4>
        {lateral !== undefined && (
          <p className="text-sm font-normal text-foreground-muted">{lateral}</p>
        )}
        {count !== undefined && (
          <h4 className="text-2xl font-medium text-foreground-muted">
            {count.toLocaleString()}
          </h4>
        )}
      </div>
    </div>
  );
}
