interface PageHeaderProps {
  headline: string;
  count?: number;
  itemId?: string;
}

export function PageHeader({ headline, count, itemId }: PageHeaderProps) {
  return (
    <div className="pb-8 pt-16">
      <div className="flex items-center gap-2">
        <h4 className="text-2xl font-medium">{headline}</h4>
        {itemId !== undefined && (
          <h4 className="text-2xl font-medium text-foreground-tertiary">
            {itemId}
          </h4>
        )}
        {count !== undefined && (
          <h4 className="text-2xl font-medium text-foreground-tertiary">
            {count.toLocaleString()}
          </h4>
        )}
      </div>
    </div>
  );
}
