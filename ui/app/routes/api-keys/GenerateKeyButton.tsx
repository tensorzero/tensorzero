import { Plus } from "lucide-react";
import { Button } from "~/components/ui/button";
import { useReadOnly } from "~/context/read-only";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

interface GenerateKeyButtonProps {
  onClick: () => void;
  className?: string;
}

export function GenerateKeyButton({
  className,
  ...props
}: GenerateKeyButtonProps) {
  const isReadOnly = useReadOnly();

  return (
    <ReadOnlyGuard>
      <Button
        variant="outline"
        size="sm"
        className={className}
        disabled={isReadOnly}
        {...props}
      >
        <Plus className="text-fg-tertiary mr-2 h-4 w-4" aria-hidden />
        Generate API Key
      </Button>
    </ReadOnlyGuard>
  );
}
