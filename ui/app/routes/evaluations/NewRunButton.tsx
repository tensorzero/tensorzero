import { Plus } from "lucide-react";
import { Button } from "~/components/ui/button";
import { useReadOnly } from "~/context/read-only";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

interface NewRunButtonProps {
  onClick: () => void;
  className?: string;
}

export function NewRunButton({ className, ...props }: NewRunButtonProps) {
  const isReadOnly = useReadOnly();

  return (
    <ReadOnlyGuard asChild>
      <Button
        variant="outline"
        size="sm"
        className={className}
        disabled={isReadOnly}
        {...props}
      >
        <Plus className="text-fg-tertiary mr-2 h-4 w-4" aria-hidden />
        New Run
      </Button>
    </ReadOnlyGuard>
  );
}
