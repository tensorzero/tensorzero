import { Plus } from "lucide-react";
import { Button } from "~/components/ui/button";
import { useReadOnly } from "~/context/read-only";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

interface NewDatapointButtonProps {
  onClick: () => void;
  className?: string;
}

export function NewDatapointButton(props: NewDatapointButtonProps) {
  const isReadOnly = useReadOnly();

  return (
    <ReadOnlyGuard asChild>
      <Button variant="outline" size="sm" disabled={isReadOnly} {...props}>
        <Plus className="text-fg-tertiary mr-2 h-4 w-4" />
        New Datapoint
      </Button>
    </ReadOnlyGuard>
  );
}
