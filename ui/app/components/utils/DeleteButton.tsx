import { Trash2 } from "lucide-react";
import { Button } from "~/components/ui/button";
import { useState } from "react";

interface DeleteButtonProps {
  onClick: () => void;
  className?: string;
  isLoading?: boolean;
}

export function DeleteButton({
  onClick,
  className,
  isLoading,
}: DeleteButtonProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  const handleConfirm = () => {
    onClick();
    setConfirmDelete(false);
  };

  const handleCancel = () => {
    setConfirmDelete(false);
  };

  const handleInitialClick = () => {
    setConfirmDelete(true);
  };

  if (confirmDelete) {
    return (
      <div className="flex gap-2">
        <Button
          variant="destructive"
          size="sm"
          className="group bg-red-600 text-white hover:bg-red-700"
          disabled={isLoading}
          onClick={handleConfirm}
        >
          {isLoading ? "Deleting..." : "Yes, delete permanently"}
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="bg-bg-muted"
          disabled={isLoading}
          onClick={handleCancel}
        >
          No, keep it
        </Button>
      </div>
    );
  }

  return (
    <Button
      variant="outline"
      size="iconSm"
      className={className}
      disabled={isLoading}
      onClick={handleInitialClick}
    >
      <Trash2 className="h-4 w-4" />
    </Button>
  );
}
