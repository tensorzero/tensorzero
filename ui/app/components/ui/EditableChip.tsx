import { useState } from "react";
import Chip, { type ChipProps } from "~/components/ui/Chip";
import { Input } from "~/components/ui/input";
import { EditButton } from "../utils/EditButton";
import { CancelButton } from "../utils/CancelButton";
import { SaveButton } from "../utils/SaveButton";

interface EditableChipProps extends ChipProps {
  // EditableChip-specific prop
  onConfirm?: (newValue: string) => void | Promise<void>;
}

export default function EditableChip({
  onConfirm,
  ...chipProps
}: EditableChipProps) {
  const { label, font = "sans" } = chipProps;
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleEditClick = () => {
    // TODO: distinguish between empty and default label.
    // Callsite is <EditableChip label={datapoint.name || "-"} />, and if datapoint.name is empty,
    // we don't want to start with "-" as the edit value.
    setEditValue(label);
    setIsEditing(true);
  };

  const handleCancel = () => {
    setIsEditing(false);
    setEditValue("");
  };

  const handleConfirm = async () => {
    if (!onConfirm) return;

    setIsLoading(true);
    try {
      await onConfirm(editValue);
      setIsEditing(false);
      setEditValue("");
    } catch (error) {
      // Error handling will be managed by the parent component
      console.error("Failed to update value:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleConfirm();
    } else if (e.key === "Escape") {
      handleCancel();
    }
  };

  if (isEditing) {
    return (
      <div className="flex items-center gap-2">
        <Input
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onKeyDown={handleKeyDown}
          className={`h-5 ml-1 text-sm ${font === "mono" ? "font-mono" : ""}`}
          disabled={isLoading}
          autoFocus
        />
        <SaveButton
          onClick={handleConfirm}
          className="size-5"
          disabled={isLoading}
        />
        <CancelButton onClick={handleCancel} className="size-5" />
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <Chip {...chipProps} />
      {onConfirm && (
        <EditButton
          className="size-5"
          onClick={handleEditClick}
          tooltip="Rename datapoint"
        />
      )}
    </div>
  );
}
