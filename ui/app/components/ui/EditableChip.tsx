import { useMemo, useState } from "react";
import Chip, { type ChipProps } from "~/components/ui/Chip";
import { Input } from "~/components/ui/input";
import { EditButton } from "../utils/EditButton";
import { CancelButton } from "../utils/CancelButton";
import { SaveButton } from "../utils/SaveButton";

/**
 * EditableChipProps extends ChipProps and adds EditableChip-specific props.
 *
 * @extends ChipProps
 * @property {string} defaultLabel - The label to render if the label is nullish. If not provided, renders an empty string if label is empty.
 * @property {function} onConfirm - The callback function for confirmation.
 */
interface EditableChipProps extends Omit<ChipProps, "label"> {
  label: string | null | undefined;
  defaultLabel?: string;
  onConfirm?: (newValue: string) => void | Promise<void>;
}

/**
 * EditableChip is a component that allows the user to edit the label of a Chip.
 */
export default function EditableChip({
  onConfirm,
  defaultLabel,
  ...chipProps
}: EditableChipProps) {
  const { label, font = "sans" } = chipProps;
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const labelToRender = useMemo(() => {
    if (label !== null && label !== undefined) {
      return label;
    }
    if (defaultLabel !== null && defaultLabel !== undefined) {
      return defaultLabel;
    }
    return "";
  }, [label, defaultLabel]);

  const handleEditClick = () => {
    setEditValue(label || "");
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
          className={`ml-1 h-5 text-sm ${font === "mono" ? "font-mono" : ""}`}
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
      <Chip {...chipProps} label={labelToRender} />
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
