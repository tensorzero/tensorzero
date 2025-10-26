import { AddButton } from "~/components/ui/AddButton";
import { DeleteButton } from "~/components/ui/DeleteButton";
import type { JsonValue } from "~/types/tensorzero";
import TextContentBlock from "./content_blocks/TextContentBlock";
import TemplateContentBlock from "./content_blocks/TemplateContentBlock";
import MessageWrapper from "./MessageWrapper";
import ExpandableElement from "./ExpandableElement";

interface SystemElementProps {
  system?: JsonValue;
  isEditing?: boolean;
  onSystemChange?: (system: string | object | null) => void;
  maxHeight?: number | "Content";
}

export default function SystemElement({
  system,
  isEditing,
  onSystemChange,
  maxHeight,
}: SystemElementProps) {
  if (system == null) {
    return (
      isEditing && (
        <MessageWrapper role="system">
          <div className="flex items-center gap-2 py-2">
            <AddButton label="Text" onAdd={() => onSystemChange?.("")} />
            {/* TODO (GabrielBianconi): we should hide the following button if this function has no variants with a `system` template; it'll error on submission */}
            <AddButton label="Template" onAdd={() => onSystemChange?.({})} />
          </div>
        </MessageWrapper>
      )
    );
  }

  // TODO (GabrielBianconi): The fact we need this means that `system` is not sufficiently narrowly typed.
  // https://github.com/tensorzero/tensorzero/issues/4187
  if (
    typeof system === "boolean" ||
    typeof system === "number" ||
    Array.isArray(system)
  ) {
    throw new Error(
      `Invalid system: expected string or object, got ${typeof system}`,
    );
  }

  const systemDisplay =
    typeof system === "object" ? (
      <TemplateContentBlock
        block={{
          name: "system",
          arguments: system,
        }}
        isEditing={isEditing}
        onChange={(block) => onSystemChange?.(block.arguments)}
      />
    ) : (
      <TextContentBlock
        label="Text"
        text={system}
        isEditing={isEditing}
        onChange={onSystemChange}
      />
    );

  // In editing mode, show a delete button in the action bar
  const actionBar = isEditing ? (
    <DeleteButton
      label="Delete system"
      onDelete={() => onSystemChange?.(null)}
    />
  ) : undefined;

  return (
    <ExpandableElement maxHeight={maxHeight}>
      <MessageWrapper role="system" actionBar={actionBar}>
        {systemDisplay}
      </MessageWrapper>
    </ExpandableElement>
  );
}
