import { Lightbulb } from "lucide-react";
import { type ReactNode } from "react";
import { ContentBlockLabel } from "~/components/input_output/content_blocks/ContentBlockLabel";
import { CodeEditor } from "~/components/ui/code-editor";
import { Input } from "~/components/ui/input";
import { AddButton } from "~/components/ui/AddButton";
import { DeleteButton } from "~/components/ui/DeleteButton";
import type { Thought } from "~/types/tensorzero";

interface ThoughtContentBlockProps {
  block: Thought;
  isEditing?: boolean;
  onChange?: (updatedBlock: Thought) => void;
  actionBar?: ReactNode;
}

export function ThoughtContentBlock({
  block,
  isEditing,
  onChange,
  actionBar,
}: ThoughtContentBlockProps) {
  // TODO (GabrielBianconi): The E2E tests don't cover editing summaries entirely.
  // Once this makes it to the application, we should add tests for editing summaries.

  // In editing mode, show all fields even if undefined
  const showText = isEditing || block.text !== undefined;
  const showSignature = isEditing || block.signature !== undefined;
  const showSummary = isEditing || block.summary !== undefined;

  const isEmpty =
    block.text === undefined &&
    block.signature === undefined &&
    (block.summary === undefined || block.summary.length === 0);

  const onDeleteSummaryBlock = (index: number) => {
    const currentSummary = block.summary ?? [];
    const updatedSummary = [...currentSummary];
    updatedSummary.splice(index, 1);
    onChange?.({
      ...block,
      summary: updatedSummary.length > 0 ? updatedSummary : undefined,
    });
  };

  const onAddSummaryBlock = () => {
    const currentSummary = block.summary ?? [];
    onChange?.({
      ...block,
      summary: [
        ...currentSummary,
        {
          type: "summary_text",
          text: "",
        },
      ],
    });
  };

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<Lightbulb className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        Thought
      </ContentBlockLabel>
      <div className="border-border bg-bg-tertiary/50 grid grid-flow-row grid-cols-[min-content_1fr] place-content-center gap-x-4 gap-y-1 rounded-sm px-3 py-2 text-xs">
        {/* Empty state */}
        {isEmpty && !isEditing && (
          <div className="text-fg-muted col-span-2 flex items-center justify-center py-8 text-sm">
            Empty thought
          </div>
        )}

        {/* Text field */}
        {showText && (
          <>
            <p className="text-fg-secondary font-medium">Text</p>
            <CodeEditor
              value={block.text ?? ""}
              className="bg-bg-secondary"
              readOnly={!isEditing}
              onChange={(updatedText) => {
                // Store empty string as undefined
                onChange?.({
                  ...block,
                  text: updatedText.trim() === "" ? undefined : updatedText,
                });
              }}
            />
          </>
        )}

        {/* Signature field  */}
        {showSignature && (
          <>
            <p className="text-fg-secondary font-medium">Signature</p>
            {!isEditing ? (
              <p className="self-center truncate font-mono text-[0.6875rem]">
                {block.signature}
              </p>
            ) : (
              <Input
                type="text"
                value={block.signature ?? ""}
                className="font-mono"
                data-testid="thought-signature-input"
                onChange={(e) => {
                  // Store empty string as undefined
                  onChange?.({
                    ...block,
                    signature:
                      e.target.value.trim() === "" ? undefined : e.target.value,
                  });
                }}
              />
            )}
          </>
        )}

        {/* Summary field */}
        {showSummary && (
          <>
            <p className="text-fg-secondary font-medium">Summary</p>
            <div className="flex flex-col gap-1">
              {(block.summary ?? []).map((summaryBlock, index) => (
                <div
                  key={index}
                  className="border-border bg-bg-secondary flex items-center gap-2 rounded px-2 py-1 text-xs"
                >
                  <div className="flex-1">
                    {summaryBlock.type === "summary_text" && (
                      <CodeEditor
                        value={summaryBlock.text}
                        className="bg-bg-tertiary"
                        readOnly={!isEditing}
                        onChange={(updatedText) => {
                          const currentSummary = block.summary ?? [];
                          const updatedSummary = [...currentSummary];
                          updatedSummary[index] = {
                            ...summaryBlock,
                            text: updatedText,
                          };
                          onChange?.({
                            ...block,
                            summary: updatedSummary,
                          });
                        }}
                      />
                    )}
                  </div>
                  {isEditing && (
                    <DeleteButton
                      onDelete={() => onDeleteSummaryBlock(index)}
                      label="Delete summary block"
                    />
                  )}
                </div>
              ))}
              {isEditing && (
                <div className="flex items-center gap-2">
                  <AddButton onAdd={onAddSummaryBlock} label="Summary Block" />
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
