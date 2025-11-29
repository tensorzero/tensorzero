import { useEffect, useState } from "react";
import type { JsonInferenceOutput } from "~/types/tensorzero";
import { EmptyMessage } from "./ContentBlockElement";
import { AddButton } from "~/components/ui/AddButton";
import {
  SnippetContent,
  SnippetTabs,
  type SnippetTab,
} from "~/components/layout/SnippetLayout";
import { CodeEditor } from "../ui/code-editor";
import { MessageWrapper } from "./MessageWrapper";
import { DeleteButton } from "~/components/ui/DeleteButton";

interface JsonOutputElementProps {
  output?: JsonInferenceOutput;
  // TODO: add output schema
  isEditing?: boolean;
  onOutputChange?: (output?: JsonInferenceOutput) => void;
  maxHeight?: number | "Content";
}

export function JsonOutputElement({
  output,
  isEditing,
  onOutputChange,
  maxHeight,
}: JsonOutputElementProps) {
  const [displayValue, setDisplayValue] = useState<string | undefined>(
    output?.raw ?? undefined,
  );
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string | undefined>();
  const [hasEdited, setHasEdited] = useState<boolean>(false);

  useEffect(() => {
    // Update display value when output.raw changes externally
    setDisplayValue(output?.raw ?? undefined);
  }, [output?.raw]);

  useEffect(() => {
    // Switch to raw tab when entering edit mode
    if (isEditing) {
      setActiveTab("raw");
    }
  }, [isEditing]);

  const handleRawChange = (value: string) => {
    if (onOutputChange) {
      setHasEdited(true);
      setDisplayValue(value);

      try {
        // Attempt to parse the JSON to validate it
        const parsedValue = JSON.parse(value);
        setJsonError(null);
        onOutputChange({
          ...output,
          raw: value,
          parsed: parsedValue,
        });
      } catch {
        setJsonError("Invalid JSON format");
        onOutputChange({
          ...output,
          raw: value,
          parsed: null,
        });
      }
    }
  };

  const onAddOutput = () => {
    onOutputChange?.({
      parsed: {},
      raw: "{}",
    });
  };

  const onDeleteOutput = () => {
    onOutputChange?.(undefined);
  };

  if (output === undefined) {
    return (
      <div className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4">
        {isEditing ? (
          <AddOutputButtons onAdd={onAddOutput} />
        ) : (
          <EmptyMessage message="No output" />
        )}
      </div>
    );
  }

  const tabs: SnippetTab[] = [
    {
      id: "parsed",
      label: "Parsed Output",
      indicator: output.parsed ? (hasEdited ? "empty" : "content") : "fail",
    },
    {
      id: "raw",
      label: "Raw Output",
    },
  ];

  // Set default tab to "Raw" when editing, otherwise "Parsed" if it has content
  const defaultTab = isEditing ? "raw" : output.parsed ? "parsed" : "raw";

  const renderCurrentTab = (currentTab: string) => {
    switch (currentTab) {
      case "parsed":
        if (output.parsed) {
          return (
            <CodeEditor
              allowedLanguages={["json"]}
              value={JSON.stringify(output.parsed, null, 2)}
              readOnly
            />
          );
        } else {
          return <EmptyMessage message="Output failed to parse" />;
        }
      case "raw":
        return (
          <>
            <CodeEditor
              allowedLanguages={["json"]}
              value={isEditing ? displayValue : (output.raw ?? undefined)}
              readOnly={!isEditing}
              onChange={isEditing ? handleRawChange : undefined}
            />
            {isEditing && jsonError && (
              <div className="text-xs text-red-600">{jsonError}</div>
            )}
          </>
        );
      // TODO: render schema tab
      // case "schema":
      //   return <CodeEditor allowedLanguages={["json"]} value={"{}"} readOnly />;
      default:
        // This should never happen
        return <EmptyMessage message="Error" />;
    }
  };

  return (
    <div className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4">
      <MessageWrapper
        role="assistant"
        actionBar={
          isEditing && (
            <DeleteButton onDelete={onDeleteOutput} label="Delete output" />
          )
        }
      >
        <SnippetTabs
          tabs={tabs}
          defaultTab={defaultTab}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        >
          {(currentTab) => (
            <SnippetContent maxHeight={maxHeight}>
              {renderCurrentTab(currentTab)}
            </SnippetContent>
          )}
        </SnippetTabs>
      </MessageWrapper>
    </div>
  );
}

function AddOutputButtons({ onAdd: onAdd }: { onAdd: () => void }) {
  return (
    <div className="flex items-center gap-2 py-2">
      <AddButton label="Output" onAdd={onAdd} />
    </div>
  );
}
