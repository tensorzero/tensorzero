import { useState, useMemo, useCallback } from "react";
import type {
  Tool,
  ToolChoice,
  ProviderTool,
  DynamicToolParams,
} from "~/types/tensorzero";
import {
  SnippetLayout,
  SnippetContent,
} from "~/components/layout/SnippetLayout";
import { CodeEditor } from "~/components/ui/code-editor";
import { Badge } from "~/components/ui/badge";
import { Combobox } from "~/components/ui/combobox";

export function ToolParametersSection({
  allowed_tools,
  additional_tools,
  tool_choice,
  parallel_tool_calls,
  provider_tools,
}: DynamicToolParams) {
  const hasAllowedTools = allowed_tools && allowed_tools.length > 0;
  const hasAdditionalTools = additional_tools && additional_tools.length > 0;
  const hasProviderTools = provider_tools && provider_tools.length > 0;
  const hasToolChoice = tool_choice !== undefined;
  const hasParallelToolCalls = parallel_tool_calls !== undefined;

  const hasAnyToolParameters =
    hasAllowedTools ||
    hasAdditionalTools ||
    hasProviderTools ||
    hasToolChoice ||
    hasParallelToolCalls;

  if (!hasAnyToolParameters) {
    return (
      <SnippetLayout>
        <SnippetContent>
          <div className="text-fg-muted text-sm">
            No tool parameters configured
          </div>
        </SnippetContent>
      </SnippetLayout>
    );
  }

  return (
    <SnippetLayout>
      <SnippetContent>
        <div className="flex flex-col gap-4">
          {/* Tool Choice and Parallel Tool Calls */}
          {(hasToolChoice || hasParallelToolCalls) && (
            <div className="flex flex-wrap gap-4">
              {hasToolChoice && (
                <div className="flex flex-col gap-1">
                  <span className="text-fg-muted text-xs font-medium uppercase">
                    Tool Choice
                  </span>
                  <ToolChoiceBadge toolChoice={tool_choice} />
                </div>
              )}
              {hasParallelToolCalls && (
                <div className="flex flex-col gap-1">
                  <span className="text-fg-muted text-xs font-medium uppercase">
                    Parallel Tool Calls
                  </span>
                  <Badge
                    variant={parallel_tool_calls ? "default" : "secondary"}
                  >
                    {parallel_tool_calls ? "enabled" : "disabled"}
                  </Badge>
                </div>
              )}
            </div>
          )}

          {/* Allowed Tools */}
          {hasAllowedTools && (
            <div className="flex flex-col gap-2">
              <span className="text-fg-muted text-xs font-medium uppercase">
                Allowed Tools ({allowed_tools.length})
              </span>
              <div className="flex flex-wrap gap-2">
                {allowed_tools.map((toolName) => (
                  <Badge key={toolName} variant="outline">
                    {toolName}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Additional Tools */}
          {hasAdditionalTools && (
            <div className="flex flex-col gap-2">
              <span className="text-fg-muted text-xs font-medium uppercase">
                Additional Tools ({additional_tools.length})
              </span>
              <ToolsList
                tools={additional_tools}
                getLabel={(tool) => getToolName(tool)}
                renderCard={(tool) => <ToolCard tool={tool} />}
              />
            </div>
          )}

          {/* Provider Tools */}
          {hasProviderTools && (
            <div className="flex flex-col gap-2">
              <span className="text-fg-muted text-xs font-medium uppercase">
                Provider Tools ({provider_tools.length})
              </span>
              <ToolsList
                tools={provider_tools}
                getLabel={getProviderToolLabel}
                renderCard={(tool) => <ProviderToolCard tool={tool} />}
              />
            </div>
          )}
        </div>
      </SnippetContent>
    </SnippetLayout>
  );
}

function ToolChoiceBadge({ toolChoice }: { toolChoice: ToolChoice }) {
  if (typeof toolChoice === "string") {
    return <Badge variant="outline">{toolChoice}</Badge>;
  }

  // toolChoice is { specific: string }
  return (
    <Badge variant="outline">
      specific: <span className="font-mono">{toolChoice.specific}</span>
    </Badge>
  );
}

interface ToolsListProps<T> {
  tools: T[];
  getLabel: (tool: T, index: number) => string;
  renderCard: (tool: T) => React.ReactNode;
}

function ToolsList<T>({ tools, getLabel, renderCard }: ToolsListProps<T>) {
  const [selectedIndex, setSelectedIndex] = useState<number>(0);

  const items = useMemo(() => {
    // Track label counts to detect duplicates
    const labelCounts = new Map<string, number>();
    tools.forEach((tool, index) => {
      const label = getLabel(tool, index);
      labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
    });

    // Build unique display labels, appending index only for duplicates
    const seenLabels = new Map<string, number>();
    return tools.map((tool, index) => {
      const baseLabel = getLabel(tool, index);
      const count = labelCounts.get(baseLabel) || 1;

      if (count > 1) {
        const occurrence = (seenLabels.get(baseLabel) || 0) + 1;
        seenLabels.set(baseLabel, occurrence);
        return `${baseLabel} (${occurrence})`;
      }
      return baseLabel;
    });
  }, [tools, getLabel]);

  const handleSelect = useCallback(
    (value: string) => {
      const index = items.indexOf(value);
      if (index !== -1) {
        setSelectedIndex(index);
      }
    },
    [items],
  );

  const selectedTool = tools[selectedIndex];

  return (
    <div className="flex flex-col gap-3">
      <Combobox
        selected={items[selectedIndex]}
        onSelect={handleSelect}
        items={items}
        placeholder="Select tool"
        emptyMessage="No tools found"
      />
      {selectedTool && renderCard(selectedTool)}
    </div>
  );
}

function getToolName(tool: Tool): string {
  return tool.name;
}

function ToolCard({ tool }: { tool: Tool }) {
  const isFunctionTool = tool.type === "function";
  const schemaData = isFunctionTool ? tool.parameters : tool.format;
  const schemaLabel = isFunctionTool ? "Parameters" : "Format";

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {tool.description && (
          <span className="text-fg-muted text-sm">{tool.description}</span>
        )}
        <Badge variant="secondary">{tool.type}</Badge>
        {isFunctionTool && tool.strict && (
          <Badge variant="outline" className="text-xs">
            strict
          </Badge>
        )}
      </div>
      {schemaData && (
        <div className="mt-1">
          <span className="text-fg-muted mb-1 block text-xs font-medium uppercase">
            {schemaLabel}
          </span>
          <CodeEditor
            allowedLanguages={["json"]}
            value={JSON.stringify(schemaData, null, 2)}
            readOnly
          />
        </div>
      )}
    </div>
  );
}

function getProviderToolLabel(tool: ProviderTool, index: number): string {
  // Try to extract a name from the tool JSON if it has one
  if (
    tool.tool &&
    typeof tool.tool === "object" &&
    "type" in tool.tool &&
    typeof tool.tool.type === "string"
  ) {
    return tool.tool.type;
  }
  return `Tool ${index + 1}`;
}

function ProviderToolCard({ tool }: { tool: ProviderTool }) {
  const scopeLabel = tool.scope
    ? `${tool.scope.model_name}${tool.scope.provider_name ? ` (${tool.scope.provider_name})` : ""}`
    : "All providers";

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">provider</Badge>
        <span className="text-fg-muted text-sm">Scope: {scopeLabel}</span>
      </div>
      <div className="mt-1">
        <span className="text-fg-muted mb-1 block text-xs font-medium uppercase">
          Configuration
        </span>
        <CodeEditor
          allowedLanguages={["json"]}
          value={JSON.stringify(tool.tool, null, 2)}
          readOnly
        />
      </div>
    </div>
  );
}
