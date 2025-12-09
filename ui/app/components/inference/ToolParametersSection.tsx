import type {
  Tool,
  ToolChoice,
  ProviderTool,
  FunctionTool,
  OpenAICustomTool,
} from "~/types/tensorzero";
import {
  SnippetLayout,
  SnippetContent,
  SnippetTabs,
} from "~/components/layout/SnippetLayout";
import { CodeEditor } from "~/components/ui/code-editor";
import { Badge } from "~/components/ui/badge";

export interface ToolParametersProps {
  allowedTools?: string[];
  additionalTools?: Tool[];
  toolChoice?: ToolChoice;
  parallelToolCalls?: boolean;
  providerTools: ProviderTool[];
}

export function ToolParametersSection({
  allowedTools,
  additionalTools,
  toolChoice,
  parallelToolCalls,
  providerTools,
}: ToolParametersProps) {
  const hasAllowedTools = allowedTools && allowedTools.length > 0;
  const hasAdditionalTools = additionalTools && additionalTools.length > 0;
  const hasProviderTools = providerTools && providerTools.length > 0;
  const hasToolChoice = toolChoice !== undefined;
  const hasParallelToolCalls = parallelToolCalls !== undefined;

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
                  <ToolChoiceBadge toolChoice={toolChoice} />
                </div>
              )}
              {hasParallelToolCalls && (
                <div className="flex flex-col gap-1">
                  <span className="text-fg-muted text-xs font-medium uppercase">
                    Parallel Tool Calls
                  </span>
                  <Badge variant={parallelToolCalls ? "default" : "secondary"}>
                    {parallelToolCalls ? "enabled" : "disabled"}
                  </Badge>
                </div>
              )}
            </div>
          )}

          {/* Allowed Tools */}
          {hasAllowedTools && (
            <div className="flex flex-col gap-2">
              <span className="text-fg-muted text-xs font-medium uppercase">
                Allowed Tools ({allowedTools.length})
              </span>
              <div className="flex flex-wrap gap-2">
                {allowedTools.map((toolName) => (
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
                Additional Tools ({additionalTools.length})
              </span>
              <AdditionalToolsList tools={additionalTools} />
            </div>
          )}

          {/* Provider Tools */}
          {hasProviderTools && (
            <div className="flex flex-col gap-2">
              <span className="text-fg-muted text-xs font-medium uppercase">
                Provider Tools ({providerTools.length})
              </span>
              <ProviderToolsList tools={providerTools} />
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

function AdditionalToolsList({ tools }: { tools: Tool[] }) {
  if (tools.length === 1) {
    return <ToolCard tool={tools[0]} />;
  }

  const tabs = tools.map((tool, index) => ({
    id: String(index),
    label: getToolName(tool),
    content: <ToolCard tool={tool} />,
  }));

  return <SnippetTabs tabs={tabs} />;
}

function getToolName(tool: Tool): string {
  if (tool.type === "function") {
    return tool.name;
  } else if (tool.type === "openai_custom") {
    return tool.name;
  }
  return "Unknown Tool";
}

function ToolCard({ tool }: { tool: Tool }) {
  if (tool.type === "function") {
    return <FunctionToolCard tool={tool} />;
  } else if (tool.type === "openai_custom") {
    return <OpenAICustomToolCard tool={tool} />;
  }
  return null;
}

type FunctionToolWithType = { type: "function" } & FunctionTool;
type OpenAICustomToolWithType = { type: "openai_custom" } & OpenAICustomTool;

function FunctionToolCard({ tool }: { tool: FunctionToolWithType }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">function</Badge>
        <span className="font-mono text-sm font-medium">{tool.name}</span>
        {tool.strict && (
          <Badge variant="outline" className="text-xs">
            strict
          </Badge>
        )}
      </div>
      {tool.description && (
        <p className="text-fg-muted text-sm">{tool.description}</p>
      )}
      <div className="mt-1">
        <span className="text-fg-muted mb-1 block text-xs font-medium uppercase">
          Parameters
        </span>
        <CodeEditor
          allowedLanguages={["json"]}
          value={JSON.stringify(tool.parameters, null, 2)}
          readOnly
        />
      </div>
    </div>
  );
}

function OpenAICustomToolCard({ tool }: { tool: OpenAICustomToolWithType }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">openai_custom</Badge>
        <span className="font-mono text-sm font-medium">{tool.name}</span>
      </div>
      {tool.description && (
        <p className="text-fg-muted text-sm">{tool.description}</p>
      )}
      {tool.format && (
        <div className="mt-1">
          <span className="text-fg-muted mb-1 block text-xs font-medium uppercase">
            Format
          </span>
          <CodeEditor
            allowedLanguages={["json"]}
            value={JSON.stringify(tool.format, null, 2)}
            readOnly
          />
        </div>
      )}
    </div>
  );
}

function ProviderToolsList({ tools }: { tools: ProviderTool[] }) {
  if (tools.length === 1) {
    return <ProviderToolCard tool={tools[0]} />;
  }

  const tabs = tools.map((tool, index) => ({
    id: String(index),
    label: getProviderToolLabel(tool, index),
    content: <ProviderToolCard tool={tool} />,
  }));

  return <SnippetTabs tabs={tabs} />;
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
