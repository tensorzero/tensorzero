import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import type { FunctionConfig, JSONSchema } from "~/utils/config/function";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "~/components/ui/collapsible";
import { ChevronDown } from "lucide-react";
import { Code } from "~/components/ui/code";

interface BasicFunctionInfoProps {
  functionConfig: FunctionConfig;
}

interface SchemaFieldProps {
  title: string;
  content?: JSONSchema;
}

function SchemaField({ title, content }: SchemaFieldProps) {
  if (!content) {
    return (
      <div className="col-span-2">
        <dt className="text-lg font-semibold">{title}</dt>
        <dd className="text-sm text-muted-foreground">None</dd>
      </div>
    );
  }

  return (
    <div className="col-span-2">
      <dt className="text-lg font-semibold">{title}</dt>
      <dd>
        <Collapsible>
          <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
            <ChevronDown className="h-4 w-4" />
            Show full schema
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="max-h-[500px] overflow-auto rounded-md bg-muted p-4">
              <code className="text-sm">
                {typeof content === "object"
                  ? JSON.stringify(content, null, 2)
                  : content}
              </code>
            </pre>
          </CollapsibleContent>
        </Collapsible>
      </dd>
    </div>
  );
}

export default function BasicFunctionInfo({
  functionConfig,
}: BasicFunctionInfoProps) {
  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle className="text-xl">Basic Information</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-4">
          {/* Type */}
          <div>
            <dt className="text-lg font-semibold">Type</dt>
            <dd>
              <Code>{functionConfig.type}</Code>
            </dd>
          </div>

          {/* Fields specific to chat type */}
          {functionConfig.type === "chat" && (
            <>
              <div>
                <dt className="text-lg font-semibold">Tools</dt>
                <dd>
                  {functionConfig.tools?.length
                    ? functionConfig.tools.join(", ")
                    : "—"}
                </dd>
              </div>
              <div>
                <dt className="text-lg font-semibold">Tool Choice</dt>
                <dd>
                  <Code>{functionConfig.tool_choice}</Code>
                </dd>
              </div>
              <div>
                <dt className="text-lg font-semibold">Parallel Tool Calls</dt>
                <dd>
                  <Code>
                    {functionConfig.parallel_tool_calls ? "true" : "false"}
                  </Code>
                </dd>
              </div>
            </>
          )}

          {/* Optional schemas */}
          <SchemaField
            title="System Schema"
            content={functionConfig.system_schema?.content}
          />
          <SchemaField
            title="User Schema"
            content={functionConfig.user_schema?.content}
          />
          <SchemaField
            title="Assistant Schema"
            content={functionConfig.assistant_schema?.content}
          />
          {functionConfig.type === "json" && (
            <SchemaField
              title="Output Schema"
              content={functionConfig.output_schema?.content}
            />
          )}
        </dl>
      </CardContent>
    </Card>
  );
}
