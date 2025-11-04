import { Sheet, SheetContent } from "~/components/ui/sheet";
import { useConfig } from "~/context/config";
import { useCallback, useEffect, useState } from "react";
import type { StaticToolConfig } from "~/types/tensorzero";
import { CodeEditor } from "../ui/code-editor";
import {
  PageHeader,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "../layout/PageLayout";
import {
  BasicInfoItem,
  BasicInfoItemContent,
  BasicInfoItemTitle,
  BasicInfoLayout,
} from "../layout/BasicInfoLayout";

interface ToolDetailsProps {
  toolName: string | null;
  onClose?: () => void;
}

export function ToolDetails({ toolName, onClose }: ToolDetailsProps) {
  const { tools } = useConfig();

  const [tool, setTool] = useState<StaticToolConfig | undefined>(
    Object.values(tools)[0],
  );
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!toolName) return;

    const tool = tools[toolName];
    if (!tool) throw new Error(`"${toolName}" is not present in config.tools`);

    setTool(tool);
    setOpen(true);
  }, [tools, toolName]);

  const onOpenChange = useCallback(
    (v: boolean) => {
      setOpen(v);
      if (!v) onClose?.();
    },
    [onClose],
  );

  if (!tool) return null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent>
        <PageHeader name={tool.name} label={"Tool Information"}>
          <BasicInfoLayout>
            <BasicInfoItem>
              <BasicInfoItemTitle>Description</BasicInfoItemTitle>
              <BasicInfoItemContent wrap>
                <span className="text-sm md:px-2">{tool.description}</span>
              </BasicInfoItemContent>
            </BasicInfoItem>
            <BasicInfoItem>
              <BasicInfoItemTitle>Strict</BasicInfoItemTitle>
              <BasicInfoItemContent wrap>
                <span className="text-sm md:px-2">
                  {tool.strict ? "True" : "False"}
                </span>
              </BasicInfoItemContent>
            </BasicInfoItem>
          </BasicInfoLayout>
        </PageHeader>
        <SectionsGroup>
          <SectionLayout>
            <SectionHeader heading="Schema" />
            <CodeEditor
              allowedLanguages={["json"]}
              value={JSON.stringify(tool.parameters, null, 2)}
              readOnly
            />
          </SectionLayout>
        </SectionsGroup>
      </SheetContent>
    </Sheet>
  );
}
