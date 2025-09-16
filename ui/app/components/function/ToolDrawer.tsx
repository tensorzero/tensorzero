import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerFooter,
  DrawerHeader,
} from "~/components/ui/drawer";
import { Button } from "../ui/button";
import { useConfig } from "~/context/config";
import { useEffect, useState } from "react";
import type { StaticToolConfig } from "tensorzero-node";
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

interface ToolDrawerProps {
  selectedTool: string | null;
  onClose?: () => void;
}

export function ToolDrawer({ selectedTool, onClose }: ToolDrawerProps) {
  const { tools } = useConfig();

  const [tool, setTool] = useState<StaticToolConfig>(Object.values(tools)[0]!);

  // We update the tool using an effect to ensure that there
  // is never an edge case where the contents of the <Drawer>
  // are empty. We don't want the content to disappear when it
  // animates away
  useEffect(() => {
    if (!selectedTool) return;

    const tool = tools[selectedTool];
    if (!tool)
      throw new Error(`"${selectedTool}" is not present in config.tools`);
    console.log({ tool });
    setTool(tool);
  }, [tools, selectedTool]);

  return (
    <Drawer open={selectedTool !== null} onClose={onClose}>
      <DrawerContent>
        <DrawerHeader>
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
        </DrawerHeader>
        <DrawerFooter>
          <DrawerClose>
            <Button>Close</Button>
          </DrawerClose>
        </DrawerFooter>
      </DrawerContent>
    </Drawer>
  );
}
