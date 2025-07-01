import { useForm } from "react-hook-form";
import type { RouteHandle } from "react-router";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useConfig } from "~/context/config";
import { Form } from "~/components/ui/form";
import { Button } from "~/components/ui/button";
import { XIcon } from "lucide-react";
import type { VariantConfig } from "~/utils/config/variant";
// import { ModelTag } from "@lobehub/icons";

// TODO Replace with Dnd kit - recommended by motion for advanced use cases
import { LayoutGroup } from "motion/react";
import { useAtom, useAtomValue } from "jotai";
import { VariantTable } from "./VariantTable";
import { CodeEditor, JsonEditor } from "~/components/ui/code-editor";
import {
  datapointsAtom,
  playgroundLayoutAtom,
  showAssistantPromptAtom,
  showSystemPromptAtom,
  showUserPromptAtom,
} from "./state";
import VariantBasicInfo from "../observability/functions/$function_name/variants/VariantBasicInfo";
import type { FunctionConfig } from "~/utils/config/function";
import { useState } from "react";
import clsx from "clsx";
import RunButton from "./ui/RunButton";
import { GRID_TEMPLATE_ROWS, GridRow, setRow } from "./grid";
import AnimatedCollapsible from "./ui/AnimatedCollapsible";
import DatapointRow from "./DatapointRow";

// TODO Fix title by moving it to root (!)
export const handle: RouteHandle = {
  hideBreadcrumbs: true,
  crumb: () => ["Playground"],
};

export default function Playground() {
  const config = useConfig();
  const form = useForm<{ function: string }>({
    defaultValues: {
      function: "tensorzero::llm_judge::entity_extraction::count_sports",
    },
  });

  const functionName = form.watch("function");

  const functionConfig = config.functions[functionName];
  const variantEntries = Object.entries(functionConfig.variants);

  const [playgroundLayout, setPlaygroundLayout] = useAtom(playgroundLayoutAtom);
  // TODO Should filter this by variants that actually exist for this function
  const selectedVariants =
    playgroundLayout?.functions[functionName]?.selectedVariants ?? [];
  const unselectedVariants = variantEntries.filter(
    ([name]) => !selectedVariants.includes(name),
  );

  const setVariantOrder = (selectedVariants: string[]) => {
    setPlaygroundLayout((layout) => ({
      ...layout,
      functions: {
        ...layout.functions,
        [functionName]: {
          selectedVariants: selectedVariants,

          // TODO what should the default be?
          showSystemPrompt: false,
          showAssistantPrompt: false,
          showUserPrompt: false,
          showOutputSchema: false,
        },
      },
    }));
  };

  const canSelectAnother =
    selectedVariants.length < 3 && unselectedVariants.length > 0;
  const columnCount = 1 + selectedVariants.length + (canSelectAnother ? 1 : 0);

  const datapoints = useAtomValue(datapointsAtom);

  return (
    <Form {...form}>
      <main
        className="bg-bg-tertiary grid h-full w-full gap-x-4 gap-y-8 pt-4 pr-4"
        style={{
          gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))`,
          gridTemplateRows: GRID_TEMPLATE_ROWS,
        }}
      >
        <LayoutGroup>
          <FunctionPanel functionConfig={functionConfig}>
            {/* TODO Fix this? */}
            <FunctionSelector
              control={form.control}
              name="function"
              inferenceCount={0}
              config={config}
            />
          </FunctionPanel>

          {selectedVariants.map((variantName) => (
            <VariantPanel
              variantName={variantName}
              variant={functionConfig.variants[variantName]}
              onClose={() =>
                setVariantOrder(
                  selectedVariants.filter((v) => v !== variantName),
                )
              }
              key={variantName}
            />
          ))}

          {canSelectAnother && (
            <SelectVariantsPanel
              variants={unselectedVariants}
              onSelect={(variant) =>
                setVariantOrder([...selectedVariants, variant])
              }
            />
          )}

          <section
            className={clsx(
              // "col-span-full grid grid-cols-subgrid gap-y-2",
              "col-span-full grid grid-cols-subgrid grid-rows-subgrid",
              setRow(GridRow.Datapoints),
            )}
          >
            {datapoints.map((datapointId) => {
              const variants = selectedVariants.map<[string, VariantConfig]>(
                (name) => [name, functionConfig.variants[name]],
              );
              return (
                <DatapointRow
                  key={datapointId}
                  datapointId={datapointId}
                  variants={variants}
                />
              );
            })}
          </section>
        </LayoutGroup>
      </main>
    </Form>
  );
}

const VariantPanel: React.FC<{
  variantName: string;
  variant: VariantConfig;
  onClose?: () => void;
}> = ({ variant, variantName, onClose }) => {
  const systemTemplate =
    variant.type === "chat_completion" && variant.system_template?.content;
  const assistantTemplate =
    variant.type === "chat_completion" && variant.assistant_template?.content;
  const userTemplate =
    variant.type === "chat_completion" && variant.user_template?.content;

  // TODO Replace these with playground layout so they're saved on a per-function basis?
  const [showSystemPrompt, setShowSystemPrompt] = useAtom(showSystemPromptAtom);
  const [showAssistantPrompt, setShowAssistantPrompt] = useAtom(
    showAssistantPromptAtom,
  );
  const [showUserPrompt, setShowUserPrompt] = useAtom(showUserPromptAtom);

  return (
    <div className="row-start-1 -row-end-2 grid grid-cols-1 grid-rows-subgrid gap-y-1 p-4">
      <header
        className={clsx(
          "flex flex-row items-center justify-between",
          setRow(GridRow.Header),
        )}
      >
        <h2 className="font-mono text-base font-normal">{variantName}</h2>

        {onClose && (
          <Button
            // TODO Figure out this styling
            className="hover:bg-bg-hover h-6 w-6 bg-none"
            variant="secondary"
            size="icon"
            onClick={onClose}
          >
            <XIcon className="h-3 w-3" />
          </Button>
        )}
      </header>

      {/* TODO FIX THE VITE ISSUES */}
      {/* {model && <ModelTag model={model} />} */}

      <section className={setRow(GridRow.Metadata)}>
        <VariantBasicInfo
          function_name={"TODO remove this"}
          function_type={"chat"} // TODO remove
          variantConfig={variant}
        />

        <h5 className="mt-4 text-sm font-bold">Input</h5>
      </section>

      <section className={setRow(GridRow.SystemPrompt)}>
        {systemTemplate && (
          <AnimatedCollapsible
            label="System"
            isOpen={showSystemPrompt}
            onOpenChange={setShowSystemPrompt}
          >
            <CodeEditor
              readOnly={true}
              value={systemTemplate}
              allowedLanguages={["jinja2"]}
              className="border bg-white p-2"
            />
          </AnimatedCollapsible>
        )}
      </section>

      <section className={setRow(GridRow.AssistantPrompt)}>
        {/* TODO Show these only if *any* variant has a template defined */}
        {assistantTemplate && (
          <AnimatedCollapsible
            label="Assistant"
            isOpen={showAssistantPrompt}
            onOpenChange={setShowAssistantPrompt}
          >
            <CodeEditor
              readOnly={true}
              value={assistantTemplate}
              allowedLanguages={["jinja2"]}
              className="border bg-white p-2"
            />
          </AnimatedCollapsible>
        )}
      </section>

      <section className={setRow(GridRow.UserPrompt)}>
        {userTemplate && (
          <AnimatedCollapsible
            label="User"
            isOpen={showUserPrompt}
            onOpenChange={setShowUserPrompt}
          >
            <CodeEditor
              readOnly={true}
              value={userTemplate}
              allowedLanguages={["jinja2"]}
              className="border bg-white p-2"
            />
          </AnimatedCollapsible>
        )}
      </section>

      <footer className={clsx("justify-self-end", setRow(GridRow.Footer))}>
        <RunButton />
      </footer>
    </div>
  );
};

const FunctionPanel: React.FC<
  React.PropsWithChildren<{
    functionConfig: FunctionConfig;
  }>
> = ({ functionConfig, children }) => {
  // TODO ...
  const [showSystemPrompt, setShowSystemPrompt] = useAtom(showSystemPromptAtom);
  const [showAssistantPrompt, setShowAssistantPrompt] = useAtom(
    showAssistantPromptAtom,
  );
  const [showUserPrompt, setShowUserPrompt] = useAtom(showUserPromptAtom);

  const systemSchema = functionConfig.system_schema?.content;
  const assistantSchema = functionConfig.assistant_schema?.content;
  const userSchema = functionConfig.user_schema?.content;
  const outputSchema =
    functionConfig.type === "json" && functionConfig.output_schema?.content;

  // TODO Move this to playground layout state
  const [showOutputSchema, setShowOutputSchema] = useState(false);

  return (
    // TODO p-4 here screwed things up?
    <div className="relative z-10 row-start-1 -row-end-2 grid grid-cols-1 grid-rows-subgrid gap-y-1 rounded-2xl bg-white p-4 shadow-md">
      {/* TODO Move function selection here? */}
      <header className={setRow(GridRow.Header)}>{children}</header>

      <section className={setRow(GridRow.Metadata)}>
        <h5 className="mt-4 text-sm font-bold">Input</h5>
      </section>

      <section className={setRow(GridRow.SystemPrompt)}>
        <AnimatedCollapsible
          label="System"
          isOpen={showSystemPrompt}
          onOpenChange={setShowSystemPrompt}
        >
          {systemSchema ? (
            <JsonEditor value={systemSchema} className="border bg-white p-2" />
          ) : (
            <span className="text-muted-foreground text-xs">No schema</span>
          )}
        </AnimatedCollapsible>
      </section>

      <section className={setRow(GridRow.AssistantPrompt)}>
        <AnimatedCollapsible
          label="Assistant"
          isOpen={showAssistantPrompt}
          onOpenChange={setShowAssistantPrompt}
        >
          {assistantSchema ? (
            <JsonEditor
              value={assistantSchema}
              className="border bg-white p-2"
            />
          ) : (
            <span className="text-muted-foreground text-xs">No schema</span>
          )}
        </AnimatedCollapsible>
      </section>

      <section className={setRow(GridRow.UserPrompt)}>
        <AnimatedCollapsible
          label="User"
          isOpen={showUserPrompt}
          onOpenChange={setShowUserPrompt}
        >
          {userSchema ? (
            <JsonEditor value={userSchema} className="border bg-white p-2" />
          ) : (
            <span className="text-muted-foreground text-xs">No schema</span>
          )}
        </AnimatedCollapsible>
      </section>

      <section className={setRow(GridRow.OutputSchema)}>
        <h5 className="mt-4 text-sm font-bold">Structured output</h5>
        {outputSchema && (
          <AnimatedCollapsible
            isOpen={showOutputSchema}
            onOpenChange={setShowOutputSchema}
            label="Schema"
          >
            <JsonEditor value={outputSchema} className="border bg-white p-2" />
          </AnimatedCollapsible>
        )}
      </section>

      {/* TODO Add tools here, too! */}

      <footer className={clsx("justify-self-end", setRow(GridRow.Footer))}>
        <RunButton>Run all</RunButton>
      </footer>
    </div>
  );
};

const SelectVariantsPanel: React.FC<{
  variants: [string, VariantConfig][];
  onSelect: (variant: string) => void;
}> = ({ variants, onSelect }) => {
  return (
    <div className="row-start-1 -row-end-2 grid grid-cols-1 grid-rows-subgrid gap-y-4 rounded-xl bg-transparent">
      <header
        className={clsx(
          "flex items-center justify-start",
          setRow(GridRow.Header),
        )}
      >
        <h4 className="text-lg font-normal">Compare a variant</h4>
      </header>

      <section className="row-start-2 -row-end-2">
        <VariantTable
          variants={variants}
          onVariantSelect={(variantName) => {
            if (variantName) {
              onSelect(variantName);
            }
          }}
        />
      </section>
    </div>
  );
};
