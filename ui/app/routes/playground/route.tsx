import type { RouteHandle } from "react-router";
import { useSearchParams } from "react-router";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { useConfig } from "~/context/config";

import { useVariantSelection, lastViewedFunctionAtom } from "./state";
import { useEffect, useMemo } from "react";
import clsx from "clsx";
import DataSection from "./DataSection/DataSection";
import ResizableQuadrant, {
  HorizontalResizeHandle,
  resizableGrid,
  VerticalResizeHandle,
} from "./ui/ResizableQuadrant";
import type { FunctionConfig } from "tensorzero-node";
import { useAtom } from "jotai";
import { LayoutGroup } from "motion/react";
import FunctionPanel from "./FunctionSection/FunctionPanel";
import { VariantPanel } from "./FunctionSection/VariantPanel";
import { PLAYGROUND_GRID_ROWS } from "./layout";

export const handle: RouteHandle = {
  excludeContentWrapper: true,
  crumb: () => ["Playground"],
};

export default function PlaygroundRoute() {
  /**
   * Prefer browser history for current function state - store function in query param.
   *
   * However, if user opens the Playground directly/in a new tab,
   * default to the mostly recently viewed function.
   */
  const [lastViewedFunction, setLastViewedFunction] = useAtom(
    lastViewedFunctionAtom,
  );

  const [searchParams, setSearchParams] = useSearchParams(
    lastViewedFunction && {
      function: lastViewedFunction,
    },
  );
  const functionName = searchParams.get("function") ?? undefined;

  // TODO Ensure there isn't a re-render loop here - I don't believe so because `lastViewedFunction` is only used as an initializer
  useEffect(() => {
    // On navigation, save in localStorage
    setLastViewedFunction(functionName);
  }, [functionName, setLastViewedFunction]);

  const config = useConfig();
  const functionConfig = functionName
    ? config.functions[functionName]
    : undefined;

  return (
    <main className="bg-bg-tertiary relative flex h-full w-full flex-col gap-2 overflow-y-auto pt-4 pr-4 pl-4">
      <header className="sticky top-0 flex flex-row items-center gap-4">
        <h2 className="text-2xl font-bold">Playground</h2>
        <div className="w-84">
          <FunctionSelector
            selected={functionName ?? null}
            onSelect={(functionName) =>
              setSearchParams({ function: functionName })
            }
            functions={config.functions}
          />
        </div>
      </header>

      {functionName && functionConfig && (
        <PlaygroundContainer
          functionName={functionName}
          functionConfig={functionConfig}
        />
      )}
    </main>
  );
}

/** Full playground UI given selected function */
function PlaygroundContainer({
  functionName,
  functionConfig,
}: {
  functionName: string;
  functionConfig: FunctionConfig;
}) {
  const { selectedVariants, addVariant, removeVariant } = useVariantSelection(
    functionName,
    functionConfig,
  );

  // TODO Abstract this?
  const variantsSectionColumns = useMemo(
    () => selectedVariants.map(() => "minmax(0, 600px)").join(" "),
    [selectedVariants],
  );

  return (
    <ResizableQuadrant extraColumnsTemplate={variantsSectionColumns}>
      {/* "Function section" */}
      <section
        className={clsx(
          "sticky top-0 z-20 col-span-full grid grid-cols-subgrid",
          resizableGrid({ row: "header" }),
        )}
        style={{
          gridTemplateRows: PLAYGROUND_GRID_ROWS,
        }}
      >
        <LayoutGroup>
          <FunctionPanel
            key="function-panel"
            functionConfig={functionConfig}
            functionName={functionName}
          />

          <VerticalResizeHandle key="resize-handle" />

          {selectedVariants.map(([variantName, variant]) => (
            <VariantPanel
              key={`variant-${variantName}`}
              functionName={functionName}
              variantName={variantName}
              variant={variant}
              onClose={() => removeVariant(variantName)}
              onOpenVariant={addVariant}
            />
          ))}
        </LayoutGroup>
      </section>

      <HorizontalResizeHandle className="sticky z-20" />

      <DataSection variants={selectedVariants} functionName={functionName} />
    </ResizableQuadrant>
  );
}
