import { VariantFilter } from "~/components/function/variant/variant-filter";
import type { PlaygroundVariantInfo } from "./utils";

export const BuiltinVariantFilter = ({
  variants,
  updateSearchParams,
  builtInVariantNames,
  disabled,
}: {
  variants: PlaygroundVariantInfo[];
  updateSearchParams: (params: { variants: string }) => void;
  builtInVariantNames: string[];
  disabled: boolean;
}) => {
  const builtinVariants = variants.filter((v) => v.type === "builtin");
  const otherVariants = variants.filter((v) => v.type !== "builtin");

  const handleChange = (
    selectedNamesOrUpdater: React.SetStateAction<string[]>,
  ) => {
    const currentBuiltinNames = builtinVariants.map((v) => v.name);
    const selectedNames =
      typeof selectedNamesOrUpdater === "function"
        ? selectedNamesOrUpdater(currentBuiltinNames)
        : selectedNamesOrUpdater;

    const newVariants: PlaygroundVariantInfo[] = [
      ...otherVariants,
      ...selectedNames.map((name) => ({ type: "builtin" as const, name })),
    ];
    updateSearchParams({ variants: JSON.stringify(newVariants) });
  };
  const variantData = builtInVariantNames.map((name) => ({
    name,
    color: undefined,
  }));

  return (
    <VariantFilter
      disabled={disabled}
      variants={variantData}
      selectedValues={builtinVariants.map((v) => v.name)}
      setSelectedValues={handleChange}
    />
  );
};
