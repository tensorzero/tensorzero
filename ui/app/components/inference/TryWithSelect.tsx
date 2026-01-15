import { ButtonSelect } from "~/components/ui/select/ButtonSelect";
import { ButtonIcon } from "~/components/ui/button";
import { Compare } from "~/components/icons/Icons";
import { useReadOnly } from "~/context/read-only";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

export interface TryWithSelectProps {
  options: string[];
  onSelect: (option: string) => void;
  isLoading: boolean;
  isDefaultFunction?: boolean;
}

export function TryWithSelect({
  options,
  onSelect,
  isLoading,
  isDefaultFunction,
}: TryWithSelectProps) {
  const isReadOnly = useReadOnly();
  const isDisabled = isLoading || isReadOnly;

  const label = isDefaultFunction ? "model" : "variant";

  const handleSelect = (item: string) => {
    onSelect(item);
  };

  return (
    <ReadOnlyGuard asChild>
      <ButtonSelect
        items={options}
        onSelect={handleSelect}
        trigger={
          <>
            <ButtonIcon as={Compare} variant="tertiary" />
            Try with {label}
          </>
        }
        placeholder={`Search ${label}s...`}
        emptyMessage={`No ${label}s found`}
        disabled={isDisabled}
        menuClassName="min-w-[32rem]"
      />
    </ReadOnlyGuard>
  );
}
