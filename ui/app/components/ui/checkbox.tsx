import * as React from "react";
import { Checkbox as CheckboxPrimitive } from "radix-ui";
import { CheckIcon } from "lucide-react";
import { cn } from "~/utils/common";

function CheckboxRoot({
  as: Comp,
  className,
  ...props
}: {
  as: React.ElementType;
  className?: string;
  children?: React.ReactNode;
}) {
  return (
    <Comp
      className={cn(
        "border-input dark:bg-input/30 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground dark:data-[state=checked]:bg-primary data-[state=checked]:border-primary focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive shadow-xs peer size-4 shrink-0 rounded-[4px] border outline-none transition-shadow focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}

function CheckboxIndicator({
  as: Comp,
  className,
  ...props
}: {
  as: React.ElementType;
  className?: string;
  children?: React.ReactNode;
}) {
  return (
    <Comp
      className={cn(
        "flex items-center justify-center text-current transition-none data-[state=unchecked]:opacity-0",
        className,
      )}
      {...props}
    />
  );
}

function Checkbox(props: React.ComponentProps<typeof CheckboxPrimitive.Root>) {
  return (
    <CheckboxRoot as={CheckboxPrimitive.Root} {...props}>
      <CheckboxIndicator as={CheckboxPrimitive.Indicator}>
        <CheckIcon className="size-3.5" />
      </CheckboxIndicator>
    </CheckboxRoot>
  );
}

function DummyCheckbox({
  className,
  checked,
  ...props
}: React.ComponentProps<"span"> & { checked?: boolean }) {
  return (
    <CheckboxRoot
      as="span"
      aria-hidden
      data-state={checked ? "checked" : "unchecked"}
      className={cn(
        "border-input dark:bg-input/30 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground dark:data-[state=checked]:bg-primary data-[state=checked]:border-primary focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive shadow-xs peer size-4 shrink-0 rounded-[4px] border outline-none transition-shadow focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    >
      <CheckboxIndicator
        as="span"
        data-state={checked ? "checked" : "unchecked"}
      >
        <CheckIcon className="size-3.5" />
      </CheckboxIndicator>
    </CheckboxRoot>
  );
}

export { Checkbox, DummyCheckbox };
