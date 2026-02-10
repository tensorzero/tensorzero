import * as React from "react";
import { useToast } from "~/hooks/use-toast";
import { cn } from "~/utils/common";
import * as Toast from "~/components/ui/toast";

export function Toaster() {
  const { toasts } = useToast();
  return (
    <Toast.Provider>
      {Array.from(toasts.values()).map(
        ({
          id,
          icon: Icon,
          iconClassName,
          title,
          description,
          action,
          ...props
        }) => (
          <Toast.Root key={id} {...props}>
            {Icon && (
              <Icon
                className={cn(
                  "text-foreground h-5 w-5 shrink-0",
                  iconClassName,
                )}
              />
            )}
            <div className="grid flex-1 gap-1">
              {title && <Toast.Title>{title}</Toast.Title>}
              {description && (
                <Toast.Description>{description}</Toast.Description>
              )}
            </div>
            {action}
            <Toast.Close />
          </Toast.Root>
        ),
      )}
      <Toast.Viewport />
    </Toast.Provider>
  );
}
