import * as React from "react";
import { useToast } from "~/hooks/use-toast";
import * as Toast from "~/components/ui/toast";

export function Toaster() {
  const { toasts } = useToast();
  return (
    <Toast.Provider>
      {Array.from(toasts.values()).map(
        ({ id, title, description, action, ...props }) => (
          <Toast.Root key={id} {...props}>
            <div className="grid gap-1">
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
