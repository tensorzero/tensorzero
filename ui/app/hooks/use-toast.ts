// Inspired by react-hot-toast library
import * as React from "react";
import { ToastContext } from "~/context/toast-context";

function useToast() {
  const context = React.use(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }

  return {
    toasts: context.toasts,
    toast: context.toaster,
  };
}

export { useToast };
