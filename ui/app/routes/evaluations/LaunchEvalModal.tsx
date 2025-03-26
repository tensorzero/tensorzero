import { useState } from "react";
import { useFetcher } from "react-router";
import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { useConfig } from "~/context/config";

interface LaunchEvalModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function EvalForm() {
  const fetcher = useFetcher();
  const config = useConfig();
  const eval_names = Object.keys(config.evals);
  const [selectedEvalName, setSelectedEvalName] = useState<string | null>(null);

  return (
    <fetcher.Form method="post">
      <div className="mt-4">
        <label htmlFor="eval_name" className="mb-1 block text-sm font-medium">
          Evaluation
        </label>
      </div>
      <Select
        name="eval_name"
        onValueChange={(value) => setSelectedEvalName(value)}
      >
        <SelectTrigger>
          <SelectValue placeholder="Select an evaluation" />
        </SelectTrigger>
        <SelectContent>
          {eval_names.map((eval_name) => (
            <SelectItem key={eval_name} value={eval_name}>
              {eval_name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <div className="mt-4">
        <label
          htmlFor="variant_name"
          className="mb-1 block text-sm font-medium"
        >
          Variant
        </label>
      </div>
      <Select name="variant_name" disabled={!selectedEvalName}>
        <SelectTrigger>
          <SelectValue placeholder="Select a variant" />
        </SelectTrigger>
        <SelectContent>
          {(() => {
            if (!selectedEvalName) return null;

            const eval_function = config.evals[selectedEvalName].function_name;
            const variant_names = Object.keys(
              config.functions[eval_function].variants,
            );

            return variant_names.map((variant_name) => (
              <SelectItem key={variant_name} value={variant_name}>
                {variant_name}
              </SelectItem>
            ));
          })()}
        </SelectContent>
      </Select>
      <div className="mt-4">
        <label
          htmlFor="concurrency_limit"
          className="mb-1 block text-sm font-medium"
        >
          Concurrency Limit
        </label>
        <input
          type="number"
          id="concurrency_limit"
          name="concurrency_limit"
          min="1"
          defaultValue="5"
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          required
        />
      </div>
      <DialogFooter>
        <Button className="mt-2" type="submit">
          Launch
        </Button>
      </DialogFooter>
    </fetcher.Form>
  );
}

export default function LaunchEvalModal({
  isOpen,
  onClose,
}: LaunchEvalModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Launch Eval</DialogTitle>
        </DialogHeader>

        <EvalForm />
      </DialogContent>
    </Dialog>
  );
}
