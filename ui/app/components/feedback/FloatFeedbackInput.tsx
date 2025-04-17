import { Label } from "~/components/ui/label";
import { Input } from "~/components/ui/input";

interface FloatFeedbackInputProps {
  value: string;
  onChange: (value: string) => void;
}

export default function FloatFeedbackInput({
  value,
  onChange,
}: FloatFeedbackInputProps) {
  return (
    <div className="mt-4 space-y-2">
      <Label htmlFor="float-input">Value</Label>
      <Input
        id="float-input"
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Enter a number"
      />
    </div>
  );
}
