import { Label } from "~/components/ui/label";
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group";

interface BooleanFeedbackInputProps {
  metricName: string;
  value: string | null;
  onChange: (value: string | null) => void;
}

export default function BooleanFeedbackInput({
  value,
  onChange,
}: BooleanFeedbackInputProps) {
  return (
    <div className="mt-4">
      <Label>Value</Label>
      <RadioGroup
        value={value ?? undefined}
        onValueChange={onChange}
        className="mt-2 flex gap-4"
      >
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="true" id={`true`} />
          <Label htmlFor={`true`}>True</Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="false" id={`false`} />
          <Label htmlFor={`false`}>False</Label>
        </div>
      </RadioGroup>
    </div>
  );
}
