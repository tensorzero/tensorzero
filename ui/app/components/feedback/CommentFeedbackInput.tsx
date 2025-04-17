import { Label } from "~/components/ui/label";
import { Textarea } from "~/components/ui/textarea";

interface CommentFeedbackInputProps {
  value: string;
  onChange: (value: string) => void;
}

export default function CommentFeedbackInput({
  value,
  onChange,
}: CommentFeedbackInputProps) {
  return (
    <div className="mt-4 space-y-2">
      <Label htmlFor="comment-input">Comment</Label>
      <Textarea
        id="comment-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Enter your comment"
        rows={4}
      />
    </div>
  );
}
