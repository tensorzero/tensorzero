import { Progress } from "~/components/ui/progress";
import { CountdownTimer } from "./CountdownTimer";

function getProgressPercentage(
  createdAt: Date,
  estimatedCompletion: Date,
): number {
  const now = new Date();
  const total = estimatedCompletion.getTime() - createdAt.getTime();
  const elapsed = now.getTime() - createdAt.getTime();

  return Math.min(Math.max((elapsed / total) * 100, 0), 100);
}

interface ProgressIndicatorProps {
  createdAt: Date;
  estimatedCompletion: Date;
}

export function ProgressIndicator({
  createdAt,
  estimatedCompletion,
}: ProgressIndicatorProps) {
  return (
    <div className="max-w-lg space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-medium">Estimated Completion</span>
        <CountdownTimer endTime={estimatedCompletion.getTime()} />
      </div>
      <Progress
        value={getProgressPercentage(createdAt, estimatedCompletion)}
        className="w-full"
      />
    </div>
  );
}
