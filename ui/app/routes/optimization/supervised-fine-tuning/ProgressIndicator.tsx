import { Progress } from "~/components/ui/progress";
import { CountdownTimer } from "~/components/ui/CountdownTimer";
import { useState, useEffect } from "react";

function getProgressPercentage(
  createdAt: Date,
  estimatedCompletion: Date,
  currentTime: Date,
): number {
  const total = estimatedCompletion.getTime() - createdAt.getTime();
  const elapsed = currentTime.getTime() - createdAt.getTime();

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
  const [progress, setProgress] = useState(
    getProgressPercentage(createdAt, estimatedCompletion, new Date())
  );

  useEffect(() => {
    // Update progress every second
    const intervalId = setInterval(() => {
      const now = new Date();
      setProgress(getProgressPercentage(
        createdAt,
        estimatedCompletion,
        now
      ));
      if (now >= estimatedCompletion) {
        clearInterval(intervalId);
      }
    }, 1000);

    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, [createdAt, estimatedCompletion]);

  return (
    <div className="max-w-lg space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-medium">Estimated Completion</span>
        <CountdownTimer targetDate={estimatedCompletion} />
      </div>
      <Progress
        value={progress}
        className="w-full"
      />
    </div>
  );
}
