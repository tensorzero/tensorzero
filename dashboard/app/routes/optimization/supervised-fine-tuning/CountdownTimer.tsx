"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "~/components/ui/card";

interface CountdownTimerProps {
  endTime: number; // Unix timestamp in milliseconds
}

export function CountdownTimer({ endTime }: CountdownTimerProps) {
  const [timeLeft, setTimeLeft] = useState(calculateTimeLeft());

  function calculateTimeLeft() {
    const difference = endTime - Date.now();
    if (difference <= 0) {
      return { hours: 0, minutes: 0, seconds: 0 };
    }

    return {
      hours: Math.floor((difference / (1000 * 60 * 60)) % 24),
      minutes: Math.floor((difference / 1000 / 60) % 60),
      seconds: Math.floor((difference / 1000) % 60),
    };
  }

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeLeft(calculateTimeLeft());
    }, 1000);

    return () => clearInterval(timer);
  }, [endTime]);

  return (
    <Card className="w-full max-w-sm mx-auto">
      <CardContent className="flex justify-center items-center h-24">
        <div className="grid grid-flow-col gap-4 text-center auto-cols-max">
          <div className="flex flex-col">
            <span className="countdown font-mono text-4xl">
              {String(timeLeft.hours).padStart(2, "0")}
            </span>
            hours
          </div>
          <div className="flex flex-col">
            <span className="countdown font-mono text-4xl">
              {String(timeLeft.minutes).padStart(2, "0")}
            </span>
            min
          </div>
          <div className="flex flex-col">
            <span className="countdown font-mono text-4xl">
              {String(timeLeft.seconds).padStart(2, "0")}
            </span>
            sec
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
