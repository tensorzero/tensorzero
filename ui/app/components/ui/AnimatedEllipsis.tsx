import { useState, useEffect } from "react";

type AnimatedEllipsisProps = {
  interval?: number;
};

export function AnimatedEllipsis({ interval = 400 }: AnimatedEllipsisProps) {
  const [dots, setDots] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setDots((prev) => (prev + 1) % 4);
    }, interval);
    return () => clearInterval(timer);
  }, [interval]);

  return <span>{".".repeat(dots)}</span>;
}
