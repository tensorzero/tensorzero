import { useEffect, useState } from "react";

export function AnimatedEllipsis() {
  const [dots, setDots] = useState("");

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? "" : prev + "."));
    }, 400);
    return () => clearInterval(interval);
  }, []);

  return <span className="inline-block w-3 text-left">{dots}</span>;
}
