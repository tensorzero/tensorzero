// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  delay: number,
): (...args: Parameters<T>) => void {
  let lastCallTime = 0;
  let timeoutId: NodeJS.Timeout | undefined;

  return (...args: Parameters<T>) => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCallTime;

    if (timeSinceLastCall >= delay) {
      // Enough time has passed, execute immediately
      lastCallTime = now;
      func(...args);
    } else {
      // Not enough time has passed, schedule for later
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      const remainingTime = delay - timeSinceLastCall;
      timeoutId = setTimeout(() => {
        lastCallTime = Date.now();
        func(...args);
      }, remainingTime);
    }
  };
}
