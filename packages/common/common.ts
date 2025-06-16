/**
 * Utility function to debounce a request for a certain number of milliseconds
 * in route client loaders.
 *
 * @see
 * https://programmingarehard.com/2025/02/24/debouncing-in-react-router-v7.html/
 */
export function abortableTimeout(request: Request, ms: number) {
  const { signal } = request;
  return new Promise((resolve, reject) => {
    // If the signal is aborted by the time it reaches this, reject
    if (signal.aborted) {
      reject(signal.reason);
      return;
    }

    // Schedule the resolve function to be called in the future a certain number
    // of milliseconds
    const timeoutId = setTimeout(resolve, ms);

    // Listen for the abort event. If it fires, reject
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timeoutId);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}
