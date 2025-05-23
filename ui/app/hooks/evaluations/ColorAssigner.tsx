import { useRef, useState, useEffect, createContext, useContext } from "react";

// Color palette with 10 distinct options
const COLORS = [
  "bg-blue-600 hover:bg-blue-700",
  "bg-purple-600 hover:bg-purple-700",
  "bg-green-600 hover:bg-green-700",
  "bg-red-600 hover:bg-red-700",
  "bg-amber-600 hover:bg-amber-700",
  "bg-pink-600 hover:bg-pink-700",
  "bg-teal-600 hover:bg-teal-700",
  "bg-indigo-600 hover:bg-indigo-700",
  "bg-cyan-600 hover:bg-cyan-700",
  "bg-orange-600 hover:bg-orange-700",
];

// Non-hover versions of the colors for badges and indicators
const BASE_COLORS = [
  "bg-blue-600",
  "bg-purple-600",
  "bg-green-600",
  "bg-red-600",
  "bg-amber-600",
  "bg-pink-600",
  "bg-teal-600",
  "bg-indigo-600",
  "bg-cyan-600",
  "bg-orange-600",
];

// Default fallback color
const DEFAULT_COLOR = "bg-gray-300 hover:bg-gray-500";
const DEFAULT_BASE_COLOR = "bg-gray-300";

// Create a context to share color assignment between components
const ColorAssignerContext = createContext<{
  getColor: (runId: string, withHover?: boolean) => string;
}>({
  getColor: () => DEFAULT_COLOR,
});
ColorAssignerContext.displayName = "ColorAssignerContext";

export function ColorAssignerProvider({
  children,
  selectedRunIds,
}: {
  children: React.ReactNode;
  selectedRunIds: string[];
}) {
  // Store the mapping of runId to color
  const [colorMap, setColorMap] = useState(new Map<string, number>());

  // Function to get a color for a run ID
  const getColor = (runId: string, withHover = true) => {
    if (colorMap.has(runId)) {
      const colorIndex = colorMap.get(runId)!;
      return withHover ? COLORS[colorIndex] : BASE_COLORS[colorIndex];
    }
    return withHover ? DEFAULT_COLOR : DEFAULT_BASE_COLOR;
  };

  // Update color assignments when selected run IDs change
  const selectedRunIdsRef = useRef<string[]>([]);
  useEffect(() => {
    // because selectedRunIds is an array of strings that may not be memoized,
    // do a manual shallow comparison before running the sync effect to prevent
    // render loops
    const previousRunIds = selectedRunIdsRef.current;
    selectedRunIdsRef.current = selectedRunIds;
    if (arraysEqual(selectedRunIds, previousRunIds)) {
      return;
    }

    setColorMap((colorMap) => {
      // Create a new map to store updated assignments
      const newColorMap = new Map<string, number>();
      const usedColorIndices = new Set<number>();

      // First, preserve existing assignments for IDs that are still selected
      selectedRunIds.forEach((runId) => {
        if (colorMap.has(runId)) {
          const colorIndex = colorMap.get(runId)!;
          newColorMap.set(runId, colorIndex);
          usedColorIndices.add(colorIndex);
        }
      });

      // Then assign new colors to IDs that don't have one yet
      selectedRunIds.forEach((runId) => {
        if (!newColorMap.has(runId)) {
          // Find the first available color
          let colorIndex = 0;
          while (
            usedColorIndices.has(colorIndex) &&
            colorIndex < COLORS.length
          ) {
            colorIndex++;
          }

          // If we have more runs than colors, use the default
          if (colorIndex < COLORS.length) {
            newColorMap.set(runId, colorIndex);
            usedColorIndices.add(colorIndex);
          }
        }
      });

      // update the state
      return newColorMap;
    });
  }, [selectedRunIds]);

  return (
    <ColorAssignerContext.Provider value={{ getColor }}>
      {children}
    </ColorAssignerContext.Provider>
  );
}

// Hook to use the color assigner context
export function useColorAssigner() {
  return useContext(ColorAssignerContext);
}

function arraysEqual(a: unknown[] = [], b: unknown[] = []) {
  if (a.length !== b.length) {
    return false;
  }

  return a.every((item, index) => Object.is(item, b[index]));
}
