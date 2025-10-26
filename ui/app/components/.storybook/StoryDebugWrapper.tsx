import type { ReactNode } from "react";

export function StoryDebugWrapper({
  children,
  debugLabel,
  debugData,
}: {
  children: ReactNode;
  debugLabel: string;
  debugData: unknown;
}) {
  return (
    <div className="w-[80vw] space-y-4">
      <div className="bg-orange-100 p-8">
        <div className="bg-white p-4">{children}</div>
      </div>
      <div className="bg-orange-100 p-8">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="font-semibold">
            Debug:{" "}
            <span className="text-md font-mono font-semibold">
              {debugLabel}
            </span>
          </h3>
        </div>
        <pre className="mt-2 overflow-auto rounded bg-white p-2 text-xs">
          {debugData !== undefined
            ? JSON.stringify(debugData, null, 2)
            : "undefined"}
        </pre>
      </div>
    </div>
  );
}
