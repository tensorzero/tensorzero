import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";
import type { RawData } from "~/utils/supervised_fine_tuning/common";
import { useEffect, useState } from "react";
import { Textarea } from "~/components/ui/textarea";

interface RawDataAccordionProps {
  rawData: RawData;
}

type RawDataWithTimestamp = RawData & {
  timestamp: Date;
};

export function RawDataAccordion({ rawData }: RawDataAccordionProps) {
  const [history, setHistory] = useState<RawDataWithTimestamp[]>([]);
  // Add state for accordion value
  const [accordionValue, setAccordionValue] = useState<string | undefined>(
    undefined,
  );

  // Update accordion value when new data comes in
  useEffect(() => {
    if (rawData && rawData.status === "error") {
      setAccordionValue("raw-data");
    } else {
      setAccordionValue(undefined);
    }
  }, [rawData]);

  // Add new raw data to history when it changes
  useEffect(() => {
    if (rawData) {
      setHistory((prev) => [...prev, { ...rawData, timestamp: new Date() }]);
    }
  }, [rawData]);

  // Add these utility functions
  const getSerializedData = (entry: RawData) => {
    return JSON.stringify(entry, null, 2);
  };

  const getTextAreaHeight = (text: string) => {
    const lineCount = text.split("\n").length;
    return Math.max(lineCount * 21, 100); // Adjust multiplier as needed
  };

  const getEntryClassName = (entry: RawData) => {
    return entry.status === "error" ? "border-red-200" : "border-gray-200";
  };

  return (
    <Accordion
      type="single"
      collapsible
      className="w-full rounded-md border"
      value={accordionValue}
      onValueChange={setAccordionValue}
    >
      <AccordionItem value="raw-data" className="border-none">
        <AccordionTrigger className="px-4">Raw Data History</AccordionTrigger>
        <AccordionContent className="space-y-4 px-4">
          {[...history].reverse().map((entry, index) => {
            const serializedData = getSerializedData(entry);
            const height = getTextAreaHeight(serializedData);

            return (
              <div
                key={index}
                className={`rounded-md border p-3 ${getEntryClassName(entry)}`}
              >
                <div className="mb-1 flex items-center justify-between">
                  <div className="text-sm text-gray-500">
                    {new Date(entry.timestamp).toLocaleTimeString()}
                  </div>
                  {entry.status === "error" && (
                    <div className="text-sm font-medium text-red-600">
                      Error
                    </div>
                  )}
                </div>
                <Textarea
                  value={serializedData}
                  style={{ height: `${height}px` }}
                  className={`w-full resize-none rounded border-none bg-slate-100 p-2 font-mono text-sm focus:ring-0 dark:bg-slate-800 ${
                    entry.status === "error" ? "bg-red-50 text-red-700" : ""
                  }`}
                  readOnly
                />
              </div>
            );
          })}
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
