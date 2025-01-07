import { Badge } from "~/components/ui/badge";
import {
  Clock,
  ExternalLink,
  Server,
  ActivityIcon as Function,
  BarChart2,
  MessageSquare,
} from "lucide-react";
import type { SFTJobStatus } from "~/utils/supervised_fine_tuning/common";
import { ModelBadge } from "~/components/model/ModelBadge";
import { extractTimestampFromUUIDv7 } from "~/utils/common";
import { MetadataItem } from "./MetadataItem";
import { RawDataAccordion } from "./RawDataAccordion";
import { ProgressIndicator } from "./ProgressIndicator";

// const jobStatus = "running"; // This could be "Running", "Completed", or "Error"
// const modelProvider = "openai"; // This could be "OpenAI" or "Fireworks"
// const jobId = "ft-1234567890";
// const createdTime = new Date("2024-01-06T11:00:53.000Z");
// const baseModel = "gpt-4o-mini-2024-07-18";
// const functionName = "write_haiku";
// const metric = "user_score";
// const prompt = "v2";

// const rawData = {
//   id: jobId,
//   status: jobStatus,
//   provider: modelProvider,
//   created_at: createdTime.toISOString(),
//   base_model: baseModel,
//   estimated_completion: estimatedCompletionTime.toISOString(),
//   function: functionName,
//   metric: metric,
//   prompt: prompt,
//   // Add more fields as needed
// };

export default function LLMFineTuningStatus({
  status,
}: {
  status: SFTJobStatus;
}) {
  if (status.status === "idle") return null;
  const createdAt = extractTimestampFromUUIDv7(status.formData.jobId);
  return (
    <div className="container mx-auto space-y-6 bg-background p-6">
      <div className="space-y-4">
        <div className="space-y-2">
          <h3 className="text-lg font-semibold">
            Job{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-sm">
              {status.formData.jobId}
            </code>
          </h3>
          <div className="flex items-center gap-2">
            <Badge
              variant={
                status.status === "running"
                  ? "default"
                  : status.status === "completed"
                    ? "secondary"
                    : "destructive"
              }
            >
              {status.status}
            </Badge>
            <ModelBadge provider={status.modelProvider} />
          </div>
        </div>

        <div className="space-y-2">
          <MetadataItem
            icon={Server}
            label="Base Model"
            value={status.formData.model.name}
          />
          <MetadataItem
            icon={Function}
            label="Function"
            value={status.formData.function}
          />
          <MetadataItem
            icon={BarChart2}
            label="Metric"
            value={status.formData.metric ?? "None"}
          />
          <MetadataItem
            icon={MessageSquare}
            label="Prompt"
            value={status.formData.variant}
          />
          <MetadataItem
            icon={Clock}
            label="Created"
            value={createdAt.toLocaleString()}
            isRaw
          />
        </div>

        {/* hide if not available from provider, eg fireworks */}
        {status.status === "running" &&
          "estimatedCompletionTime" in status &&
          status.estimatedCompletionTime && (
            <div className="max-w-lg space-y-2">
              <ProgressIndicator
                createdAt={createdAt}
                estimatedCompletion={status.estimatedCompletionTime}
              />
            </div>
          )}

        <a
          href={status.jobUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center text-primary hover:underline"
        >
          View Job Details
          <ExternalLink className="ml-1 h-4 w-4" />
        </a>
        <RawDataAccordion rawData={status.rawData} />
      </div>
    </div>
  );
}

/*
 <Accordion
          type="single"
          collapsible
          className="w-full rounded-md border"
          /* If rawData exists, is an object with a status property that equals "error",
             automatically expand the raw data accordion to show error details.
             Otherwise leave accordion collapsed by setting defaultValue to undefined
             defaultValue={
                status.rawData &&
                typeof status.rawData === "object" &&
                "status" in status.rawData &&
                status.rawData.status === "error"
                  ? "raw-data"
                  : undefined
              }
            >
              <AccordionItem value="raw-data" className="border-none">
                <AccordionTrigger className="px-4">Raw Data</AccordionTrigger>
                <AccordionContent className="px-4">
                  <pre className="overflow-x-auto rounded-md bg-muted p-4">
                    {JSON.stringify(status.rawData, null, 2)}
                  </pre>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
*/
