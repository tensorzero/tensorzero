import { useEffect } from "react";
import { Link, useFetcher } from "react-router";
import { Sheet, SheetContent } from "~/components/ui/sheet";
import { useInferenceSideSheet } from "./InferenceSideSheetContext";
import type { StoredInference } from "~/types/tensorzero";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import Chip from "~/components/ui/Chip";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import { Calendar } from "~/components/icons/Icons";
import { toFunctionUrl, toVariantUrl, toEpisodeUrl } from "~/utils/urls";
import { formatDateWithSeconds } from "~/utils/date";
import { getFunctionTypeIcon } from "~/utils/icon";
import {
  BasicInfoLayout,
  BasicInfoLayoutSkeleton,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import {
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { ExternalLink } from "lucide-react";

export function InferenceSideSheet() {
  const { inferenceId, closeSheet } = useInferenceSideSheet();
  const fetcher = useFetcher<StoredInference>({
    key: inferenceId ? `inference-sheet-${inferenceId}` : "inference-sheet",
  });

  useEffect(() => {
    if (inferenceId && fetcher.state === "idle" && !fetcher.data) {
      fetcher.load(
        `/api/tensorzero/inference_sheet/${encodeURIComponent(inferenceId)}`,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inferenceId]);

  const inference = fetcher.data as StoredInference | undefined;
  const isLoading =
    fetcher.state !== "idle" || (Boolean(inferenceId) && !inference);

  return (
    <Sheet
      open={Boolean(inferenceId)}
      onOpenChange={(open) => !open && closeSheet()}
    >
      <SheetContent className="overflow-y-auto">
        <div className="flex flex-col gap-6 pt-4">
          <div className="flex items-start justify-between gap-3">
            <div className="flex flex-col gap-1">
              <span className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
                Inference
              </span>
              <span className="font-mono text-sm break-all">{inferenceId}</span>
            </div>
            {inferenceId && (
              <Link
                to={`/observability/inferences/${encodeURIComponent(inferenceId)}`}
                className="text-muted-foreground hover:text-foreground mt-1 flex shrink-0 items-center gap-1 text-xs transition-colors"
              >
                Open full page
                <ExternalLink className="h-3 w-3" />
              </Link>
            )}
          </div>

          {isLoading ? (
            <BasicInfoLayoutSkeleton rows={4} />
          ) : inference ? (
            <>
              <InferenceBasicInfo inference={inference} />
              <SectionsGroup>
                <SectionLayout>
                  <SectionHeader heading="Output" />
                  {inference.type === "json" ? (
                    <JsonOutputElement
                      output={inference.output}
                      outputSchema={inference.output_schema}
                    />
                  ) : (
                    <ChatOutputElement output={inference.output} />
                  )}
                </SectionLayout>
              </SectionsGroup>
            </>
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function InferenceBasicInfo({ inference }: { inference: StoredInference }) {
  const functionIconConfig = getFunctionTypeIcon(inference.type);

  return (
    <BasicInfoLayout>
      <BasicInfoItem>
        <BasicInfoItemTitle>Function</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={functionIconConfig.icon}
            iconBg={functionIconConfig.iconBg}
            label={inference.function_name}
            secondaryLabel={`· ${inference.type}`}
            link={toFunctionUrl(inference.function_name)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Variant</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            label={inference.variant_name}
            link={toVariantUrl(inference.function_name, inference.variant_name)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Episode</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            label={inference.episode_id}
            link={toEpisodeUrl(inference.episode_id)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      <BasicInfoItem>
        <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={<Calendar className="text-fg-tertiary" />}
            label={formatDateWithSeconds(new Date(inference.timestamp))}
            tooltip={<TimestampTooltip timestamp={inference.timestamp} />}
          />
        </BasicInfoItemContent>
      </BasicInfoItem>
    </BasicInfoLayout>
  );
}
