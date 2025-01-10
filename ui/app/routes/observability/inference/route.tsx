import {
  queryInferenceById,
  queryModelInferencesByInferenceId,
} from "~/utils/clickhouse/inference";
import {
  queryFeedbackBoundsByTargetId,
  queryFeedbackByTargetId,
} from "~/utils/clickhouse/feedback";
import type { Route } from "./+types/route";
import { data, isRouteErrorResponse, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import BasicInfo from "./BasicInfo";
import Input from "./Input";
import Output from "./Output";
import FeedbackTable from "~/components/feedback/FeedbackTable";
import { ParameterCard } from "./InferenceParameters";
import { TagsTable } from "./TagsTable";
import { ModelInferencesAccordion } from "./ModelInferencesAccordion";
import { TooltipContent } from "~/components/ui/tooltip";
import { TooltipTrigger } from "~/components/ui/tooltip";
import { Tooltip } from "~/components/ui/tooltip";
import { TooltipProvider } from "~/components/ui/tooltip";
import { Badge } from "~/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { inference_id } = params;
  const url = new URL(request.url);
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }

  const [inference, model_inferences, feedback, feedback_bounds] =
    await Promise.all([
      queryInferenceById(inference_id),
      queryModelInferencesByInferenceId(inference_id),
      queryFeedbackByTargetId({
        target_id: inference_id,
        before: beforeFeedback || undefined,
        after: afterFeedback || undefined,
        page_size: pageSize,
      }),
      queryFeedbackBoundsByTargetId({ target_id: inference_id }),
    ]);
  if (!inference) {
    throw data(`No inference found for id ${inference_id}.`, {
      status: 404,
    });
  }

  return {
    inference,
    model_inferences,
    feedback,
    feedback_bounds,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const { inference, model_inferences, feedback, feedback_bounds } = loaderData;
  const navigate = useNavigate();

  const topFeedback = feedback[0] as { id: string } | undefined;
  const bottomFeedback = feedback[feedback.length - 1] as
    | { id: string }
    | undefined;

  const handleNextFeedbackPage = () => {
    if (!bottomFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterFeedback");
    searchParams.set("beforeFeedback", bottomFeedback.id);
    navigate(`?${searchParams.toString()}`);
  };

  const handlePreviousFeedbackPage = () => {
    if (!topFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeFeedback");
    searchParams.set("afterFeedback", topFeedback.id);
    navigate(`?${searchParams.toString()}`);
  };

  // These are swapped because the table is sorted in descending order
  const disablePreviousFeedbackPage =
    !topFeedback?.id ||
    !feedback_bounds.last_id ||
    feedback_bounds.last_id === topFeedback.id;

  const disableNextFeedbackPage =
    !bottomFeedback?.id ||
    !feedback_bounds.first_id ||
    feedback_bounds.first_id === bottomFeedback.id;

  const num_feedbacks = feedback.length;

  return (
    <div className="container mx-auto space-y-6 p-4">
      <h2 className="mb-4 text-2xl font-semibold">
        Inference{" "}
        <code className="rounded bg-gray-100 p-1 text-2xl">{inference.id}</code>
      </h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>

      <BasicInfo inference={inference} />
      <Input input={inference.input} />
      <Output output={inference.output} />
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            Feedback
            <TooltipProvider>
              <Tooltip delayDuration={0}>
                <TooltipTrigger asChild>
                  <Badge variant="outline" className="px-2 py-0.5 text-xs">
                    inference
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    This table only includes inference-level feedback. To see
                    episode-level feedback, open the detail page for that
                    episode.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <Badge variant="secondary">Count: {num_feedbacks}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <FeedbackTable feedback={feedback} />
          <PageButtons
            onNextPage={handleNextFeedbackPage}
            onPreviousPage={handlePreviousFeedbackPage}
            disableNext={disableNextFeedbackPage}
            disablePrevious={disablePreviousFeedbackPage}
          />
        </CardContent>
      </Card>
      <ParameterCard
        title="Inference Parameters"
        parameters={inference.inference_params}
      />
      {inference.function_type === "chat" && (
        <ParameterCard
          title="Tool Parameters"
          parameters={inference.tool_params}
        />
      )}
      {inference.function_type === "json" && (
        <ParameterCard
          title="Output Schema"
          parameters={inference.output_schema}
        />
      )}
      {Object.keys(inference.tags).length > 0 && (
        <TagsTable tags={inference.tags} />
      )}
      <ModelInferencesAccordion modelInferences={model_inferences} />
    </div>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
