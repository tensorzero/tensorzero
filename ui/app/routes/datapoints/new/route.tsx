import type { ActionFunctionArgs, RouteHandle } from "react-router";
import { data, redirect } from "react-router";
import { z } from "zod";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import { toDatapointUrl } from "~/utils/urls";
import { createDatapoint } from "./datapointOperations.server";
import { parseCreateDatapointFormData } from "./formDataUtils";
import { NewDatapointForm } from "./NewDatapointForm";

export const handle: RouteHandle = {
  crumb: () => ["New"],
};

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();

  try {
    const parsedData = parseCreateDatapointFormData(formData);

    const { id } = await createDatapoint({
      datasetName: parsedData.dataset_name,
      functionName: parsedData.function_name,
      functionType: parsedData.function_type,
      input: parsedData.input,
      output: parsedData.output,
      tags: parsedData.tags,
      name: parsedData.name,
    });

    return redirect(toDatapointUrl(parsedData.dataset_name, id));
  } catch (error) {
    if (error instanceof z.ZodError) {
      return data(
        {
          success: false,
          error: `Validation failed: ${error.errors.map((e) => e.message).join(", ")}`,
        },
        { status: 400 },
      );
    }

    logger.error("Error creating datapoint:", error);
    return data(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }
}

export default function NewDatapointPage() {
  return (
    <PageLayout>
      <PageHeader heading="New Datapoint" />
      <SectionLayout>
        <NewDatapointForm />
      </SectionLayout>
    </PageLayout>
  );
}
