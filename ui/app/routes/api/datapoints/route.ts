import { type ActionFunctionArgs } from "react-router";
import { handleAddToDatasetAction } from "~/utils/dataset.server";

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  return handleAddToDatasetAction(formData);
}
