import { useAsyncError } from "react-router";
import { Table, TableBody, TableCell, TableRow } from "~/components/ui/table";
import { TableErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";
import { AlertCircle } from "lucide-react";
import PageButtons from "~/components/utils/PageButtons";
import {
  FeedbackTableHeaders,
  ModelInferencesTableHeaders,
} from "./InferenceSkeletons";

/**
 * Error state for BasicInfo section in header.
 * Shows a compact error message since it's in the page header.
 */
export function BasicInfoError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load inference details";

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      {message}
    </div>
  );
}

/**
 * Error state for ActionBar section.
 * Shows a compact error message since actions can't work without data.
 */
export function ActionsError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load actions";

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      {message}
    </div>
  );
}

export function FeedbackSectionError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load feedback";

  return (
    <>
      <Table>
        <FeedbackTableHeaders />
        <TableBody>
          <TableRow>
            <TableCell colSpan={5}>
              <TableErrorNotice
                icon={AlertCircle}
                title="Error loading data"
                description={message}
              />
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
      <PageButtons disabled />
    </>
  );
}

export function ModelInferencesSectionError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load model inferences";

  return (
    <Table>
      <ModelInferencesTableHeaders />
      <TableBody>
        <TableRow>
          <TableCell colSpan={6}>
            <TableErrorNotice
              icon={AlertCircle}
              title="Error loading data"
              description={message}
            />
          </TableCell>
        </TableRow>
      </TableBody>
    </Table>
  );
}

export function InputSectionError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load input";

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
      {message}
    </div>
  );
}
