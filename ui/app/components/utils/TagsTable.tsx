import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Code } from "~/components/ui/code";
import { useNavigate } from "react-router";

const FF_ENABLE_DATASETS =
  import.meta.env.VITE_TENSORZERO_UI_FF_ENABLE_DATASETS === "1";

interface TagsTableProps {
  tags: Record<string, string>;
}

export function TagsTable({ tags }: TagsTableProps) {
  const navigate = useNavigate();

  // Hardcoded list of keys that should trigger navigation
  const navigableKeys = FF_ENABLE_DATASETS
    ? [
        "tensorzero::evaluation_name",
        "tensorzero::dataset_name",
        "tensorzero::evaluator_inference_id",
      ]
    : [];
  // The following 2 keys only get links if the evaluation or dataset name is present
  // Since that information is not guaranteed to be present,
  // we shouldn't add a cursor pointer to the row if that is missing
  if (tags["tensorzero::evaluation_name"]) {
    navigableKeys.push("tensorzero::evaluation_run_id");
  }
  if (tags["tensorzero::dataset_name"]) {
    navigableKeys.push("tensorzero::datapoint_id");
  }

  // Function to handle row click and navigation
  const handleRowClick = (key: string, value: string) => {
    if (navigableKeys.includes(key)) {
      switch (key) {
        case "tensorzero::evaluation_run_id": {
          const evaluationName = tags["tensorzero::evaluation_name"];
          // Guaranteed to be present by the check above
          if (!evaluationName) {
            return;
          }
          navigate(
            `/evaluations/${evaluationName}?evaluation_run_ids=${value}`,
          );
          break;
        }
        case "tensorzero::datapoint_id": {
          const datasetName = tags["tensorzero::dataset_name"];
          // Guaranteed to be present by the check above
          if (!datasetName) {
            return;
          }
          navigate(`/datasets/${datasetName}/datapoint/${value}`);
          break;
        }
        case "tensorzero::evaluation_name":
          navigate(`/evaluations/${value}`);
          break;
        case "tensorzero::dataset_name":
          navigate(`/datasets/${value}`);
          break;
        case "tensorzero::evaluator_inference_id":
          navigate(`/observability/inferences/${value}`);
          break;
      }
    }
  };

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Key</TableHead>
          <TableHead>Value</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {Object.keys(tags).length === 0 ? (
          <TableRow className="hover:bg-bg-primary">
            <TableCell
              colSpan={2}
              className="px-3 py-8 text-center text-fg-muted"
            >
              No tags found.
            </TableCell>
          </TableRow>
        ) : (
          Object.entries(tags).map(([key, value]) => (
            <TableRow
              key={key}
              onClick={() => handleRowClick(key, value)}
              className={
                navigableKeys.includes(key)
                  ? "hover:bg-bg-subtle cursor-pointer"
                  : ""
              }
            >
              <TableCell>
                <Code>{key}</Code>
              </TableCell>
              <TableCell>
                <Code>{value}</Code>
              </TableCell>
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  );
}
