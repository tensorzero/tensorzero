import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { Code } from "~/components/ui/code";
import { useNavigate } from "~/safe-navigation";

interface TagsTableProps {
  tags: Record<string, string>;
}

export function TagsTable({ tags }: TagsTableProps) {
  const navigate = useNavigate();

  // Hardcoded list of keys that should trigger navigation
  const navigableKeys = [
    "tensorzero::evaluation_name",
    "tensorzero::dataset_name",
    "tensorzero::evaluator_inference_id",
    "tensorzero::dynamic_evaluation_run_id",
  ];
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
          navigate({
            pathname: [
              "/evaluations/:evaluation_name",
              { evaluation_name: evaluationName },
            ],
            search: `?evaluation_run_ids=${value}`,
          });
          break;
        }
        case "tensorzero::datapoint_id": {
          const datasetName = tags["tensorzero::dataset_name"];
          // Guaranteed to be present by the check above
          if (!datasetName) {
            return;
          }
          navigate([
            "/datasets/:dataset_name/datapoint/:id",
            {
              dataset_name: datasetName,
              id: value,
            },
          ]);
          break;
        }
        case "tensorzero::evaluation_name":
          navigate([
            "/evaluations/:evaluation_name",
            { evaluation_name: value },
          ]);
          break;
        case "tensorzero::dataset_name":
          navigate(["/datasets/:dataset_name", { dataset_name: value }]);
          break;
        case "tensorzero::evaluator_inference_id":
          navigate([
            "/observability/inferences/:inference_id",
            { inference_id: value },
          ]);
          break;
        case "tensorzero::dynamic_evaluation_run_id":
          navigate(["/dynamic_evaluations/runs/:run_id", { run_id: value }]);
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
          <TableEmptyState message="No tags found" />
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
