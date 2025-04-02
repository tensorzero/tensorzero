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
        "tensorzero::evaluation_run_id",
        "tensorzero::datapoint_id",
        "tensorzero::evaluation_name",
        "tensorzero::dataset_name",
      ]
    : [];

  // Function to handle row click and navigation
  const handleRowClick = (key: string, value: string) => {
    if (navigableKeys.includes(key)) {
      switch (key) {
        case "tensorzero::evaluation_run_id": {
          const evaluationName = tags["tensorzero::evaluation_name"];
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
