import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Code } from "~/components/ui/code";

interface TagsTableProps {
  tags: Record<string, string>;
}

export function TagsTable({ tags }: TagsTableProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">Tags</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Key</TableHead>
              <TableHead>Value</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {Object.entries(tags).map(([key, value]) => (
              <TableRow key={key}>
                <TableCell>
                  <Code>{key}</Code>
                </TableCell>
                <TableCell>
                  <Code>{value}</Code>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
