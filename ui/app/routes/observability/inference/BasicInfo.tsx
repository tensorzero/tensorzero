import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Code } from "~/components/ui/code";
import { Badge } from "~/components/ui/badge";
import { Link } from "react-router";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import { useConfig } from "~/context/config";

interface BasicInfoProps {
  inference: ParsedInferenceRow;
}

export default function BasicInfo({ inference }: BasicInfoProps) {
  const config = useConfig();
  const variantType =
    config.functions[inference.function_name]?.variants[inference.variant_name]
      ?.type;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Basic Information</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-4">
          <div>
            <dt className="font-semibold">Function</dt>
            <dd>
              <Code>{inference.function_name}</Code>
            </dd>
          </div>
          <div>
            <dt className="font-semibold">Variant</dt>
            <dd className="flex items-center gap-2">
              <Code>{inference.variant_name}</Code>
              <Badge variant="destructive">{variantType}</Badge>
            </dd>
          </div>
          <div>
            <dt className="font-semibold">Episode ID</dt>
            <dd>
              <Link to={`/observability/episode/${inference.episode_id}`}>
                <Code>{inference.episode_id}</Code>
              </Link>
            </dd>
          </div>
          <div>
            <dt className="font-semibold">Timestamp</dt>
            <dd>{new Date(inference.timestamp).toLocaleString()}</dd>
          </div>
          <div>
            <dt className="font-semibold">Processing Time</dt>
            <dd>{inference.processing_time_ms}ms</dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
}
