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
        <CardTitle className="text-xl">Basic Information</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-4">
          <div>
            <dt className="text-lg font-semibold">Function</dt>
            <dd>
              <Code>{inference.function_name}</Code>
            </dd>
          </div>
          <div>
            <dt className="text-lg font-semibold">Variant</dt>
            <Code>{inference.variant_name}</Code>
            <Badge variant="outline" className="ml-2 bg-blue-200">
              {variantType}
            </Badge>
          </div>
          <div>
            <dt className="text-lg font-semibold">Episode ID</dt>
            <dd>
              <Link to={`/observability/episode/${inference.episode_id}`}>
                <Code>{inference.episode_id}</Code>
              </Link>
            </dd>
          </div>
          <div>
            <dt className="text-lg font-semibold">Timestamp</dt>
            <dd>{new Date(inference.timestamp).toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-lg font-semibold">Processing Time</dt>
            <dd>{inference.processing_time_ms}ms</dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
}
