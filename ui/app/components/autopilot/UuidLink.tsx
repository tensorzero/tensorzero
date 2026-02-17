import { Link } from "react-router";
import { useResolveUuid } from "~/hooks/useResolveUuid";
import { toResolvedObjectUrl } from "~/utils/urls";
import { UuidHoverCard } from "./UuidHoverCard";

export function UuidLink({ uuid }: { uuid: string }) {
  const { data } = useResolveUuid(uuid);

  const obj = data?.object_types.length === 1 ? data.object_types[0] : null;
  const url = obj ? toResolvedObjectUrl(uuid, obj) : null;

  if (!url || !obj) {
    return (
      <code className="bg-muted rounded px-1.5 py-0.5 font-mono text-xs font-medium">
        {uuid}
      </code>
    );
  }

  return (
    <UuidHoverCard uuid={uuid} obj={obj} url={url}>
      <Link
        to={url}
        className="rounded bg-orange-50 px-1 py-0.5 font-mono text-xs text-orange-500 no-underline hover:underline"
      >
        {uuid}
      </Link>
    </UuidHoverCard>
  );
}
