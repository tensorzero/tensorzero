import { Link } from "react-router";

export default function KVChip({
  k,
  v,
  k_href,
  v_href,
}: {
  k: string;
  v: string;
  k_href?: string;
  v_href?: string;
}) {
  return (
    <div className="flex flex-wrap gap-1" key={k}>
      <div className="flex items-center gap-1 rounded bg-gray-100 px-2 py-0.5 font-mono text-xs">
        {k_href ? (
          <Link to={k_href} className="text-blue-600 hover:underline">
            {k}
          </Link>
        ) : (
          <span>{k}</span>
        )}
        <span>:</span>
        {v_href ? (
          <Link to={v_href} className="text-blue-600 hover:underline">
            {v}
          </Link>
        ) : (
          <span>{v}</span>
        )}
      </div>
    </div>
  );
}
