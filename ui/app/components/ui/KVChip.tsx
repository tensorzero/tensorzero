import type { HTMLAttributes } from "react";
import { Link } from "~/safe-navigation";

interface KVChipProps extends HTMLAttributes<HTMLDivElement> {
  k: string;
  v: string;
  k_href?: string;
  v_href?: string;
  separator?: string;
}

export default function KVChip({
  k,
  v,
  k_href,
  v_href,
  separator = ":",
  ...props
}: KVChipProps) {
  return (
    <div className="flex flex-wrap gap-1" key={k} {...props}>
      <div className="flex items-center gap-1 rounded bg-gray-100 px-2 py-0.5 font-mono text-xs">
        {k_href ? (
          <Link unsafeTo={k_href} className="text-blue-600 hover:underline">
            {k}
          </Link>
        ) : (
          <span>{k}</span>
        )}
        <span className="text-gray-400">{separator}</span>
        {v_href ? (
          <Link unsafeTo={v_href} className="text-blue-600 hover:underline">
            {v}
          </Link>
        ) : (
          <span>{v}</span>
        )}
      </div>
    </div>
  );
}
