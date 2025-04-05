import type { ResolvedImageContent } from "~/utils/clickhouse/common";

export default function ImageBlock({ image }: { image: ResolvedImageContent }) {
  return (
    <div className="w-60 rounded bg-slate-100 p-2 text-xs text-slate-300">
      <div className="mb-2">Image</div>
      <a
        href={image.image.url}
        target="_blank"
        rel="noopener noreferrer"
        download={"tensorzero_" + image.storage_path.path}
      >
        <img src={image.image.url} alt="Image" />
      </a>
    </div>
  );
}
