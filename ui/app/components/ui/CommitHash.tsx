import { Code } from "./code";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";
import { Check } from "../icons/Icons";
import { X } from "lucide-react";
import KVChip from "./KVChip";

interface CommitHashProps {
  tags: Record<string, string>;
}

export function CommitHash({ tags }: CommitHashProps) {
  if (!("tensorzero::git_commit_hash" in tags)) {
    return null;
  }
  const hash = tags["tensorzero::git_commit_hash"];
  const message = tags["tensorzero::git_commit_message"];
  const branch = tags["tensorzero::git_branch"];
  // If branch starts with refs/heads/, remove the prefix
  const displayBranch = branch?.startsWith("refs/heads/")
    ? branch.slice(11)
    : branch;

  const origin = tags["tensorzero::git_origin"];
  const author = tags["tensorzero::git_author"];
  const author_email = tags["tensorzero::git_author_email"];
  const untracked_files = tags["tensorzero::git_untracked_files"] === "true";
  const modified_files = tags["tensorzero::git_modified_files"] === "true";
  const shortHash = hash.slice(0, 7);
  const commitLink = origin ? `${origin}/commit/${hash}` : undefined;
  const branchLink = origin ? `${origin}/tree/${displayBranch}` : undefined;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {commitLink ? (
          displayBranch ? (
            <KVChip
              k={displayBranch}
              v={shortHash}
              k_href={branchLink}
              v_href={commitLink}
              separator="@"
            />
          ) : (
            <Code className="flex items-center gap-1 rounded bg-gray-100 px-2 py-0.5 font-mono text-xs">
              {shortHash}
            </Code>
          )
        ) : (
          <Code className="flex items-center gap-1 rounded bg-gray-100 px-2 py-0.5 font-mono text-xs">
            {shortHash}
          </Code>
        )}
      </TooltipTrigger>
      <TooltipContent>
        <div style={{ minWidth: 220 }}>
          {message && (
            <div>
              <strong>Message:</strong> {message}
            </div>
          )}
          {branch && (
            <div>
              <strong>Branch:</strong> {branch}
            </div>
          )}
          {author && (
            <div>
              <strong>Author:</strong> {author}{" "}
              {author_email && `(${author_email})`}
            </div>
          )}
          {origin && (
            <div>
              <strong>Origin:</strong> {origin}
            </div>
          )}
          {untracked_files !== undefined && (
            <div className="flex items-center gap-1">
              <strong>Untracked files:</strong>
              {untracked_files ? <Check /> : <X />}
            </div>
          )}
          {modified_files !== undefined && (
            <div className="flex items-center gap-1">
              <strong>Modified files:</strong>
              {modified_files ? <Check /> : <X />}
            </div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
