import { Link } from "react-router";
import { Code } from "./code";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";
import { TooltipProvider } from "./tooltip";

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
  const origin = tags["tensorzero::git_origin"];
  const author = tags["tensorzero::git_author"];
  const author_email = tags["tensorzero::git_author_email"];
  const untracked_files = tags["tensorzero::git_untracked_files"] === "true";
  const modified_files = tags["tensorzero::git_modified_files"] === "true";
  const shortHash = hash.slice(0, 7);
  const link = origin ? `${origin}/commit/${hash}` : undefined;

  return (
    <TooltipProvider delayDuration={100}>
      <Tooltip>
        <TooltipTrigger asChild>
          {link ? (
            <Link to={link} target="_blank" rel="noopener noreferrer">
              <Code>{shortHash}</Code>
            </Link>
          ) : (
            <Code>{shortHash}</Code>
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
                <strong>Author:</strong> {author}
              </div>
            )}
            {author_email && (
              <div>
                <strong>Email:</strong> {author_email}
              </div>
            )}
            {origin && (
              <div>
                <strong>Origin:</strong> {origin}
              </div>
            )}
            {untracked_files !== undefined && (
              <div>
                <strong>Untracked files:</strong>{" "}
                {untracked_files ? "✅" : "❌"}
              </div>
            )}
            {modified_files !== undefined && (
              <div>
                <strong>Modified files:</strong> {modified_files ? "✅" : "❌"}
              </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
