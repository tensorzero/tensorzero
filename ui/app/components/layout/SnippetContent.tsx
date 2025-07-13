import * as React from "react";
import { Link } from "react-router";
import {
  Terminal,
  ArrowRight,
  ImageIcon,
  ImageOff,
  Download,
  ExternalLink,
  FileText,
  FileAudio,
  AlignLeftIcon,
  BlocksIcon,
} from "lucide-react";
import { useBase64UrlToBlobUrl } from "~/hooks/use-blob-url";
import { CodeEditor, useFormattedJson } from "../ui/code-editor";

export function EmptyMessage({ message = "No content" }: { message?: string }) {
  return (
    <div className="text-fg-muted flex items-center justify-center py-12 text-sm">
      {message}
    </div>
  );
}

interface LabelProps {
  text?: string;
  icon?: React.ReactNode;
}

export function Label({ text, icon }: LabelProps) {
  return (
    text && (
      <div className="flex flex-row items-center gap-1">
        {icon}
        <span className="text-fg-tertiary text-xs font-medium">{text}</span>
      </div>
    )
  );
}

// TODO Remove `CodeMessage` - unused for input snippet rendering, but used elsewhere

// Code content component
interface CodeMessageProps {
  label?: string;
  content?: string;
  showLineNumbers?: boolean;
  emptyMessage?: string;
}

export function CodeMessage({
  label,
  content,
  showLineNumbers = false,
  emptyMessage,
}: CodeMessageProps) {
  if (!content) {
    return <EmptyMessage message={emptyMessage} />;
  }

  // We still need line count for line numbers, but won't split the content for rendering
  const lineCount = content ? content.split("\n").length : 0;

  return (
    <div className="relative w-full">
      <Label text={label} />

      <div className="w-full overflow-hidden">
        <div className="w-full">
          <div className="flex w-full">
            {showLineNumbers && (
              <div className="text-fg-muted pointer-events-none sticky left-0 min-w-[2rem] shrink-0 pr-3 text-right font-mono select-none">
                {Array.from({ length: lineCount }, (_, i) => (
                  <div key={i} className="text-sm leading-6">
                    {i + 1}
                  </div>
                ))}
              </div>
            )}
            <div className="w-0 grow overflow-auto">
              <pre className="w-full">
                <code className="text-fg-primary block font-mono text-sm leading-6 whitespace-pre">
                  {content || ""}
                </code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface TextMessageProps {
  label?: string;
  content?: string;
  emptyMessage?: string;
}

export function TextMessage({
  label,
  content,
  emptyMessage,
}: TextMessageProps) {
  const formattedContent = useFormattedJson(content || "");

  return !content ? (
    <EmptyMessage message={emptyMessage} />
  ) : (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label
        text={label}
        icon={<AlignLeftIcon className="text-fg-muted h-3 w-3" />}
      />
      <CodeEditor value={formattedContent} readOnly />
    </div>
  );
}

export function ParameterizedMessage({ parameters }: { parameters?: unknown }) {
  const formattedJson = useFormattedJson(parameters ?? {});

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label
        text="Template Arguments"
        icon={<BlocksIcon className="text-fg-muted h-3 w-3" />}
      />
      <CodeEditor allowedLanguages={["json"]} value={formattedJson} readOnly />
    </div>
  );
}

function ToolDetails({
  name,
  nameLabel,
  id,
  payload,
  payloadLabel,
  enforceJson = false,
}: {
  name: string;
  nameLabel: string;
  id: string;
  payload: string;
  payloadLabel: string;
  enforceJson?: boolean;
}) {
  const formattedPayload = useFormattedJson(payload);

  return (
    <div className="border-border bg-bg-tertiary/50 grid grid-flow-row grid-cols-[min-content_1fr] grid-rows-[repeat(3,min-content)] place-content-center gap-x-4 gap-y-1 rounded-sm px-3 py-2 text-xs">
      <p className="text-fg-secondary font-medium">{nameLabel}</p>
      <p className="self-center truncate font-mono text-[0.6875rem]">{name}</p>

      <p className="text-fg-secondary font-medium">ID</p>
      <p className="self-center truncate font-mono text-[0.6875rem]">{id}</p>

      <p className="text-fg-secondary font-medium">{payloadLabel}</p>
      <CodeEditor
        allowedLanguages={enforceJson ? ["json"] : undefined}
        value={formattedPayload}
        className="bg-bg-secondary"
        readOnly
      />
    </div>
  );
}

interface ToolCallMessageProps {
  toolName: string | null;
  toolRawName: string;
  toolArguments: string | null;
  toolRawArguments: string;
  toolCallId: string;
}

export function ToolCallMessage({
  toolName,
  toolRawName,
  toolArguments,
  toolRawArguments,
  toolCallId,
}: ToolCallMessageProps) {
  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label
        text="Tool Call"
        icon={<Terminal className="text-fg-muted h-3 w-3" />}
      />
      <ToolDetails
        name={toolName || toolRawName}
        nameLabel={toolName ? "Name" : "Name (Invalid)"}
        id={toolCallId}
        payload={toolArguments || toolRawArguments}
        payloadLabel={toolArguments ? "Arguments" : "Arguments (Invalid)"}
        enforceJson={true}
      />
    </div>
  );
}

interface ToolResultMessageProps {
  toolName: string;
  toolResult: string;
  toolResultId: string;
}

export function ToolResultMessage({
  toolName,
  toolResult,
  toolResultId,
}: ToolResultMessageProps) {
  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label
        text="Tool Result"
        icon={<ArrowRight className="text-fg-muted h-3 w-3" />}
      />
      <ToolDetails
        name={toolName}
        nameLabel="Name"
        id={toolResultId}
        payload={toolResult}
        payloadLabel="Result"
      />
    </div>
  );
}

interface ImageMessageProps {
  url: string;
  downloadName?: string;
}

export function ImageMessage({ url, downloadName }: ImageMessageProps) {
  return (
    <div className="flex flex-col gap-1">
      <Label
        text="Image"
        icon={<ImageIcon className="text-fg-muted h-3 w-3" />}
      />
      <div>
        <Link
          to={url}
          target="_blank"
          rel="noopener noreferrer"
          download={downloadName}
          className="border-border bg-bg-tertiary text-fg-tertiary flex min-h-20 w-60 items-center justify-center rounded border p-2 text-xs"
        >
          <img src={url} alt="Image" />
        </Link>
      </div>
    </div>
  );
}

interface FileErrorMessageProps {
  error: string;
}

// Image Error Message component
export function FileErrorMessage({ error }: FileErrorMessageProps) {
  return (
    <div className="flex flex-col gap-1">
      <Label
        text="Image (Error)"
        icon={<ImageIcon className="text-fg-muted h-3 w-3" />}
      />
      <div className="border-border bg-bg-tertiary relative aspect-video w-60 min-w-60 rounded-md border">
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-2">
          <ImageOff className="text-fg-muted h-4 w-4" />
          <span className="text-fg-tertiary text-center text-xs font-medium">
            {error}
          </span>
        </div>
      </div>
    </div>
  );
}

function TruncatedFileName({
  filename,
  maxLength = 32,
}: {
  filename: string;
  maxLength?: number;
}) {
  if (filename.length <= maxLength) {
    return filename;
  }

  const extension =
    filename.lastIndexOf(".") > 0
      ? filename.substring(filename.lastIndexOf("."))
      : "";
  const name = extension
    ? filename.substring(0, filename.lastIndexOf("."))
    : filename;

  if (extension.length >= maxLength - 3) {
    // If extension is too long, just truncate from the end
    return (
      <>
        <span>{filename.substring(0, maxLength - 3)}</span>
        <span className="text-fg-muted">...</span>
      </>
    );
  }

  const availableLength = maxLength - extension.length - 3; // 3 for "..."
  const frontLength = Math.ceil(availableLength / 2);
  const backLength = Math.floor(availableLength / 2);

  if (name.length <= availableLength) {
    return filename;
  }

  return (
    <>
      <span>{name.substring(0, frontLength)}</span>
      <span className="text-fg-muted">...</span>
      <span>{name.substring(name.length - backLength) + extension}</span>
    </>
  );
}

function FileMetadata({
  mimeType,
  filePath,
}: {
  mimeType: string;
  filePath: string;
}) {
  return (
    <div className="flex flex-col">
      <div className="text-fg-primary text-sm font-medium" title={filePath}>
        <TruncatedFileName filename={filePath} />
      </div>
      <div className="text-fg-tertiary text-xs">{mimeType}</div>
    </div>
  );
}

interface FileMessageProps {
  /** Base64-encoded "data:" URL containing the file data */
  fileData: string;
  filePath: string;
  mimeType: string;
}

export const AudioMessage: React.FC<FileMessageProps> = ({
  fileData,
  mimeType,
  filePath,
}) => {
  const url = useBase64UrlToBlobUrl(fileData, mimeType);

  return (
    <div className="flex flex-col gap-1">
      <Label
        text="Audio"
        icon={<FileAudio className="text-fg-muted h-3 w-3" />}
      />

      <div className="border-border flex w-80 flex-col gap-4 rounded-md border p-3">
        <FileMetadata mimeType={mimeType} filePath={filePath} />
        <audio controls preload="none" className="w-full">
          <source src={url} type={mimeType} />
        </audio>
      </div>
    </div>
  );
};

export function FileMessage({
  fileData,
  filePath,
  mimeType,
}: FileMessageProps) {
  const url = useBase64UrlToBlobUrl(fileData, mimeType);

  return (
    <div className="flex flex-col gap-1">
      <Label
        text="File"
        icon={<FileText className="text-fg-muted h-3 w-3" />}
      />
      <div className="border-border flex w-80 flex-row gap-3 rounded-md border p-3">
        <div className="flex-1">
          <FileMetadata filePath={filePath} mimeType={mimeType} />
        </div>

        <Link
          to={fileData}
          download={`tensorzero_${filePath}`}
          aria-label={`Download ${filePath}`}
        >
          <Download className="h-5 w-5" />
        </Link>

        <Link
          to={url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`Open ${filePath} in new tab`}
        >
          <ExternalLink className="h-5 w-5" />
        </Link>
      </div>
    </div>
  );
}
