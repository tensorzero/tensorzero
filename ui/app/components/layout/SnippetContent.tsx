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
  FileCode,
} from "lucide-react";
import { useBase64UrlToBlobUrl } from "~/hooks/use-blob-url";
import { CodeEditor, useFormattedJson } from "../ui/code-editor";
import { useState } from "react";

export function EmptyMessage({ message = "No content" }: { message?: string }) {
  return (
    <div className="text-fg-muted flex items-center justify-center py-12 text-sm">
      {message}
    </div>
  );
}

interface LabelProps {
  children?: React.ReactNode;
  icon?: React.ReactNode;
}

function Label({ children, icon }: LabelProps) {
  return (
    children && (
      <div className="flex flex-row items-center gap-1">
        {icon}
        <span className="text-fg-tertiary text-xs font-medium">{children}</span>
      </div>
    )
  );
}

interface TextMessageProps {
  label?: string;
  content?: string;
  footer?: string | React.ReactNode | null;
  emptyMessage?: string;
  isEditing?: boolean;
  onChange?: (value: string) => void;
}

export function TextMessage({
  label,
  content,
  footer,
  emptyMessage,
  isEditing,
  onChange,
}: TextMessageProps) {
  const formattedContent = useFormattedJson(content || "");

  return content === undefined && !isEditing ? (
    <EmptyMessage message={emptyMessage} />
  ) : (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label icon={<AlignLeftIcon className="text-fg-muted h-3 w-3" />}>
        {label}
      </Label>
      <CodeEditor
        value={formattedContent}
        readOnly={!isEditing}
        onChange={onChange}
      />
      {footer ? (
        <div className="text-fg-tertiary text-xs font-medium">{footer}</div>
      ) : null}
    </div>
  );
}

export function TemplateMessage({
  templateName,
  arguments: templateArguments,
  isEditing,
  onChange,
}: {
  templateName: string;
  arguments: unknown;
  isEditing?: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onChange?: (value: any) => void;
}) {
  const formattedJson = useFormattedJson(templateArguments ?? {});
  const [jsonError, setJsonError] = useState<string | null>(null);

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label icon={<FileCode className="text-fg-muted h-3 w-3" />}>
        Template: <span className="font-mono text-xs">{templateName}</span>
      </Label>
      <CodeEditor
        allowedLanguages={["json"]}
        value={formattedJson}
        readOnly={!isEditing}
        onChange={(updatedJson) => {
          try {
            const parsedJson = JSON.parse(updatedJson);
            setJsonError(null);
            onChange?.(parsedJson);
          } catch {
            setJsonError("Invalid JSON format");
          }
        }}
      />
      {jsonError && <div className="text-xs text-red-500">{jsonError}</div>}
    </div>
  );
}

function ToolDetails({
  name,
  nameLabel,
  id,
  payload,
  payloadLabel,
  isEditing,
  onChange,
  enforceJson = false,
}: {
  name: string;
  nameLabel: string;
  id: string;
  payload: string;
  payloadLabel: string;
  isEditing?: boolean;
  onChange?: (
    toolCallId: string,
    toolName: string,
    toolArguments: string,
  ) => void;
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
        readOnly={!isEditing}
        onChange={(updatedPayload) => {
          onChange?.(id, name, updatedPayload);
        }}
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
  isEditing?: boolean;
  onChange?: (
    toolCallId: string,
    toolName: string,
    toolArguments: string,
  ) => void;
}

interface ModelInferenceToolCallMessageProps {
  toolName: string;
  toolArguments: string;
  toolCallId: string;
}

export function ToolCallMessage(
  toolCall: ToolCallMessageProps | ModelInferenceToolCallMessageProps,
) {
  let toolName: string;
  let toolArguments: string;
  let nameLabel: string;
  let payloadLabel: string;

  if ("toolRawArguments" in toolCall) {
    nameLabel = toolCall.toolName ? "Name" : "Name (Invalid)";
    payloadLabel = toolCall.toolArguments ? "Arguments" : "Arguments (Invalid)";
    toolName = toolCall.toolName || toolCall.toolRawName;
    toolArguments = toolCall.toolArguments || toolCall.toolRawArguments;
  } else {
    nameLabel = "Name";
    payloadLabel = "Arguments";
    toolName = toolCall.toolName;
    toolArguments = toolCall.toolArguments;
  }

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label icon={<Terminal className="text-fg-muted h-3 w-3" />}>
        Tool Call
      </Label>
      <ToolDetails
        name={toolName}
        nameLabel={nameLabel}
        id={toolCall.toolCallId}
        payload={toolArguments}
        payloadLabel={payloadLabel}
        isEditing={"isEditing" in toolCall ? toolCall.isEditing : undefined}
        onChange={"onChange" in toolCall ? toolCall.onChange : undefined}
        enforceJson={true}
      />
    </div>
  );
}

interface ToolResultMessageProps {
  toolName: string;
  toolResult: string;
  toolResultId: string;
  isEditing?: boolean;
  onChange?: (
    toolResultId: string,
    toolName: string,
    toolResult: string,
  ) => void;
}

export function ToolResultMessage({
  toolName,
  toolResult,
  toolResultId,
  isEditing,
  onChange,
}: ToolResultMessageProps) {
  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <Label icon={<ArrowRight className="text-fg-muted h-3 w-3" />}>
        Tool Result
      </Label>
      <ToolDetails
        name={toolName}
        nameLabel="Name"
        id={toolResultId}
        payload={toolResult}
        payloadLabel="Result"
        isEditing={isEditing}
        onChange={onChange}
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
      <Label icon={<ImageIcon className="text-fg-muted h-3 w-3" />}>
        Image
      </Label>
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
      <Label icon={<ImageIcon className="text-fg-muted h-3 w-3" />}>
        Image (Error)
      </Label>
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
      <Label icon={<FileAudio className="text-fg-muted h-3 w-3" />}>
        Audio
      </Label>

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
      <Label icon={<FileText className="text-fg-muted h-3 w-3" />}>File</Label>
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
