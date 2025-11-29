import { Link } from "react-router";
import {
  ImageIcon,
  ImageOff,
  Download,
  ExternalLink,
  FileText,
  FileAudio,
  Link as LinkIcon,
} from "lucide-react";
import { type ReactNode } from "react";
import { useBase64UrlToBlobUrl } from "~/hooks/use-blob-url";
import { ContentBlockLabel } from "~/components/input_output/content_blocks/ContentBlockLabel";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";
import { Input } from "~/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import type { File, Detail } from "~/types/tensorzero";

interface FileContentBlockProps {
  block: File;
  actionBar?: ReactNode;
  isEditing?: boolean;
  onChange?: (updatedBlock: File) => void;
}

/**
 * Main component for rendering file content blocks.
 * Dispatches to specific renderers based on MIME type.
 */
export function FileContentBlock({
  block,
  actionBar,
  isEditing,
  onChange,
}: FileContentBlockProps) {
  switch (block.file_type) {
    case "object_storage":
      break; // handled below
    case "url":
      return (
        <UrlFileContentBlock
          block={block}
          actionBar={actionBar}
          isEditing={isEditing}
          onChange={onChange}
        />
      );
    case "base64":
      // TODO (#4407): we'll need to support this to allow CRUD on file content blocks
      throw new Error(
        "The UI should never receive a base64 file. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.",
      );
    case "object_storage_pointer":
      // TODO: should we handle this better?
      throw new Error(
        "The UI should never receive an object storage pointer. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.",
      );
    case "object_storage_error":
      return (
        <FileErrorContentBlock error={block.error} actionBar={actionBar} />
      );
  }

  // If we got here, we know that block.file_type is "object_storage"
  block satisfies Extract<File, { file_type: "object_storage" }>; // compile-time sanity check

  // Determine which component to render based on mime type
  if (block.mime_type?.startsWith("image/")) {
    return (
      <ImageContentBlock
        imageUrl={block.data}
        filePath={block.storage_path.path}
        actionBar={actionBar}
      />
    );
  }

  if (block.mime_type?.startsWith("audio/")) {
    return (
      <AudioContentBlock
        base64Data={block.data}
        mimeType={block.mime_type}
        filePath={block.storage_path.path}
        actionBar={actionBar}
      />
    );
  }

  return (
    <GenericFileContentBlock
      base64Data={block.data}
      mimeType={block.mime_type}
      filePath={block.storage_path.path}
      actionBar={actionBar}
    />
  );
}

interface ImageContentBlockProps {
  /** HTTP or data URL for the image */
  imageUrl: string;
  filePath: string;
  actionBar?: ReactNode;
}

/**
 * Renders image files with preview and download link.
 */
function ImageContentBlock({
  imageUrl,
  filePath,
  actionBar,
}: ImageContentBlockProps) {
  return (
    <div className="flex flex-col gap-1">
      <ContentBlockLabel
        icon={<ImageIcon className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File
      </ContentBlockLabel>
      <Link
        to={imageUrl}
        target="_blank"
        rel="noopener noreferrer"
        download={`tensorzero_${filePath}`}
        className="border-border bg-bg-tertiary text-fg-tertiary flex min-h-20 w-60 items-center justify-center rounded border p-2 text-xs"
      >
        <img src={imageUrl} alt="Image" />
      </Link>
    </div>
  );
}

interface AudioContentBlockProps {
  /** Base64-encoded data URL (data:audio/...) */
  base64Data: string;
  mimeType: string;
  filePath: string;
  actionBar?: ReactNode;
}

/**
 * Renders audio files with player controls and metadata.
 * Converts base64 data URL to blob URL for audio playback.
 */
function AudioContentBlock({
  base64Data,
  mimeType,
  filePath,
  actionBar,
}: AudioContentBlockProps) {
  const blobUrl = useBase64UrlToBlobUrl(base64Data, mimeType);

  return (
    <div className="flex flex-col gap-1">
      <ContentBlockLabel
        icon={<FileAudio className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File
      </ContentBlockLabel>

      <div className="border-border flex w-80 flex-col gap-4 rounded-md border p-3">
        <FileMetadata mimeType={mimeType} filePath={filePath} />
        <audio controls preload="none" className="w-full">
          <source src={blobUrl} type={mimeType} />
        </audio>
      </div>
    </div>
  );
}

interface GenericFileContentBlockProps {
  /** Base64-encoded data URL (data:...) */
  base64Data: string;
  mimeType: string;
  filePath: string;
  actionBar?: ReactNode;
}

/**
 * Renders generic files with metadata and download/open actions.
 * Converts base64 data URL to blob URL for browser preview.
 */
function GenericFileContentBlock({
  base64Data,
  filePath,
  mimeType,
  actionBar,
}: GenericFileContentBlockProps) {
  const blobUrl = useBase64UrlToBlobUrl(base64Data, mimeType);

  return (
    <div className="flex flex-col gap-1">
      <ContentBlockLabel
        icon={<FileText className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File
      </ContentBlockLabel>
      <div className="border-border flex w-80 flex-row gap-3 rounded-md border p-3">
        <div className="flex-1">
          <FileMetadata filePath={filePath} mimeType={mimeType} />
        </div>

        <Link
          to={base64Data}
          download={`tensorzero_${filePath}`}
          aria-label={`Download ${filePath}`}
        >
          <Download className="h-5 w-5" />
        </Link>

        <Link
          to={blobUrl}
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

interface FileErrorContentBlockProps {
  error?: string;
  actionBar?: ReactNode;
}

/**
 * Renders an error state when file cannot be loaded.
 */
function FileErrorContentBlock({
  error,
  actionBar,
}: FileErrorContentBlockProps) {
  const errorMessage = error || "Failed to retrieve file";

  return (
    <div className="flex flex-col gap-1">
      <ContentBlockLabel
        icon={<FileText className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File
      </ContentBlockLabel>
      <div className="border-border bg-bg-tertiary relative aspect-video w-60 min-w-60 rounded-md border">
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-2">
          <ImageOff className="text-fg-muted h-4 w-4" />
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="text-fg-tertiary line-clamp-2 w-full cursor-default text-center text-xs font-medium break-all">
                {errorMessage}
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-md break-words">
              {errorMessage}
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}

interface UrlFileContentBlockProps {
  block: Extract<File, { file_type: "url" }>;
  actionBar?: ReactNode;
  isEditing?: boolean;
  onChange?: (updatedBlock: File) => void;
}

/**
 * Renders URL-based file content blocks with editing support.
 */
function UrlFileContentBlock({
  block,
  actionBar,
  isEditing,
  onChange,
}: UrlFileContentBlockProps) {
  const handleUrlChange = (url: string) => {
    onChange?.({ ...block, url });
  };

  const handleMimeTypeChange = (mime_type: string) => {
    onChange?.({
      ...block,
      mime_type: mime_type.trim() === "" ? null : mime_type,
    });
  };

  const handleDetailChange = (detail: string) => {
    onChange?.({
      ...block,
      detail: detail === "none" ? undefined : (detail as Detail),
    });
  };

  const handleFilenameChange = (filename: string) => {
    onChange?.({
      ...block,
      filename: filename.trim() === "" ? undefined : filename,
    });
  };

  if (isEditing) {
    return (
      <div className="flex max-w-240 min-w-80 flex-col gap-1">
        <ContentBlockLabel
          icon={<LinkIcon className="text-fg-muted h-3 w-3" />}
          actionBar={actionBar}
        >
          File URL
        </ContentBlockLabel>
        <div className="border-border bg-bg-tertiary/50 flex flex-col gap-2 rounded-sm px-3 py-2">
          <Input
            type="url"
            placeholder="https://example.com/image.png"
            value={block.url}
            onChange={(e) => handleUrlChange(e.target.value)}
            className="text-xs"
          />
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="advanced" className="border-none">
              <AccordionTrigger className="text-fg-tertiary hover:text-fg-secondary [&>svg]:text-fg-tertiary [&:hover>svg]:text-fg-secondary cursor-pointer justify-start gap-1 py-1 text-xs hover:no-underline [&>svg]:order-first [&>svg]:mr-0 [&>svg]:ml-0">
                Advanced
              </AccordionTrigger>
              <AccordionContent className="pb-1">
                <div className="flex flex-col gap-2 px-0.5 pt-0.5">
                  <div className="flex flex-col gap-1">
                    <label className="text-fg-tertiary text-xs">
                      MIME Type
                    </label>
                    <Input
                      type="text"
                      placeholder="image/png"
                      value={block.mime_type ?? ""}
                      onChange={(e) => handleMimeTypeChange(e.target.value)}
                      className="text-xs"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-fg-tertiary text-xs">Detail</label>
                    <Select
                      value={block.detail ?? "none"}
                      onValueChange={handleDetailChange}
                    >
                      <SelectTrigger className="text-xs">
                        <SelectValue placeholder="Select detail level" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">None</SelectItem>
                        <SelectItem value="auto">Auto</SelectItem>
                        <SelectItem value="low">Low</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-fg-tertiary text-xs">Filename</label>
                    <Input
                      type="text"
                      placeholder="image.png"
                      value={block.filename ?? ""}
                      onChange={(e) => handleFilenameChange(e.target.value)}
                      className="text-xs"
                    />
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </div>
    );
  }

  // Read-only view
  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<LinkIcon className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File URL
      </ContentBlockLabel>
      <div className="border-border bg-bg-tertiary/50 rounded-sm px-3 py-2">
        <Link
          to={block.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-fg-primary text-xs break-all underline"
        >
          {block.url || "(empty URL)"}
        </Link>
        {(block.mime_type || block.filename || block.detail) && (
          <div className="text-fg-tertiary mt-1 text-xs">
            {block.filename && <span>{block.filename}</span>}
            {block.filename && block.mime_type && <span> · </span>}
            {block.mime_type && <span>{block.mime_type}</span>}
            {(block.filename || block.mime_type) && block.detail && (
              <span> · </span>
            )}
            {block.detail && <span>detail: {block.detail}</span>}
          </div>
        )}
      </div>
    </div>
  );
}

export interface TruncatedFilenameSegment {
  text: string;
  isMuted: boolean;
}

/**
 * Pure function that calculates how to truncate a filename.
 * Returns segments with styling information for rendering.
 *
 * @param filename - The filename to truncate
 * @param maxLength - Maximum length before truncation (default: 32)
 * @returns Array of segments with text and muted status
 */
export function truncateFilename(
  filename: string,
  maxLength: number = 32,
): TruncatedFilenameSegment[] {
  if (filename.length <= maxLength) {
    return [{ text: filename, isMuted: false }];
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
    return [
      { text: filename.substring(0, maxLength - 3), isMuted: false },
      { text: "...", isMuted: true },
    ];
  }

  const availableLength = maxLength - extension.length - 3; // 3 for "..."
  const frontLength = Math.ceil(availableLength / 2);
  const backLength = Math.floor(availableLength / 2);

  if (name.length <= availableLength) {
    return [{ text: filename, isMuted: false }];
  }

  return [
    { text: name.substring(0, frontLength), isMuted: false },
    { text: "...", isMuted: true },
    {
      text: name.substring(name.length - backLength) + extension,
      isMuted: false,
    },
  ];
}

interface TruncatedFileNameProps {
  filename: string;
  maxLength?: number;
}

/**
 * Truncates long filenames while preserving extension.
 * Always returns JSX.Element for consistent typing.
 */
function TruncatedFileName({
  filename,
  maxLength = 32,
}: TruncatedFileNameProps): React.JSX.Element {
  const segments = truncateFilename(filename, maxLength);

  return (
    <>
      {segments.map((segment, index) => (
        <span
          key={index}
          className={segment.isMuted ? "text-fg-muted" : undefined}
        >
          {segment.text}
        </span>
      ))}
    </>
  );
}

interface FileMetadataProps {
  mimeType: string;
  filePath: string;
}

/**
 * Displays file metadata (name and MIME type).
 */
function FileMetadata({ mimeType, filePath }: FileMetadataProps) {
  return (
    <div className="flex flex-col">
      <div className="text-fg-primary text-sm font-medium" title={filePath}>
        <TruncatedFileName filename={filePath} />
      </div>
      <div className="text-fg-tertiary text-xs">{mimeType}</div>
    </div>
  );
}
