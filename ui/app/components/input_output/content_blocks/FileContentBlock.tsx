import { Link } from "react-router";
import {
  ImageIcon,
  ImageOff,
  Download,
  ExternalLink,
  FileText,
  FileAudio,
  Link as LinkIcon,
  FileCode,
  Upload,
} from "lucide-react";
import { type ReactNode, useRef } from "react";
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
import { Textarea } from "~/components/ui/textarea";
import { Button } from "~/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import type { File, Detail } from "~/types/tensorzero";

interface FileAdvancedAccordionProps {
  filename?: string;
  mimeType?: string | null;
  detail?: Detail;
  isEditing?: boolean;
  onFilenameChange?: (filename: string | undefined) => void;
  onMimeTypeChange?: (mimeType: string) => void;
  onDetailChange?: (detail: Detail | undefined) => void;
}

/**
 * Shared accordion component for advanced file properties.
 * Shows filename, MIME type, and detail level.
 */
function FileAdvancedAccordion({
  filename,
  mimeType,
  detail,
  isEditing,
  onFilenameChange,
  onMimeTypeChange,
  onDetailChange,
}: FileAdvancedAccordionProps) {
  const handleMimeTypeChange = (value: string) => {
    onMimeTypeChange?.(value);
  };

  const handleDetailChange = (value: string) => {
    onDetailChange?.(value === "none" ? undefined : (value as Detail));
  };

  const handleFilenameChange = (value: string) => {
    onFilenameChange?.(value.trim() === "" ? undefined : value);
  };

  if (isEditing) {
    return (
      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="advanced" className="border-none">
          <AccordionTrigger className="text-fg-tertiary hover:text-fg-secondary [&>svg]:text-fg-tertiary [&:hover>svg]:text-fg-secondary cursor-pointer justify-start gap-1 py-1 text-xs hover:no-underline [&>svg]:order-first [&>svg]:mr-0 [&>svg]:ml-0">
            Advanced
          </AccordionTrigger>
          <AccordionContent className="pb-1">
            <div className="flex flex-col gap-2 px-0.5 pt-0.5">
              <div className="flex flex-col gap-1">
                <label className="text-fg-tertiary text-xs">Filename</label>
                <Input
                  type="text"
                  placeholder="image.png"
                  value={filename ?? ""}
                  onChange={(e) => handleFilenameChange(e.target.value)}
                  className="text-xs"
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-fg-tertiary text-xs">MIME Type</label>
                <Input
                  type="text"
                  placeholder="image/png"
                  value={mimeType ?? ""}
                  onChange={(e) => handleMimeTypeChange(e.target.value)}
                  className="text-xs"
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-fg-tertiary text-xs">Detail</label>
                <Select
                  value={detail ?? "none"}
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
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    );
  }

  // Read-only view: show accordion with summary if any values exist
  if (!filename && !mimeType && !detail) {
    return null;
  }

  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="advanced" className="border-none">
        <AccordionTrigger className="text-fg-tertiary hover:text-fg-secondary [&>svg]:text-fg-tertiary [&:hover>svg]:text-fg-secondary cursor-pointer justify-start gap-1 py-1 text-xs hover:no-underline [&>svg]:order-first [&>svg]:mr-0 [&>svg]:ml-0">
          Advanced
        </AccordionTrigger>
        <AccordionContent className="pb-1">
          <div className="flex flex-col gap-1 px-0.5 pt-0.5 text-xs">
            {filename && (
              <div>
                <span className="text-fg-tertiary">Filename:</span>{" "}
                <span className="text-fg-tertiary font-mono text-xs">
                  {filename}
                </span>
              </div>
            )}
            {mimeType && (
              <div>
                <span className="text-fg-tertiary">MIME Type:</span>{" "}
                <span className="text-fg-tertiary font-mono text-xs">
                  {mimeType}
                </span>
              </div>
            )}
            {detail && (
              <div>
                <span className="text-fg-tertiary">Detail:</span>{" "}
                <span className="text-fg-tertiary font-mono text-xs">
                  {detail}
                </span>
              </div>
            )}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

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
      return (
        <Base64FileContentBlock
          block={block}
          actionBar={actionBar}
          isEditing={isEditing}
          onChange={onChange}
        />
      );
    case "object_storage_pointer":
      // TODO: should we handle this better?
      throw new Error(
        "The UI should never receive an object storage pointer. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.",
      );
    case "object_storage_error":
      return (
        <FileErrorContentBlock
          block={block}
          actionBar={actionBar}
          isEditing={isEditing}
          onChange={onChange}
        />
      );
  }

  // If we got here, we know that block.file_type is "object_storage"
  block satisfies Extract<File, { file_type: "object_storage" }>; // compile-time sanity check

  // Determine which component to render based on mime type
  if (block.mime_type?.startsWith("image/")) {
    return (
      <ImageContentBlock
        block={block}
        actionBar={actionBar}
        isEditing={isEditing}
        onChange={onChange}
      />
    );
  }

  if (block.mime_type?.startsWith("audio/")) {
    return (
      <AudioContentBlock
        block={block}
        actionBar={actionBar}
        isEditing={isEditing}
        onChange={onChange}
      />
    );
  }

  return (
    <GenericFileContentBlock
      block={block}
      actionBar={actionBar}
      isEditing={isEditing}
      onChange={onChange}
    />
  );
}

interface ImageContentBlockProps {
  block: Extract<File, { file_type: "object_storage" }>;
  actionBar?: ReactNode;
  isEditing?: boolean;
  onChange?: (updatedBlock: File) => void;
}

/**
 * Renders image files with preview and download link.
 */
function ImageContentBlock({
  block,
  actionBar,
  isEditing,
  onChange,
}: ImageContentBlockProps) {
  return (
    <div className="flex flex-col gap-1">
      <ContentBlockLabel
        icon={<ImageIcon className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File
      </ContentBlockLabel>
      <div className="flex flex-col">
        <Link
          to={block.data}
          target="_blank"
          rel="noopener noreferrer"
          download={`tensorzero_${block.storage_path.path}`}
          className="border-border bg-bg-tertiary text-fg-tertiary flex min-h-20 w-60 items-center justify-center rounded border p-2 text-xs"
        >
          <img src={block.data} alt="Image" />
        </Link>
        <FileAdvancedAccordion
          filename={block.filename}
          mimeType={block.mime_type}
          detail={block.detail}
          isEditing={isEditing}
          onFilenameChange={(filename) => onChange?.({ ...block, filename })}
          onMimeTypeChange={(mime_type) =>
            onChange?.({ ...block, mime_type: mime_type || block.mime_type })
          }
          onDetailChange={(detail) => onChange?.({ ...block, detail })}
        />
      </div>
    </div>
  );
}

interface AudioContentBlockProps {
  block: Extract<File, { file_type: "object_storage" }>;
  actionBar?: ReactNode;
  isEditing?: boolean;
  onChange?: (updatedBlock: File) => void;
}

/**
 * Renders audio files with player controls and metadata.
 * Converts base64 data URL to blob URL for audio playback.
 */
function AudioContentBlock({
  block,
  actionBar,
  isEditing,
  onChange,
}: AudioContentBlockProps) {
  const blobUrl = useBase64UrlToBlobUrl(block.data, block.mime_type);

  return (
    <div className="flex flex-col gap-1">
      <ContentBlockLabel
        icon={<FileAudio className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File
      </ContentBlockLabel>

      <div className="border-border flex w-80 flex-col gap-4 rounded-md border p-3">
        <FileMetadata
          mimeType={block.mime_type}
          filePath={block.storage_path.path}
        />
        <audio controls preload="none" className="w-full">
          <source src={blobUrl} type={block.mime_type} />
        </audio>
        <FileAdvancedAccordion
          filename={block.filename}
          mimeType={block.mime_type}
          detail={block.detail}
          isEditing={isEditing}
          onFilenameChange={(filename) => onChange?.({ ...block, filename })}
          onMimeTypeChange={(mime_type) =>
            onChange?.({ ...block, mime_type: mime_type || block.mime_type })
          }
          onDetailChange={(detail) => onChange?.({ ...block, detail })}
        />
      </div>
    </div>
  );
}

interface GenericFileContentBlockProps {
  block: Extract<File, { file_type: "object_storage" }>;
  actionBar?: ReactNode;
  isEditing?: boolean;
  onChange?: (updatedBlock: File) => void;
}

/**
 * Renders generic files with metadata and download/open actions.
 * Converts base64 data URL to blob URL for browser preview.
 */
function GenericFileContentBlock({
  block,
  actionBar,
  isEditing,
  onChange,
}: GenericFileContentBlockProps) {
  const blobUrl = useBase64UrlToBlobUrl(block.data, block.mime_type);

  return (
    <div className="flex flex-col gap-1">
      <ContentBlockLabel
        icon={<FileText className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File
      </ContentBlockLabel>
      <div className="border-border flex w-80 flex-col gap-3 rounded-md border p-3">
        <div className="flex flex-row gap-3">
          <div className="flex-1">
            <FileMetadata
              filePath={block.storage_path.path}
              mimeType={block.mime_type}
            />
          </div>

          <Link
            to={block.data}
            download={`tensorzero_${block.storage_path.path}`}
            aria-label={`Download ${block.storage_path.path}`}
          >
            <Download className="h-5 w-5" />
          </Link>

          <Link
            to={blobUrl}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={`Open ${block.storage_path.path} in new tab`}
          >
            <ExternalLink className="h-5 w-5" />
          </Link>
        </div>
        <FileAdvancedAccordion
          filename={block.filename}
          mimeType={block.mime_type}
          detail={block.detail}
          isEditing={isEditing}
          onFilenameChange={(filename) => onChange?.({ ...block, filename })}
          onMimeTypeChange={(mime_type) =>
            onChange?.({ ...block, mime_type: mime_type ?? block.mime_type })
          }
          onDetailChange={(detail) => onChange?.({ ...block, detail })}
        />
      </div>
    </div>
  );
}

interface FileErrorContentBlockProps {
  block: Extract<File, { file_type: "object_storage_error" }>;
  actionBar?: ReactNode;
  isEditing?: boolean;
  onChange?: (updatedBlock: File) => void;
}

/**
 * Renders an error state when file cannot be loaded.
 */
function FileErrorContentBlock({
  block,
  actionBar,
  isEditing,
  onChange,
}: FileErrorContentBlockProps) {
  const errorMessage = block.error || "Failed to retrieve file";

  return (
    <div className="flex flex-col gap-1">
      <ContentBlockLabel
        icon={<FileText className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File
      </ContentBlockLabel>
      <div className="flex flex-col">
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
        <FileAdvancedAccordion
          filename={block.filename}
          mimeType={block.mime_type}
          detail={block.detail}
          isEditing={isEditing}
          onFilenameChange={(filename) => onChange?.({ ...block, filename })}
          onMimeTypeChange={(mime_type) =>
            onChange?.({ ...block, mime_type: mime_type ?? block.mime_type })
          }
          onDetailChange={(detail) => onChange?.({ ...block, detail })}
        />
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
          <FileAdvancedAccordion
            filename={block.filename}
            mimeType={block.mime_type}
            detail={block.detail}
            isEditing={true}
            onFilenameChange={(filename) => onChange?.({ ...block, filename })}
            onMimeTypeChange={(mime_type) =>
              onChange?.({ ...block, mime_type: mime_type || null })
            }
            onDetailChange={(detail) => onChange?.({ ...block, detail })}
          />
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
        <FileAdvancedAccordion
          filename={block.filename}
          mimeType={block.mime_type}
          detail={block.detail}
        />
      </div>
    </div>
  );
}

interface Base64FileContentBlockProps {
  block: Extract<File, { file_type: "base64" }>;
  actionBar?: ReactNode;
  isEditing?: boolean;
  onChange?: (updatedBlock: File) => void;
}

/**
 * Renders base64-encoded file content blocks with editing support.
 */
function Base64FileContentBlock({
  block,
  actionBar,
  isEditing,
  onChange,
}: Base64FileContentBlockProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDataChange = (data: string) => {
    onChange?.({ ...block, data });
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // result is like "data:image/png;base64,xxxxx"
      // Extract just the base64 part
      const base64Data = result.split(",")[1] ?? "";
      onChange?.({
        ...block,
        data: base64Data,
        mime_type: file.type || block.mime_type,
        filename: block.filename || file.name,
      });
    };
    reader.readAsDataURL(file);
    // Reset input so the same file can be selected again
    e.target.value = "";
  };

  if (isEditing) {
    return (
      <div className="flex max-w-240 min-w-80 flex-col gap-1">
        <ContentBlockLabel
          icon={<FileCode className="text-fg-muted h-3 w-3" />}
          actionBar={actionBar}
        >
          File (Base64)
        </ContentBlockLabel>
        <div className="border-border bg-bg-tertiary/50 flex flex-col gap-2 rounded-sm px-3 py-2">
          <div className="flex items-center gap-2">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={handleFileSelect}
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              className="text-xs"
            >
              <Upload className="mr-1 h-3 w-3" />
              {block.data ? "Replace File" : "Choose File"}
            </Button>
            {block.filename && (
              <span className="text-fg-tertiary text-xs">{block.filename}</span>
            )}
          </div>
          <Textarea
            placeholder="Or paste base64-encoded file data..."
            value={block.data}
            onChange={(e) => handleDataChange(e.target.value)}
            className="min-h-20 font-mono text-xs"
          />
          <FileAdvancedAccordion
            filename={block.filename}
            mimeType={block.mime_type}
            detail={block.detail}
            isEditing={true}
            onFilenameChange={(filename) => onChange?.({ ...block, filename })}
            onMimeTypeChange={(mime_type) =>
              onChange?.({ ...block, mime_type: mime_type ?? block.mime_type })
            }
            onDetailChange={(detail) => onChange?.({ ...block, detail })}
          />
        </div>
      </div>
    );
  }

  // Read-only view - show truncated base64 data
  const truncatedData =
    block.data.length > 100 ? `${block.data.slice(0, 100)}...` : block.data;

  return (
    <div className="flex max-w-240 min-w-80 flex-col gap-1">
      <ContentBlockLabel
        icon={<FileCode className="text-fg-muted h-3 w-3" />}
        actionBar={actionBar}
      >
        File (Base64)
      </ContentBlockLabel>
      <div className="border-border bg-bg-tertiary/50 rounded-sm px-3 py-2">
        <div className="text-fg-tertiary font-mono text-xs break-all">
          {truncatedData || "(empty)"}
        </div>
        <FileAdvancedAccordion
          filename={block.filename}
          mimeType={block.mime_type}
          detail={block.detail}
        />
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
