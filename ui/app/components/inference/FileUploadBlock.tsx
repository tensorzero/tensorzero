import React, { useState, useRef } from "react";
import { Upload, X, FileText, Image as ImageIcon, FileAudio } from "lucide-react";
import type { ResolvedFileContent } from "~/utils/clickhouse/common";
import ImageBlock from "./ImageBlock";
import { SkeletonImage } from "./SkeletonImage";

interface FileUploadBlockProps {
  block: ResolvedFileContent;
  isEditing?: boolean;
  onContentChange?: (block: ResolvedFileContent) => void;
}

export function FileUploadBlock({
  block,
  isEditing,
  onContentChange,
}: FileUploadBlockProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !onContentChange) return;

    setIsUploading(true);
    try {
      const dataUrl = await fileToDataUrl(file);
      
      const updatedBlock: ResolvedFileContent = {
        type: "file",
        file: {
          dataUrl,
          mime_type: file.type,
        },
        storage_path: {
          kind: {
            type: "filesystem",
            path: "local_uploads",
          },
          path: `uploads/${file.name}`,
        },
      };

      onContentChange(updatedBlock);
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveFile = () => {
    if (onContentChange) {
      // Signal that this block should be removed by setting a special flag
      const updatedBlock: ResolvedFileContent & { _remove?: boolean } = {
        type: "file",
        file: {
          dataUrl: "",
          mime_type: "",
        },
        storage_path: {
          kind: {
            type: "filesystem",
            path: "local_uploads",
          },
          path: "",
        },
        _remove: true,
      };
      onContentChange(updatedBlock);
    }
  };

  const fileToDataUrl = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        resolve(reader.result as string);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const getFileIcon = (mimeType: string) => {
    if (mimeType.startsWith("image/")) {
      return <ImageIcon className="h-4 w-4" />;
    } else if (mimeType.startsWith("audio/")) {
      return <FileAudio className="h-4 w-4" />;
    } else {
      return <FileText className="h-4 w-4" />;
    }
  };

  if (isEditing) {
    return (
      <div className="w-full">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
            File
          </span>
          {block.file.dataUrl && (
            <button
              onClick={handleRemoveFile}
              className="text-red-500 hover:text-red-700"
              type="button"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        
        {block.file.dataUrl ? (
          <div className="border border-slate-300 rounded p-3 bg-slate-50 dark:bg-slate-800 dark:border-slate-600">
            <div className="flex items-center gap-2 mb-2">
              {getFileIcon(block.file.mime_type)}
              <span className="text-sm font-medium">
                {block.storage_path.path.split("/").pop() || "Uploaded file"}
              </span>
            </div>
            {block.file.mime_type.startsWith("image/") ? (
              <ImageBlock image={block} />
            ) : (
              <div className="text-sm text-slate-600 dark:text-slate-400">
                {block.file.mime_type}
              </div>
            )}
          </div>
        ) : (
          <div className="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center hover:border-slate-400 transition-colors dark:border-slate-600 dark:hover:border-slate-500">
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileUpload}
              className="hidden"
              accept="image/*,audio/*,.pdf,.txt,.doc,.docx"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="flex flex-col items-center gap-2 text-slate-600 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
              type="button"
            >
              <Upload className="h-8 w-8" />
              <span className="text-sm">
                {isUploading ? "Uploading..." : "Click to upload a file"}
              </span>
            </button>
          </div>
        )}
      </div>
    );
  }

  // Non-editing mode - display the file
  if (block.file.mime_type.startsWith("image/")) {
    return <ImageBlock image={block} />;
  } else {
    return (
      <div>
        <SkeletonImage
          error={`Unsupported file type: ${block.file.mime_type}`}
        />
      </div>
    );
  }
} 