import { useEffect, useState } from "react";

function base64ToBlob(base64: string, mime: string) {
  const byteCharacters = atob(base64);
  const byteNumbers = Array.from<number>({ length: byteCharacters.length });
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mime });
}

export function toDisplayUrl(data: string, mimeType: string) {
  if (data.startsWith("data:") || data.startsWith("blob:")) {
    return data;
  }
  return `data:${mimeType};base64,${data}`;
}

// Generate a Blob URL from raw base64 or return a URL-safe representation immediately.
export function useBase64UrlToBlobUrl(url: string, mimeType: string) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const displayUrl = toDisplayUrl(url, mimeType);

  useEffect(() => {
    let base64: string | null = null;
    if (displayUrl.startsWith("data:")) {
      const match = displayUrl.match(/^data:(.*?);base64,(.*)$/);
      if (match) {
        base64 = match[2];
      }
    } else {
      base64 = displayUrl;
    }

    let blobUrl: string | null = null;
    if (base64) {
      const blob = base64ToBlob(base64, mimeType);
      blobUrl = URL.createObjectURL(blob);
      setObjectUrl(blobUrl);
    } else {
      setObjectUrl(displayUrl);
    }

    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [displayUrl, mimeType]);

  return objectUrl ?? displayUrl;
}
