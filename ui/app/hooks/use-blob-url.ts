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

// Generate a Blob URL from base64 or data URL
export function useBase64UrlToBlobUrl(url: string, mimeType: string) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);

  useEffect(() => {
    let base64: string | null = null;
    if (url.startsWith("data:")) {
      const match = url.match(/^data:(.*?);base64,(.*)$/);
      if (match) {
        base64 = match[2];
      }
    } else {
      base64 = url;
    }

    let blobUrl: string | null = null;
    if (base64) {
      const blob = base64ToBlob(base64, mimeType);
      blobUrl = URL.createObjectURL(blob);
      setObjectUrl(blobUrl);
    } else {
      setObjectUrl(url);
    }

    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [url, mimeType]);

  return objectUrl ?? url;
}
