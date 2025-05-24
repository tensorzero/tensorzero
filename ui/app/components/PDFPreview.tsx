import React, { useEffect, useRef } from "react";
import * as pdfjsLib from "pdfjs-dist";

interface PDFPreviewProps {
  pdfUrl: string;
}

export const PDFPreview: React.FC<PDFPreviewProps> = ({ pdfUrl }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    pdfjsLib.getDocument(pdfUrl).promise.then((pdf) => {
      pdf.getPage(1).then((page) => {
        const viewport = page.getViewport({ scale: 1.0 });
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.height = viewport.height;
          canvas.width = viewport.width;
          page.render({ canvasContext: canvas.getContext("2d")!, viewport });
        }
      });
    });
  }, [pdfUrl]);

  return <canvas ref={canvasRef} />;
};
