import { PDFPreview } from "./PDFPreview";

interface InferenceResultProps {
  result: { pdf?: { url: string; file_name: string } };
}

export const InferenceResult: React.FC<InferenceResultProps> = ({ result }) => {
  return (
    <div>
      {result.pdf && (
        <>
          <PDFPreview pdfUrl={result.pdf.url} />
          <a href={result.pdf.url} download={result.pdf.file_name}>
            Download PDF
          </a>
        </>
      )}
    </div>
  );
};
