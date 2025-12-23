import { describe, it, expect } from "vitest";
import {
  truncateFilename,
  type TruncatedFilenameSegment,
} from "./FileContentBlock";

describe("truncateFilename", () => {
  describe("short filenames (no truncation)", () => {
    it("should return filename as-is when shorter than maxLength", () => {
      const result = truncateFilename("short.txt", 32);
      expect(result).toEqual([{ text: "short.txt", isMuted: false }]);
    });

    it("should return filename as-is when equal to maxLength", () => {
      const filename = "a".repeat(32);
      const result = truncateFilename(filename, 32);
      expect(result).toEqual([{ text: filename, isMuted: false }]);
      expect(result[0].text.length).toBe(32);
    });

    it("should handle filenames without extensions", () => {
      const result = truncateFilename("noextension", 32);
      expect(result).toEqual([{ text: "noextension", isMuted: false }]);
    });
  });

  describe("long filenames with extensions", () => {
    it("should truncate filename in the middle while preserving extension", () => {
      const result = truncateFilename(
        "very_long_filename_that_needs_truncation.txt",
        32,
      );

      // Should have 3 segments: front, ellipsis, back+extension
      expect(result).toHaveLength(3);
      expect(result[0].isMuted).toBe(false);
      expect(result[1].text).toBe("...");
      expect(result[1].isMuted).toBe(true);
      expect(result[2].isMuted).toBe(false);
      expect(result[2].text).toMatch(/\.txt$/); // Should end with extension

      // Total displayed length should be <= maxLength
      const totalLength = result.reduce((sum, seg) => sum + seg.text.length, 0);
      expect(totalLength).toBeLessThanOrEqual(32);
    });

    it("should handle long filenames with longer extensions", () => {
      const result = truncateFilename(
        "document_with_very_long_name.component.tsx",
        32,
      );

      expect(result).toHaveLength(3);
      expect(result[1].text).toBe("...");
      expect(result[2].text).toMatch(/\.component\.tsx$/);

      const totalLength = result.reduce((sum, seg) => sum + seg.text.length, 0);
      expect(totalLength).toBeLessThanOrEqual(32);
    });

    it("should distribute truncation evenly between front and back", () => {
      const result = truncateFilename(
        "abcdefghijklmnopqrstuvwxyz123456.txt",
        32,
      );

      expect(result).toHaveLength(3);

      // With extension ".txt" (4 chars) and "..." (3 chars), we have 32 - 4 - 3 = 25 chars available
      // Front should get ceil(25/2) = 13, back should get floor(25/2) = 12
      const nameWithoutExt = "abcdefghijklmnopqrstuvwxyz123456";
      expect(result[0].text).toBe(nameWithoutExt.substring(0, 13));
      expect(result[2].text).toBe(
        nameWithoutExt.substring(nameWithoutExt.length - 12) + ".txt",
      );
    });
  });

  describe("filenames without extensions", () => {
    it("should truncate in middle when no extension exists", () => {
      const result = truncateFilename(
        "very_long_filename_without_any_extension_at_all",
        32,
      );

      // Since there's no extension, it treats the whole thing as name
      // and does middle truncation with no extension to preserve
      expect(result).toHaveLength(3);
      expect(result[1].text).toBe("...");
      expect(result[1].isMuted).toBe(true);

      const totalLength = result.reduce((sum, seg) => sum + seg.text.length, 0);
      expect(totalLength).toBeLessThanOrEqual(32);
    });

    it("should handle files starting with a dot (hidden files)", () => {
      const result = truncateFilename(".gitignore", 32);
      expect(result).toEqual([{ text: ".gitignore", isMuted: false }]);
    });

    it("should not treat leading dot as extension", () => {
      const result = truncateFilename(
        ".very_long_hidden_filename_that_needs_truncation",
        32,
      );

      // No extension found (lastIndexOf returns 0, not > 0)
      // Does middle truncation
      expect(result).toHaveLength(3);
      expect(result[1].text).toBe("...");

      const totalLength = result.reduce((sum, seg) => sum + seg.text.length, 0);
      expect(totalLength).toBeLessThanOrEqual(32);
    });
  });

  describe("very long extensions", () => {
    it("should truncate from the end when extension is very long", () => {
      const result = truncateFilename(
        "file.verylongextensionthatisgreaterthanmaxlength",
        32,
      );

      // Extension length (49) >= maxLength - 3 (29), so truncate from end
      expect(result).toHaveLength(2);
      expect(result[0].text).toHaveLength(29);
      expect(result[1].text).toBe("...");
      expect(result[1].isMuted).toBe(true);
    });

    it("should handle edge case where extension is exactly maxLength - 3", () => {
      // Extension is 28 chars (32 total - 4 for "file"), which equals maxLength (32) - 3 (29)
      // Actually the whole filename is exactly 32 chars, so no truncation
      const result = truncateFilename("file.extensionisveryverylongabc", 32);

      // Since total length is 32, it should return as-is
      expect(result).toEqual([
        { text: "file.extensionisveryverylongabc", isMuted: false },
      ]);
    });
  });

  describe("custom maxLength values", () => {
    it("should respect custom maxLength parameter", () => {
      const result = truncateFilename("short_filename.txt", 10);

      expect(result).toHaveLength(3);

      const totalLength = result.reduce((sum, seg) => sum + seg.text.length, 0);
      expect(totalLength).toBeLessThanOrEqual(10);
    });

    it("should handle very small maxLength values", () => {
      const result = truncateFilename("filename.txt", 8);

      const totalLength = result.reduce((sum, seg) => sum + seg.text.length, 0);
      expect(totalLength).toBeLessThanOrEqual(8);
      expect(result.some((seg) => seg.text === "...")).toBe(true);
    });

    it("should handle large maxLength values (no truncation)", () => {
      const result = truncateFilename("filename.txt", 100);
      expect(result).toEqual([{ text: "filename.txt", isMuted: false }]);
    });

    it("should use default maxLength of 32 when not specified", () => {
      const longFilename = "a".repeat(40) + ".txt";
      const result = truncateFilename(longFilename);

      const totalLength = result.reduce((sum, seg) => sum + seg.text.length, 0);
      expect(totalLength).toBeLessThanOrEqual(32);
    });
  });

  describe("edge cases", () => {
    it("should handle empty string", () => {
      const result = truncateFilename("", 32);
      expect(result).toEqual([{ text: "", isMuted: false }]);
    });

    it("should handle single character", () => {
      const result = truncateFilename("a", 32);
      expect(result).toEqual([{ text: "a", isMuted: false }]);
    });

    it("should handle filename with only extension", () => {
      const result = truncateFilename(".txt", 32);
      expect(result).toEqual([{ text: ".txt", isMuted: false }]);
    });

    it("should handle filename with multiple dots", () => {
      const result = truncateFilename("archive.tar.gz", 32);
      // Last dot is at position 11, so extension is ".gz"
      expect(result).toEqual([{ text: "archive.tar.gz", isMuted: false }]);
    });

    it("should handle long filename with multiple dots", () => {
      const result = truncateFilename(
        "very.long.filename.with.many.dots.and.extensions.tar.gz",
        32,
      );

      expect(result).toHaveLength(3);
      expect(result[1].text).toBe("...");
      expect(result[2].text).toMatch(/\.gz$/);

      const totalLength = result.reduce((sum, seg) => sum + seg.text.length, 0);
      expect(totalLength).toBeLessThanOrEqual(32);
    });

    it("should handle filename ending with dot", () => {
      const result = truncateFilename("filename.", 32);
      // lastIndexOf(".") returns 8 > 0, extension is ""
      expect(result).toEqual([{ text: "filename.", isMuted: false }]);
    });

    it("should maintain segment structure for rendering", () => {
      const result = truncateFilename("very_long_filename_for_testing.txt", 20);

      // Verify all segments have required properties
      result.forEach((segment: TruncatedFilenameSegment) => {
        expect(segment).toHaveProperty("text");
        expect(segment).toHaveProperty("isMuted");
        expect(typeof segment.text).toBe("string");
        expect(typeof segment.isMuted).toBe("boolean");
      });
    });
  });

  describe("consistency checks", () => {
    it("should never exceed maxLength in total segment length", () => {
      const testCases = [
        { filename: "short.txt", maxLength: 10 },
        { filename: "medium_filename.doc", maxLength: 15 },
        { filename: "very_long_filename_that_exceeds.pdf", maxLength: 20 },
        {
          filename: "another_extremely_long_name.component.tsx",
          maxLength: 25,
        },
        { filename: "noextension_but_very_long_name", maxLength: 18 },
      ];

      testCases.forEach(({ filename, maxLength }) => {
        const result = truncateFilename(filename, maxLength);
        const totalLength = result.reduce(
          (sum, seg) => sum + seg.text.length,
          0,
        );
        expect(totalLength).toBeLessThanOrEqual(maxLength);
      });
    });

    it("should always include ellipsis when truncating", () => {
      const testCases = [
        "very_long_filename.txt",
        "another_long_name_here.pdf",
        "no_extension_but_very_long",
      ];

      testCases.forEach((filename) => {
        const result = truncateFilename(filename, 15);
        if (result.length > 1) {
          expect(result.some((seg) => seg.text === "...")).toBe(true);
        }
      });
    });

    it("should preserve extension when present and reasonable", () => {
      const testCases = [
        { basename: "longname", ext: ".txt" },
        { basename: "longname", ext: ".pdf" },
        { basename: "longname", ext: ".component.tsx" },
      ];

      testCases.forEach(({ basename, ext }) => {
        const filename = basename + "x".repeat(50) + ext;
        const result = truncateFilename(filename, 25);
        const fullText = result.map((seg) => seg.text).join("");
        expect(fullText.endsWith(ext)).toBe(true);
      });
    });
  });
});
