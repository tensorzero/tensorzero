#!/usr/bin/env node
/**
 * This script is used to generate the index.ts file from the generated TypeScript files.
 * It is not production code but should be used in CI and precommit to ensure that we did not miss a type.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const BINDINGS_DIR = path.join(__dirname, "bindings");
const INDEX_FILE = path.join(BINDINGS_DIR, "index.ts");

function getGeneratedFiles() {
  const files = [];

  function scanDirectory(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        scanDirectory(fullPath);
      } else if (
        entry.isFile() &&
        entry.name.endsWith(".ts") &&
        entry.name !== "index.ts"
      ) {
        // Convert to relative path from bindings directory and remove .ts extension
        const relativePath = path.relative(BINDINGS_DIR, fullPath);
        const moduleName = relativePath.replace(/\.ts$/, "");
        files.push(moduleName);
      }
    }
  }

  scanDirectory(BINDINGS_DIR);
  return files.sort();
}

export function generateIndex() {
  try {
    const generatedFiles = getGeneratedFiles();

    console.log(`Found ${generatedFiles.length} generated TypeScript files`);

    // Generate the index.ts content
    const indexContent =
      generatedFiles.map((file) => `export * from "./${file}";`).join("\n") +
      "\n";

    // Write the index.ts file
    fs.writeFileSync(INDEX_FILE, indexContent, "utf8");

    console.log(`✅ Generated index.ts with ${generatedFiles.length} exports`);
  } catch (error) {
    console.error("Error generating index.ts:", error.message);
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  generateIndex();
}
