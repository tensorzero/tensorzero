#!/usr/bin/env node
/**
 * This script is used to check that all the generated TypeScript files are properly exported in index.ts.
 * It is not production code but should be used in CI and precommit to ensure that we did not miss a type.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const BINDINGS_DIR = __dirname + "/bindings";
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

function getExportedModules() {
  const indexContent = fs.readFileSync(INDEX_FILE, "utf8");
  const exportLines = indexContent
    .split("\n")
    .filter((line) => line.trim().startsWith("export * from"));

  return exportLines
    .map((line) => {
      const match = line.match(/export \* from ["']\.\/(.+?)["']/);
      return match ? match[1] : null;
    })
    .filter(Boolean)
    .sort();
}

function main() {
  try {
    const generatedFiles = getGeneratedFiles();
    const exportedModules = getExportedModules();

    console.log(`Found ${generatedFiles.length} generated TypeScript files`);
    console.log(`Found ${exportedModules.length} exported modules in index.ts`);

    // Find missing exports
    const missingExports = generatedFiles.filter(
      (file) => !exportedModules.includes(file),
    );

    // Find extra exports (shouldn't happen with generated files, but good to check)
    const extraExports = exportedModules.filter(
      (module) => !generatedFiles.includes(module),
    );

    let hasErrors = false;

    if (missingExports.length > 0) {
      console.error("\n❌ Missing exports in index.ts:");
      missingExports.forEach((file) => {
        console.error(`  - ${file}`);
      });
      hasErrors = true;
    }

    if (extraExports.length > 0) {
      console.warn("\n⚠️  Extra exports in index.ts (files not found):");
      extraExports.forEach((module) => {
        console.warn(`  - ${module}`);
      });
      hasErrors = true;
    }

    if (!hasErrors) {
      console.log(
        "\n✅ All generated TypeScript files are properly exported in index.ts",
      );
      process.exit(0);
    } else {
      console.error(
        "\n💡 To fix missing exports, add these lines to index.ts:",
      );
      missingExports.forEach((file) => {
        console.error(`export * from "./${file}";`);
      });
      process.exit(1);
    }
  } catch (error) {
    console.error("Error checking exports:", error.message);
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
