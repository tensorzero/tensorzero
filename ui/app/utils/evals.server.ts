import { spawn } from "node:child_process";
import { z } from "zod";
import { getConfigPath } from "./config/index.server";
/**
 * Get the path to the evals binary from environment variables.
 * Defaults to 'evals' if not specified.
 */
function getEvalsPath(): string {
  return process.env.TENSORZERO_EVALS_PATH || "evals";
}

function getGatewayURL(): string {
  const gatewayURL = process.env.TENSORZERO_GATEWAY_URL;
  // This error is thrown on startup in tensorzero.server.ts
  if (!gatewayURL) {
    throw new Error("TENSORZERO_GATEWAY_URL environment variable is not set");
  }
  return gatewayURL;
}

export function runEval(
  evalName: string,
  variantName: string,
  concurrency: number,
): Promise<EvalStartInfo> {
  const evalsPath = getEvalsPath();
  const gatewayURL = getGatewayURL();
  // Construct the command to run the evals binary
  // Example: evals --gateway-url http://localhost:3000 -n entity_extraction -v llama_8b_initial_prompt -c 10
  const command = [
    evalsPath,
    "--gateway-url",
    gatewayURL,
    "--config-file",
    getConfigPath(),
    "-n",
    evalName,
    "-v",
    variantName,
    "-c",
    concurrency.toString(),
    "--format",
    "jsonl",
  ];

  return new Promise<EvalStartInfo>((resolve, reject) => {
    const child = spawn(command[0], command.slice(1));

    let firstLine = "";
    let dataReceived = false;

    child.stdout.on("data", (data) => {
      if (!dataReceived) {
        const output = data.toString();
        const lines = output.split("\n");
        firstLine = lines[0];
        dataReceived = true;
        resolve(evalStartInfoSchema.parse(JSON.parse(firstLine)));
      }
    });

    child.stderr.on("data", (data) => {
      console.error(`stderr: ${data}`);
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      if (!dataReceived) {
        reject(
          new Error(
            `Process exited with code ${code} without producing output`,
          ),
        );
      }
    });
  });
}

const evalStartInfoSchema = z.object({
  eval_run_id: z.string(),
  num_datapoints: z.number(),
});
export type EvalStartInfo = z.infer<typeof evalStartInfoSchema>;
