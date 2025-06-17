import { logger } from "~/utils/logger";

export const FF_ENABLE_PYTHON =
  process.env.TENSORZERO_UI_FF_ENABLE_PYTHON === "1";

logger.info("FF_ENABLE_PYTHON: " + FF_ENABLE_PYTHON);
