import type {
  CumulativeFeedbackTimeSeriesPoint,
  TimeWindow,
} from "tensorzero-node";
import { addPeriod, normalizePeriod } from "~/utils/date";

type FeedbackTimeseriesPointByVariant = {
  count: number;
  mean: number | null;
  cs_lower: number | null;
  cs_upper: number | null;
};

export type FeedbackCountsTimeseriesData = {
  date: string;
  [key: string]: string | number;
};

export type FeedbackMeansTimeseriesData = {
  date: string;
  [key: string]: string | number | null;
};

export function transformFeedbackTimeseries(
  parsedRows: CumulativeFeedbackTimeSeriesPoint[],
  timeGranularity: TimeWindow,
): {
  countsData: FeedbackCountsTimeseriesData[];
  meansData: FeedbackMeansTimeseriesData[];
  variantNames: string[];
} {
  const variantNames = [...new Set(parsedRows.map((row) => row.variant_name))];

  // If no data, return empty
  if (parsedRows.length === 0) {
    return { countsData: [], meansData: [], variantNames: [] };
  }

  // Group by date
  const groupedByDate = parsedRows.reduce<
    Record<string, Record<string, FeedbackTimeseriesPointByVariant>>
  >((acc, row) => {
    const { period_end, variant_name, count, mean, cs_lower, cs_upper } = row;

    if (!acc[period_end]) {
      acc[period_end] = {};
    }

    const sanitizedCount = Number(count);
    const sanitizedMean =
      mean === null || mean === undefined ? null : Number(mean);
    const sanitizedLower =
      cs_lower === null || cs_lower === undefined ? null : Number(cs_lower);
    const sanitizedUpper =
      cs_upper === null || cs_upper === undefined ? null : Number(cs_upper);

    acc[period_end][variant_name] = {
      count: Number.isFinite(sanitizedCount) ? sanitizedCount : 0,
      mean:
        sanitizedMean !== null && Number.isFinite(sanitizedMean)
          ? sanitizedMean
          : null,
      cs_lower:
        sanitizedLower !== null && Number.isFinite(sanitizedLower)
          ? sanitizedLower
          : null,
      cs_upper:
        sanitizedUpper !== null && Number.isFinite(sanitizedUpper)
          ? sanitizedUpper
          : null,
    };
    return acc;
  }, {});

  // Get all unique periods from the data, sorted chronologically
  const allPeriods = Object.keys(groupedByDate).sort();

  // Fill in missing periods between the ones we have from ClickHouse
  const filledPeriods: string[] = [];
  for (let i = 0; i < allPeriods.length; i++) {
    const currentPeriod = allPeriods[i];
    filledPeriods.push(currentPeriod);

    // Check if there's a next period to compare against
    if (i < allPeriods.length - 1) {
      const nextPeriod = allPeriods[i + 1];
      let current = new Date(currentPeriod);
      const next = new Date(nextPeriod);

      // Fill in any missing periods between current and next
      while (true) {
        const nextPeriodStr = addPeriod(current, timeGranularity);
        current = new Date(nextPeriodStr);
        if (current.getTime() >= next.getTime()) break;
        filledPeriods.push(nextPeriodStr);
      }
    }
  }

  // If we have fewer than 10 periods, add more going backwards from the earliest
  if (filledPeriods.length < 10) {
    const earliestPeriod = new Date(filledPeriods[0]);
    const periodsToAdd = 10 - filledPeriods.length;
    const additionalPeriods: string[] = [];

    for (let i = 1; i <= periodsToAdd; i++) {
      const period = new Date(earliestPeriod);
      switch (timeGranularity) {
        case "minute":
          period.setUTCMinutes(earliestPeriod.getUTCMinutes() - i);
          break;
        case "hour":
          period.setUTCHours(earliestPeriod.getUTCHours() - i);
          break;
        case "day":
          period.setUTCDate(earliestPeriod.getUTCDate() - i);
          break;
        case "week":
          period.setUTCDate(earliestPeriod.getUTCDate() - i * 7);
          break;
        case "month":
          period.setUTCMonth(earliestPeriod.getUTCMonth() - i);
          break;
        case "cumulative":
          // No period subtraction for cumulative
          break;
      }
      const normalized = normalizePeriod(period, timeGranularity);
      additionalPeriods.unshift(normalized.toISOString());
    }

    filledPeriods.unshift(...additionalPeriods);
  }

  // Take only the last 10 periods
  const periodsToShow = filledPeriods.slice(-10);

  // Initialize forward-filled stats
  const lastKnownCounts: Record<string, number> = {};
  const lastKnownMeans: Record<string, number | null> = {};
  const lastKnownLower: Record<string, number | null> = {};
  const lastKnownUpper: Record<string, number | null> = {};
  variantNames.forEach((variant) => {
    lastKnownCounts[variant] = 0;
    lastKnownMeans[variant] = null;
    lastKnownLower[variant] = null;
    lastKnownUpper[variant] = null;
  });

  const countsData: FeedbackCountsTimeseriesData[] = [];
  const meansData: FeedbackMeansTimeseriesData[] = [];

  periodsToShow.forEach((period) => {
    const countsRow: FeedbackCountsTimeseriesData = { date: period };
    const meansRow: FeedbackMeansTimeseriesData = { date: period };

    variantNames.forEach((variant) => {
      const periodData = groupedByDate[period]?.[variant];

      if (periodData) {
        if (Number.isFinite(periodData.count)) {
          lastKnownCounts[variant] = periodData.count;
        }

        if (periodData.mean !== null) {
          lastKnownMeans[variant] = periodData.mean;
        }

        if (
          periodData.cs_lower !== null &&
          periodData.cs_upper !== null &&
          Number.isFinite(periodData.cs_lower) &&
          Number.isFinite(periodData.cs_upper)
        ) {
          lastKnownLower[variant] = periodData.cs_lower;
          lastKnownUpper[variant] = periodData.cs_upper;
        } else {
          lastKnownLower[variant] = null;
          lastKnownUpper[variant] = null;
        }
      }

      countsRow[variant] = lastKnownCounts[variant];

      const mean = lastKnownMeans[variant];
      const lower = lastKnownLower[variant];
      const upper = lastKnownUpper[variant];
      const width = lower !== null && upper !== null ? upper - lower : null;

      meansRow[variant] = mean ?? null;
      meansRow[`${variant}_cs_lower`] = lower ?? null;
      meansRow[`${variant}_cs_upper`] = upper ?? null;
      meansRow[`${variant}_cs_width`] = width ?? null;
    });

    countsData.push(countsRow);
    meansData.push(meansRow);
  });

  return {
    countsData,
    meansData,
    variantNames,
  };
}
