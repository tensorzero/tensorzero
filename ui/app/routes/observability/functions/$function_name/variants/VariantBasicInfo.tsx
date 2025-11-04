import type {
  VariantConfig,
  BestOfNSamplingConfig,
  DiclConfig,
  MixtureOfNConfig,
  FunctionConfig,
} from "~/types/tensorzero";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import Chip from "~/components/ui/Chip";
import { getFunctionTypeIcon } from "~/utils/icon";
import { toFunctionUrl, toVariantUrl } from "~/utils/urls";

interface BasicInfoProps {
  variantConfig: VariantConfig;
  function_name: string;
  function_type: FunctionConfig["type"];
}

export default function BasicInfo({
  variantConfig,
  function_name,
  function_type,
}: BasicInfoProps) {
  // Get function icon and background
  const functionIconConfig = getFunctionTypeIcon(function_type);

  return (
    <BasicInfoLayout>
      {/* Type */}
      <BasicInfoItem>
        <BasicInfoItemTitle>Type</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip label={variantConfig.type} font="mono" />
        </BasicInfoItemContent>
      </BasicInfoItem>

      {/* Function */}
      <BasicInfoItem>
        <BasicInfoItemTitle>Function</BasicInfoItemTitle>
        <BasicInfoItemContent>
          <Chip
            icon={functionIconConfig.icon}
            iconBg={functionIconConfig.iconBg}
            label={function_name}
            secondaryLabel={function_type}
            link={toFunctionUrl(function_name)}
            font="mono"
          />
        </BasicInfoItemContent>
      </BasicInfoItem>

      {/* --- CHAT COMPLETION --- */}
      {variantConfig.type === "chat_completion" && (
        <>
          <BasicInfoItem>
            <BasicInfoItemTitle>Model</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={variantConfig.model} font="mono" />
            </BasicInfoItemContent>
          </BasicInfoItem>

          {variantConfig.temperature && (
            <BasicInfoItem>
              <BasicInfoItemTitle>Temperature</BasicInfoItemTitle>
              <BasicInfoItemContent>
                <Chip label={variantConfig.temperature.toString()} />
              </BasicInfoItemContent>
            </BasicInfoItem>
          )}

          {variantConfig.top_p && (
            <BasicInfoItem>
              <BasicInfoItemTitle>Top P</BasicInfoItemTitle>
              <BasicInfoItemContent>
                <Chip label={variantConfig.top_p.toString()} />
              </BasicInfoItemContent>
            </BasicInfoItem>
          )}

          {variantConfig.max_tokens && (
            <BasicInfoItem>
              <BasicInfoItemTitle>Max Tokens</BasicInfoItemTitle>
              <BasicInfoItemContent>
                <Chip label={variantConfig.max_tokens.toString()} />
              </BasicInfoItemContent>
            </BasicInfoItem>
          )}

          {variantConfig.presence_penalty && (
            <BasicInfoItem>
              <BasicInfoItemTitle>Presence Penalty</BasicInfoItemTitle>
              <BasicInfoItemContent>
                <Chip label={variantConfig.presence_penalty.toString()} />
              </BasicInfoItemContent>
            </BasicInfoItem>
          )}

          {variantConfig.frequency_penalty && (
            <BasicInfoItem>
              <BasicInfoItemTitle>Frequency Penalty</BasicInfoItemTitle>
              <BasicInfoItemContent>
                <Chip label={variantConfig.frequency_penalty.toString()} />
              </BasicInfoItemContent>
            </BasicInfoItem>
          )}

          {variantConfig.seed && (
            <BasicInfoItem>
              <BasicInfoItemTitle>Seed</BasicInfoItemTitle>
              <BasicInfoItemContent>
                <Chip label={variantConfig.seed.toString()} font="mono" />
              </BasicInfoItemContent>
            </BasicInfoItem>
          )}
        </>
      )}

      {/* --- BEST OF N SAMPLING --- */}
      {variantConfig.type === "best_of_n_sampling" &&
        (() => {
          const config = variantConfig as BestOfNSamplingConfig;
          return (
            <>
              <BasicInfoItem>
                <BasicInfoItemTitle>Model (Evaluator)</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <Chip label={config.evaluator.model} font="mono" />
                </BasicInfoItemContent>
              </BasicInfoItem>

              {config.evaluator.temperature && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Temperature</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.evaluator.temperature.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.evaluator.top_p && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Top P</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.evaluator.top_p.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.evaluator.max_tokens && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Max Tokens</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.evaluator.max_tokens.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.evaluator.presence_penalty && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Presence Penalty</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip
                      label={config.evaluator.presence_penalty.toString()}
                    />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.evaluator.frequency_penalty && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Frequency Penalty</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip
                      label={config.evaluator.frequency_penalty.toString()}
                    />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.evaluator.seed && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Seed</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip
                      label={config.evaluator.seed.toString()}
                      font="mono"
                    />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              <BasicInfoItem>
                <BasicInfoItemTitle>Timeout</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <Chip label={`${config.timeout_s}s`} />
                </BasicInfoItemContent>
              </BasicInfoItem>

              <BasicInfoItem>
                <BasicInfoItemTitle>Candidates</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <div className="flex flex-wrap gap-1">
                    {config.candidates.map((candidate) => (
                      <Chip
                        key={candidate}
                        label={candidate}
                        link={toVariantUrl(function_name, candidate)}
                        font="mono"
                      />
                    ))}
                  </div>
                </BasicInfoItemContent>
              </BasicInfoItem>
            </>
          );
        })()}

      {/* --- DYNAMIC IN-CONTEXT LEARNING --- */}
      {variantConfig.type === "dicl" &&
        (() => {
          const config = variantConfig as DiclConfig;
          return (
            <>
              <BasicInfoItem>
                <BasicInfoItemTitle>Model</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <Chip label={config.model} font="mono" />
                </BasicInfoItemContent>
              </BasicInfoItem>

              <BasicInfoItem>
                <BasicInfoItemTitle>Embedding Model</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <Chip label={config.embedding_model} font="mono" />
                </BasicInfoItemContent>
              </BasicInfoItem>

              <BasicInfoItem>
                <BasicInfoItemTitle>k (Neighbors)</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <Chip label={config.k.toString()} />
                </BasicInfoItemContent>
              </BasicInfoItem>

              <BasicInfoItem>
                <BasicInfoItemTitle>Max Distance</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <Chip
                    label={
                      config.max_distance !== null
                        ? config.max_distance.toString()
                        : "-"
                    }
                  />
                </BasicInfoItemContent>
              </BasicInfoItem>

              {config.temperature && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Temperature</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.temperature.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.top_p && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Top P</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.top_p.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.max_tokens && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Max Tokens</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.max_tokens.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.presence_penalty && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Presence Penalty</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.presence_penalty.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.frequency_penalty && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Frequency Penalty</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.frequency_penalty.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.seed && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Seed</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.seed.toString()} font="mono" />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}
            </>
          );
        })()}

      {/* --- MIXTURE OF N --- */}
      {variantConfig.type === "mixture_of_n" &&
        (() => {
          const config = variantConfig as MixtureOfNConfig;
          return (
            <>
              <BasicInfoItem>
                <BasicInfoItemTitle>Model (Fuser)</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <Chip label={config.fuser.model} font="mono" />
                </BasicInfoItemContent>
              </BasicInfoItem>

              {config.fuser.temperature && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Temperature</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.fuser.temperature.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.fuser.top_p && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Top P</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.fuser.top_p.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.fuser.max_tokens && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Max Tokens</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.fuser.max_tokens.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.fuser.presence_penalty && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Presence Penalty</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.fuser.presence_penalty.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.fuser.frequency_penalty && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Frequency Penalty</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.fuser.frequency_penalty.toString()} />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              {config.fuser.seed && (
                <BasicInfoItem>
                  <BasicInfoItemTitle>Seed</BasicInfoItemTitle>
                  <BasicInfoItemContent>
                    <Chip label={config.fuser.seed.toString()} font="mono" />
                  </BasicInfoItemContent>
                </BasicInfoItem>
              )}

              <BasicInfoItem>
                <BasicInfoItemTitle>Timeout</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <Chip label={`${config.timeout_s}s`} />
                </BasicInfoItemContent>
              </BasicInfoItem>

              <BasicInfoItem>
                <BasicInfoItemTitle>Candidates</BasicInfoItemTitle>
                <BasicInfoItemContent>
                  <div className="flex flex-wrap gap-1">
                    {config.candidates.map((candidate) => (
                      <Chip
                        key={candidate}
                        label={candidate}
                        link={toVariantUrl(function_name, candidate)}
                        font="mono"
                      />
                    ))}
                  </div>
                </BasicInfoItemContent>
              </BasicInfoItem>
            </>
          );
        })()}

      {/* Weight */}
      {variantConfig.weight !== null && (
        <BasicInfoItem>
          <BasicInfoItemTitle>Weight</BasicInfoItemTitle>
          <BasicInfoItemContent>
            <Chip label={variantConfig.weight.toString()} />
          </BasicInfoItemContent>
        </BasicInfoItem>
      )}
    </BasicInfoLayout>
  );
}
