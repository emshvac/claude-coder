import { Anthropic } from "@anthropic-ai/sdk"
import { ApiHandler, withoutImageData } from "."
import { ApiHandlerOptions, KoduModelId, ModelInfo, koduModels } from "../shared/api"
import { ApiHistoryItem } from "../agent/v1"
import { WebSearchResponseDto } from "./interfaces"
import { KODU_ERROR_CODES, KoduError, koduSSEResponse } from "../shared/kodu"

export class AnthropicDirectHandler implements ApiHandler {
    private options: ApiHandlerOptions
    private client: Anthropic
    private abortController: AbortController | null = null

    constructor(options: ApiHandlerOptions) {
        this.options = options
        if (!options.apiKey) {
            throw new Error("Anthropic API key is required")
        }
        this.client = new Anthropic({
            apiKey: options.apiKey,
            defaultHeaders: {
                'anthropic-beta': 'prompt-caching-2024-07-31'
            }
        })
    }

    async abortRequest(): Promise<void> {
        if (this.abortController) {
            this.abortController.abort()
            this.abortController = null
        }
    }

    createUserReadableRequest(
        userContent: Array<
            | Anthropic.TextBlockParam
            | Anthropic.ImageBlockParam
            | Anthropic.ToolUseBlockParam
            | Anthropic.ToolResultBlockParam
        >
    ): any {
        return {
            model: this.getModel().id,
            max_tokens: this.getModel().info.maxTokens,
            system: "(Direct Anthropic API)",
            messages: [{ role: "user", content: withoutImageData(userContent) }]
        }
    }

    async *createMessageStream(
        systemPrompt: string,
        messages: ApiHistoryItem[],
        creativeMode?: "normal" | "creative" | "deterministic",
        abortSignal?: AbortSignal | null,
        customInstructions?: string,
        userMemory?: string,
        environmentDetails?: string
    ): AsyncIterableIterator<koduSSEResponse> {
        this.abortController = new AbortController()

        try {
            // Build system content blocks with cache control
            const systemBlocks: Anthropic.Beta.PromptCaching.Messages.PromptCachingBetaTextBlockParam[] = []

            // Add system prompt
            systemBlocks.push({
                type: "text",
                text: systemPrompt.trim()
            })

            // Add custom instructions
            if (customInstructions?.trim()) {
                systemBlocks.push({
                    type: "text",
                    text: customInstructions.trim()
                })
            }

            // Mark the last system block with cache_control
            if (systemBlocks.length > 0) {
                systemBlocks[systemBlocks.length - 1].cache_control = { type: "ephemeral" }
            }

            // Add environment details with ephemeral cache control
            if (environmentDetails?.trim()) {
                systemBlocks.push({
                    type: "text",
                    text: environmentDetails.trim(),
                    cache_control: { type: "ephemeral" }
                })
            }

            // Convert messages to Anthropic format with cache control
            const userMsgIndices = messages.reduce(
                (acc, msg, index) => (msg.role === "user" ? [...acc, index] : acc),
                [] as number[]
            )
            const lastUserMsgIndex = userMsgIndices[userMsgIndices.length - 1] ?? -1
            const secondLastMsgUserIndex = userMsgIndices[userMsgIndices.length - 2] ?? -1

            const anthropicMessages = messages.map((msg, index) => {
                const { ts, ...message } = msg
                const isLastOrSecondLastUser = index === lastUserMsgIndex || index === secondLastMsgUserIndex

                return {
                    ...message,
                    content: typeof message.content === 'string'
                        ? [{
                            type: 'text' as const,
                            text: message.content,
                            ...(isLastOrSecondLastUser && { cache_control: { type: "ephemeral" } })
                        }]
                        : message.content.map((block, blockIndex) => {
                            if (typeof block === 'string') {
                                return {
                                    type: 'text' as const,
                                    text: block,
                                    ...(isLastOrSecondLastUser && blockIndex === message.content.length - 1 && { cache_control: { type: "ephemeral" } })
                                }
                            }
                            if ('type' in block && block.type === 'text') {
                                return {
                                    ...block,
                                    ...(isLastOrSecondLastUser && blockIndex === message.content.length - 1 && { cache_control: { type: "ephemeral" } })
                                } as Anthropic.TextBlockParam
                            }
                            return {
                                type: 'text' as const,
                                text: JSON.stringify(block),
                                ...(isLastOrSecondLastUser && blockIndex === message.content.length - 1 && { cache_control: { type: "ephemeral" } })
                            }
                        })
                }
            })

            // Get temperature settings
            const temperatures = {
                creative: { temperature: 0.3, top_p: 0.9 },
                normal: { temperature: 0.2, top_p: 0.8 },
                deterministic: { temperature: 0.1, top_p: 0.9 }
            }
            const { temperature, top_p } = temperatures[creativeMode || "normal"]

            // Start stream
            yield { code: 0, body: undefined }

            // Create stream with prompt caching enabled
            const stream = await this.client.messages.create(
                {
                    model: this.getModel().id,
                    max_tokens: this.getModel().info.maxTokens,
                    system: systemBlocks,
                    messages: anthropicMessages,
                    temperature,
                    top_p,
                    stream: true
                },
                {
                    signal: this.abortController.signal,
                    headers: {
                        'anthropic-beta': 'prompt-caching-2024-07-31'
                    }
                }
            )

            let content: Anthropic.ContentBlock[] = []
            let usage = {
                input_tokens: 0,
                output_tokens: 0
            }

            for await (const chunk of stream) {
                if (chunk.type === 'message_start') {
                    if (chunk.message?.usage) {
                        usage.input_tokens = chunk.message.usage.input_tokens
                        usage.output_tokens = chunk.message.usage.output_tokens
                    }
                    yield { code: 2, body: { text: "" } }
                } else if (chunk.type === 'content_block_start') {
                    yield { code: 2, body: { text: "" } }
                } else if (chunk.type === 'content_block_delta') {
                    if ('text' in chunk.delta) {
                        yield { code: 2, body: { text: chunk.delta.text } }
                        content.push({ type: 'text', text: chunk.delta.text })
                    }
                } else if (chunk.type === 'message_delta') {
                    if ('content' in chunk.delta && Array.isArray(chunk.delta.content)) {
                        const text = chunk.delta.content
                            .map((c: { type: string; text?: string }) => (c.type === 'text' && c.text) || '')
                            .join('')
                        yield { code: 2, body: { text } }
                    }
                    if ('usage' in chunk && chunk.usage?.output_tokens) {
                        usage.output_tokens = chunk.usage.output_tokens
                    }
                } else if (chunk.type === 'message_stop') {
                    if ('content' in chunk && Array.isArray(chunk.content)) {
                        const model = this.getModel()
                        const inputCost = (model.info.inputPrice / 1_000_000) * usage.input_tokens
                        const outputCost = (model.info.outputPrice / 1_000_000) * usage.output_tokens
                        const totalCost = inputCost + outputCost

                        const metadata = (chunk as any).metadata
                        const isCached = metadata?.cached === true

                        const cacheCreationInputTokens = isCached ? 0 : usage.input_tokens
                        const cacheReadInputTokens = isCached ? usage.input_tokens : 0

                        yield {
                            code: 1,
                            body: {
                                anthropic: {
                                    id: `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                                    type: 'message',
                                    role: 'assistant',
                                    content,
                                    model: this.getModel().id,
                                    stop_reason: 'end_turn',
                                    stop_sequence: null,
                                    usage: {
                                        input_tokens: usage.input_tokens,
                                        output_tokens: usage.output_tokens,
                                        cache_creation_input_tokens: cacheCreationInputTokens,
                                        cache_read_input_tokens: cacheReadInputTokens
                                    }
                                },
                                internal: {
                                    cost: totalCost,
                                    userCredits: 0,
                                    inputTokens: usage.input_tokens,
                                    outputTokens: usage.output_tokens,
                                    cacheCreationInputTokens,
                                    cacheReadInputTokens
                                }
                            }
                        }
                        return
                    }
                }
            }

            throw new KoduError({
                code: KODU_ERROR_CODES.NETWORK_REFUSED_TO_CONNECT
            })

        } catch (error) {
            if (error instanceof Error && error.message === "aborted") {
                return
            }

            if (error instanceof Error) {
                if (error.message.includes("prompt is too long")) {
                    yield {
                        code: -1,
                        body: {
                            msg: "prompt is too long",
                            status: 413
                        }
                    }
                } else {
                    yield {
                        code: -1,
                        body: {
                            msg: error.message,
                            status: KODU_ERROR_CODES.NETWORK_REFUSED_TO_CONNECT
                        }
                    }
                }
            }
            return
        } finally {
            this.abortController = null
        }
    }

    getModel(): { id: KoduModelId; info: ModelInfo } {
        const modelId = this.options.apiModelId
        if (modelId && modelId in koduModels) {
            const id = modelId as KoduModelId
            return { id, info: koduModels[id] }
        }
        return { id: "claude-3-5-sonnet-20240620", info: koduModels["claude-3-5-sonnet-20240620"] }
    }

    // These methods are not supported in direct Anthropic integration
    async *sendWebSearchRequest(): AsyncIterableIterator<WebSearchResponseDto> {
        throw new Error("Web search is not supported with direct Anthropic API integration")
    }

    async sendUrlScreenshotRequest(): Promise<Blob> {
        throw new Error("URL screenshots are not supported with direct Anthropic API integration")
    }

    async sendAskConsultantRequest(): Promise<any> {
        throw new Error("Ask consultant is not supported with direct Anthropic API integration")
    }

    async sendSummarizeRequest(): Promise<any> {
        throw new Error("Summarize is not supported with direct Anthropic API integration")
    }
}
