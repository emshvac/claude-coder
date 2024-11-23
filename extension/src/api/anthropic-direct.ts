import { Anthropic } from "@anthropic-ai/sdk"
import { ApiHandler, withoutImageData } from "."
import { ApiHandlerOptions, KoduModelId, ModelInfo, koduModels } from "../shared/api"
import { ApiHistoryItem } from "../agent/v1"
import { WebSearchResponseDto } from "./interfaces"
import { KODU_ERROR_CODES, KoduError, koduSSEResponse } from "../shared/kodu"

interface AnthropicMessage {
    usage?: {
        input_tokens: number;
        output_tokens: number;
    };
    metadata?: {
        cached?: boolean;
    };
}

interface AnthropicDelta {
    usage?: {
        output_tokens: number;
    };
    content?: Array<{
        type: string;
        text?: string;
    }>;
}

type ToolExecutionState = {
    isExecuting: boolean;
    toolId?: string;
    response: string;
    isCompletionTool: boolean;
}

type StreamState = {
    currentText: string;
    accumulatedContent: Array<{
        type: 'text';
        text: string;
    }>;
    usage: {
        input_tokens: number;
        output_tokens: number;
        cache_creation_input_tokens: number;
        cache_read_input_tokens: number;
    };
    messageMetadata?: {
        cached?: boolean;
    };
    toolExecution: ToolExecutionState;
}

export class AnthropicDirectHandler implements ApiHandler {
    private options: ApiHandlerOptions
    private client: Anthropic
    private abortController: AbortController | null = null
    private streamState: StreamState = {
        currentText: '',
        accumulatedContent: [],
        usage: {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0
        },
        toolExecution: {
            isExecuting: false,
            response: '',
            isCompletionTool: false
        }
    }

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

    private resetStreamState() {
        this.streamState = {
            currentText: '',
            accumulatedContent: [],
            usage: {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0
            },
            toolExecution: {
                isExecuting: false,
                response: '',
                isCompletionTool: false
            }
        }
    }

    private handleToolResponse(text: string): boolean {
        // Check for tool tags
        if (text.includes('<tool') || text.includes('</tool')) {
            // Start or continue accumulating tool response
            if (!this.streamState.toolExecution.isExecuting) {
                const toolId = this.extractToolId(text)
                this.streamState.toolExecution = {
                    isExecuting: true,
                    response: text,
                    isCompletionTool: text.includes('attempt_completion'),
                    toolId
                }
            } else {
                this.streamState.toolExecution.response += text
            }

            // Check if we have a complete tool tag
            if (this.streamState.toolExecution.response.includes('</tool>')) {
                // Add complete tool response to accumulated content
                this.streamState.accumulatedContent.push({
                    type: 'text',
                    text: this.streamState.toolExecution.response
                })
                
                const wasCompletionTool = this.streamState.toolExecution.isCompletionTool
                
                // Reset tool execution state
                this.streamState.toolExecution = {
                    isExecuting: false,
                    response: '',
                    isCompletionTool: false
                }

                // Return true if this was a completion tool
                return wasCompletionTool
            }
            return false
        }

        // If we're in the middle of processing a tool
        if (this.streamState.toolExecution.isExecuting) {
            this.streamState.toolExecution.response += text
            return false
        }

        return false
    }

    private extractToolId(text: string): string | undefined {
        const match = text.match(/<([^>]+)>/)
        return match ? match[1] : undefined
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
        this.resetStreamState()

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

            let hasProcessedMessage = false

            for await (const chunk of stream) {
                hasProcessedMessage = true
                switch (chunk.type) {
                    case 'message_start':
                        const message = chunk.message as AnthropicMessage;
                        if (message?.usage) {
                            this.streamState.usage.input_tokens = message.usage.input_tokens
                            this.streamState.usage.output_tokens = message.usage.output_tokens
                            
                            if (message.metadata?.cached) {
                                this.streamState.usage.cache_read_input_tokens = this.streamState.usage.input_tokens
                            } else {
                                this.streamState.usage.cache_creation_input_tokens = this.streamState.usage.input_tokens
                            }
                        }
                        yield { code: 2, body: { text: "" } }
                        break

                    case 'content_block_start':
                        this.streamState.currentText = ''
                        yield { code: 2, body: { text: "" } }
                        break

                    case 'content_block_delta':
                        if ('text' in chunk.delta) {
                            const isCompletionTool = this.handleToolResponse(chunk.delta.text)
                            
                            if (!this.streamState.toolExecution.isExecuting) {
                                this.streamState.currentText += chunk.delta.text
                                yield { code: 2, body: { text: chunk.delta.text } }
                            }

                            if (isCompletionTool) {
                                // Yield completion and end stream
                                yield this.createCompletionResponse()
                                return
                            }
                        }
                        break

                    case 'content_block_stop':
                        if (this.streamState.currentText && !this.streamState.toolExecution.isExecuting) {
                            this.streamState.accumulatedContent.push({
                                type: 'text',
                                text: this.streamState.currentText
                            })
                            this.streamState.currentText = ''
                        }
                        break

                    case 'message_delta':
                        const delta = chunk.delta as AnthropicDelta;
                        if (delta.usage?.output_tokens) {
                            this.streamState.usage.output_tokens = delta.usage.output_tokens
                        }
                        if (delta.content && Array.isArray(delta.content)) {
                            const text = delta.content
                                .map(c => (c.type === 'text' && c.text) || '')
                                .join('')
                            if (text) {
                                const isCompletionTool = this.handleToolResponse(text)
                                if (!this.streamState.toolExecution.isExecuting) {
                                    yield { code: 2, body: { text } }
                                }
                                if (isCompletionTool) {
                                    yield this.createCompletionResponse()
                                    return
                                }
                            }
                        }
                        break

                    case 'message_stop':
                        // Handle any remaining tool response
                        if (this.streamState.toolExecution.response) {
                            this.streamState.accumulatedContent.push({
                                type: 'text',
                                text: this.streamState.toolExecution.response
                            })
                        }

                        // Handle any remaining text
                        if (this.streamState.currentText && !this.streamState.toolExecution.isExecuting) {
                            this.streamState.accumulatedContent.push({
                                type: 'text',
                                text: this.streamState.currentText
                            })
                        }

                        yield this.createCompletionResponse()
                        return
                }
            }

            // Only throw network error if we haven't processed any messages
            if (!hasProcessedMessage) {
                throw new KoduError({
                    code: KODU_ERROR_CODES.NETWORK_REFUSED_TO_CONNECT
                })
            }

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
            this.resetStreamState()
        }
    }

    private createCompletionResponse(): koduSSEResponse {
        const model = this.getModel()
        const inputCost = (model.info.inputPrice / 1_000_000) * this.streamState.usage.input_tokens
        const outputCost = (model.info.outputPrice / 1_000_000) * this.streamState.usage.output_tokens
        const totalCost = inputCost + outputCost

        return {
            code: 1,
            body: {
                anthropic: {
                    id: `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    type: 'message',
                    role: 'assistant',
                    content: this.streamState.accumulatedContent,
                    model: this.getModel().id,
                    stop_reason: 'end_turn',
                    stop_sequence: null,
                    usage: {
                        input_tokens: this.streamState.usage.input_tokens,
                        output_tokens: this.streamState.usage.output_tokens,
                        cache_creation_input_tokens: this.streamState.usage.cache_creation_input_tokens,
                        cache_read_input_tokens: this.streamState.usage.cache_read_input_tokens
                    }
                },
                internal: {
                    cost: totalCost,
                    userCredits: 0,
                    inputTokens: this.streamState.usage.input_tokens,
                    outputTokens: this.streamState.usage.output_tokens,
                    cacheCreationInputTokens: this.streamState.usage.cache_creation_input_tokens,
                    cacheReadInputTokens: this.streamState.usage.cache_read_input_tokens
                }
            }
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
