// ── Dimension schema (from GET /api/schema) ─────────────────────────────────

/** Where a param value lives in the active processor list. */
export interface DimensionParamTarget {
  /** _target_ string that identifies the processor to patch */
  processor_target: string
  /** Dot-notation path within the processor's kwargs, e.g. "memory.n" */
  path: string
}

export interface DimensionParam {
  label: string
  type: 'int' | 'float' | 'select'
  // int / float
  min?: number
  max?: number
  default: number | string
  step?: number
  // select
  options?: string[]
  /** Where to write this param value in the processor list */
  targets: DimensionParamTarget[]
}

export interface Conflict {
  /** _target_ string; triggers when that processor IS (or is NOT when negate=true) present */
  if_processor: string
  negate?:  boolean
  message:  string
  severity: 'error' | 'warning' | 'info'
}

export interface DimensionOption {
  key: string
  label: string
  description: string
  /** Processor _target_ dicts to add/remove when this option is toggled */
  processors: Record<string, unknown>[]
  params?: DimensionParam[]
  conflicts?: Conflict[]
}

export interface Dimension {
  key: string
  label: string
  description: string
  icon: string
  multi_select?: boolean
  options: DimensionOption[]
}

export interface SchemaResponse {
  dimensions: Dimension[]
}

// ── Drag-and-drop payload (serialised into dataTransfer) ─────────────────────

export interface DimensionDragPayload {
  dimensionKey:   string
  dimensionLabel: string
  processors:     Record<string, unknown>[]
}

// ── HarnessConfig (processor-list format) ────────────────────────────────────

export interface HarnessMcpConfig {
  source: 'agent_home' | 'file' | 'inline' | 'disabled'
  path?: string | null
}

export interface HarnessConfig {
  processors:      Record<string, unknown>[]
  plugins:         unknown[] | null
  mcp_config?:     HarnessMcpConfig
  spawn_subagents?: boolean
}

export interface HarnessWorkspaceConfig {
  agent_id: string
  project:  string
}

export const DEFAULT_HARNESS_CONFIG: HarnessConfig = {
  processors:      [],
  plugins:         null,
  mcp_config:      { source: 'agent_home' },
  spawn_subagents: true,
}

// ── Example harnesses (from GET /api/examples) ──────────────────────────────

export interface ExampleDescriptor {
  key:           string
  label:         string
  description:   string
  harness_config: HarnessConfig
  workspace?:    HarnessWorkspaceConfig | null
}

// ── Providers (from GET /api/providers) ─────────────────────────────────────

export interface Provider {
  id:    string
  label: string
}

// ── Vendor catalogue (from GET /api/vendors) ─────────────────────────────────

export interface VendorModelItem {
  id:    string
  label: string
}

export interface VendorInfo {
  id:               string
  label:            string
  env_key:          string
  env_key_set?:     boolean   // true when the env var is present on the server
  default_base_url: string | null
  models:           VendorModelItem[]
}

// ── Persisted model config (from GET /api/model-config) ──────────────────────

export interface ModelConfigDef {
  id:           string
  display_name: string
  vendor:       string
  model_id:     string
  api_key:      string
  base_url:     string
  extra_headers?: Record<string, string>
  capabilities: string[]
  // Optional provider-level reasoning/thinking params
  extended_thinking?: boolean
  thinking_budget_tokens?: number
  reasoning_effort?: 'low' | 'medium' | 'high'
  reasoning_summary?: boolean
}

export interface ModelConfigSlot {
  slot_name:  string
  model_ids:  string[]
  strategy:   string
}

export interface ModelConfigResponse {
  registry: ModelConfigDef[]
  slots:    ModelConfigSlot[]
}

export interface ModelConfigSaveRequest {
  registry: ModelConfigDef[]
  slots:    ModelConfigSlot[]
}

// ── Tool info (from GET /api/tools) ──────────────────────────────────────────

export interface ToolInfo {
  name:        string
  group:       'filesystem' | 'web'
  description: string
}

// ── Skill info (from GET /api/skills) ────────────────────────────────────────

export interface SkillInfo {
  name:        string
  description: string
}

// ── Custom harness (user-defined, stored in localStorage) ────────────────────

export interface CustomHarness {
  id:             string
  name:           string
  harness_config: HarnessConfig
  workspace?:     HarnessWorkspaceConfig
}

// ── Model capability tags ────────────────────────────────────────────────────
/** Functional capability of a model — can be combined freely. */
export type ModelCapability =
  | 'text'       // standard language understanding / generation
  | 'code'       // code generation & analysis
  | 'omni'       // multimodal all-in-one (text + vision + audio)
  | 'vl'         // vision-language (image input)
  | 'tts'        // text-to-speech output
  | 'asr'        // automatic speech recognition (audio input)
  | 'embedding'  // produces vector embeddings
  | 'image_gen'  // image generation output
  | 'video_gen'  // video generation output

// ── Model definition (registry entry) ───────────────────────────────────────
/** A fully-specified model the user has configured — credentials + capabilities. */
export interface ModelDef {
  id:           string              // locally unique ID (e.g. "m1a2b3")
  display_name: string              // human-readable label
  vendor:       string              // 'anthropic' | 'openai' | 'litellm' | 'gemini' | 'deepseek' | 'custom'
  model_id:     string              // provider model string, e.g. 'claude-sonnet-4-6'
  api_key:      string              // blank → use env var
  base_url:     string              // blank → vendor default
  extra_headers?: Record<string, string> // forwarded as provider request headers
  capabilities: ModelCapability[]
  // AnthropicProvider
  extended_thinking?: boolean
  thinking_budget_tokens?: number
  // LiteLLM/OpenAI-compatible reasoning models
  reasoning_effort?: 'low' | 'medium' | 'high'
  reasoning_summary?: boolean
}

// ── Model slot ────────────────────────────────────────────────────────────────
/** Named slot that references one or more registry models with a routing strategy. */
export interface ModelSlot {
  slot_name: string          // 'main' | 'compact' | 'judge' | custom
  model_ids: string[]        // ordered list of ModelDef.id; first = primary
  strategy:  'primary' | 'fallback' | 'round_robin'
}

// ── Multimodal attachments & task content ────────────────────────────────────

/** Base64-encoded image attachment — stored on user ChatMessage for rendering. */
export interface Attachment {
  type:       'image'
  media_type: string    // image/png | image/jpeg | image/gif | image/webp
  data:       string    // raw base64, NO "data:..." prefix
  name?:      string    // original filename, display only
}

/** Single content block sent to the backend (Anthropic wire format). */
export type TaskBlock =
  | { type: 'text';  text: string }
  | { type: 'image'; source: { type: 'base64'; media_type: string; data: string } }

// ── Slot configuration ────────────────────────────────────────────────────────

export interface SlotConfig {
  enabled_tools:  string[] | null   // null = all tools
  enabled_skills: string[] | null   // null = all skills
  sandbox_type:   'local' | 'remote'
  sandbox_url:    string | null
}

// ── Chat messages ─────────────────────────────────────────────────────────────

/** A single structural block within an assistant message. */
export type MessageBlock =
  | { type: 'text';        content: string }
  | { type: 'thinking';    content: string }
  | { type: 'tool_use';    id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result'; id: string; name: string; output: string; error: string | null; duration_ms: number }

/** A processor intervention captured in the step trace. */
export interface ProcessorTrigger {
  processor: string
  hook:      string
  action:    string
  detail:    Record<string, unknown>
}

/** One entry in a step's ordered execution timeline. */
export type TimelineItem =
  | { kind: 'block';     block:   MessageBlock }
  | { kind: 'processor'; trigger: ProcessorTrigger }

/** Input context for a single step — populated from BeforeModelEvent. */
export interface StepInputContext {
  tool_names:             string[]
  message_count:          number
  on_step_start_triggers: ProcessorTrigger[]
}

/** Context captured when a task starts — system prompt, tools, on_task_start triggers. */
export interface QueryContext {
  system:                  string
  tool_names:              string[]
  on_task_start_triggers:  ProcessorTrigger[]
  post_query_triggers?:    ProcessorTrigger[]
}

/** Per-step trace data attached to an assistant message after step_end. */
export interface StepTrace {
  step:          number
  model:         string
  input_tokens:  number
  output_tokens: number
  duration_ms:   number
  cost_usd:      number
  /** Events in SSE arrival order — blocks and processor triggers interleaved. */
  timeline:      TimelineItem[]
  /** Input context (tools, message count, on_step_start triggers) for this step. */
  input?:        StepInputContext
}

export interface ChatMessage {
  role:         'user' | 'assistant' | 'system'
  /** Flat text content — always present as display text. */
  content:      string
  /** Image attachments on user messages — rendered as thumbnails in the bubble. */
  attachments?: Attachment[]
  /** Structured blocks for assistant messages (thinking, text, tool calls, results). */
  blocks?:      MessageBlock[]
  streaming?:   boolean   // true while this assistant message is actively receiving tokens
  /** Per-step trace data (model, tokens, processor timeline). Populated after step_end. */
  stepTraces?:  StepTrace[]
  /** Task-start context (system prompt, tools, on_task_start triggers). On user messages. */
  query_context?: QueryContext
}

// ── Run request / response ───────────────────────────────────────────────────

export interface RunRequest {
  harness_config:   HarnessConfig
  task:             string | TaskBlock[]
  provider_config:  Record<string, unknown>  // full ModelConfig dict (_target_ format)
  success_criteria?: string
  token_budget?:     number
  session_id?:       string
  slot_config?:      SlotConfig
  agent_id?:         string   // AGENT_HOME routing (default: "default")
  project?:          string   // AGENT_HOME routing (default: "default")
}

// ── AGENT_HOME ────────────────────────────────────────────────────────────────

export interface AgentEntry {
  id:             string
  projects:       string[]
  workspace_path: string               // AGENT_HOME/workspaces/{id}
  memory_path:    string               // AGENT_HOME/workspaces/{id}/memory
  project_paths:  Record<string, string> // project → absolute path
}

export interface HomeInfo {
  home:             string
  default_agent_id: string
  default_project:  string
  agents_tree:      AgentEntry[]
  plugins_path:     string
  skills_path:      string
  configs_path:     string
}

export interface AgentHarnessConfigResponse {
  agent_id:       string
  project:        string
  path:           string
  exists:         boolean
  harness_config: HarnessConfig
  used_default:   boolean
  persisted_default: boolean
}

export interface RunResponse {
  run_id:     string
  session_id: string
}

// ── Filesystem (from GET /api/fs) ────────────────────────────────────────────

export interface FsEntry {
  name:  string
  type:  'file' | 'dir'
  size:  number
  mtime: string
}

export interface FsListResponse {
  path:    string
  entries: FsEntry[]
}

export interface FsFileResponse {
  path:    string
  content: string
}

// ── MCP servers ───────────────────────────────────────────────────────────────

export interface McpServerConfig {
  id:        string
  name:      string
  transport: 'stdio' | 'http'
  command:   string
  url:       string
  env:       Record<string, string>
  enabled:   boolean
}

export interface McpToolInfo {
  name:         string
  description:  string
  input_schema: Record<string, unknown>
}

// ── Plugins ───────────────────────────────────────────────────────────────────

export interface PluginInfo {
  id:          string
  name:        string
  description: string
  version:     string
  path:        string
  enabled:     boolean
  tool_count:  number
  skill_count: number
  mcp_count:   number
}

// ── Custom processors ────────────────────────────────────────────────────────

export interface CustomProcessorCandidate {
  class_name: string
  label: string
  file_path: string
  doc: string
}

export interface CustomProcessorScanResponse {
  candidates: CustomProcessorCandidate[]
}

export interface CustomProcessorInfo {
  id: string
  label: string
  class_name: string
  target: string
  source_path: string
  installed_path: string
}

export interface CustomProcessorTestResponse {
  ok: boolean
  instantiable: boolean
  required_args: string[]
  message: string
}

// ── Help / Docs ───────────────────────────────────────────────────────────────

export interface DocEntry {
  path:  string   // e.g. "feats/plugins"
  title: string   // extracted H1 or humanised filename
}

export interface DocSection {
  name:  string
  items: DocEntry[]
}

export interface DocTree {
  sections: DocSection[]
}

export interface DocContent {
  path:    string
  title:   string
  content: string  // raw markdown
}

// ── Validate ─────────────────────────────────────────────────────────────────

export interface ValidateResponse {
  ok:      boolean
  error:   string | null
  hint:    string | null
  details: string | null
}

// ── Session history ───────────────────────────────────────────────────────────

export interface SessionMeta {
  session_id:    string
  agent_id:      string
  project:       string
  first_query:   string
  created_at:    string   // ISO timestamp
  updated_at:    string
  run_count:     number
  match_snippet?: string | null
}

export interface SessionListResponse {
  sessions:  SessionMeta[]
  total:     number
  page:      number
  page_size: number
}

export interface SessionDisplayMessage {
  role:          'user' | 'assistant' | 'system'
  content:       string
  tool_calls:    Array<{ name: string; id: string; output?: string }>
  /** Per-step trace data populated from the companion trace JSONL (assistant messages). */
  step_traces?:  StepTrace[]
  /** Task-start context: system prompt, tools, processor triggers (user messages). */
  query_context?: QueryContext
}

export interface SessionMessagesResponse {
  session_id: string
  messages:   SessionDisplayMessage[]
}

// ── SSE event shapes ─────────────────────────────────────────────────────────

export type RunEvent =
  | { type: 'thinking';           run_id?: string; content: string }
  | { type: 'token';              run_id?: string; content: string }
  | { type: 'tool_use';           run_id?: string; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result';        run_id?: string; id: string; name: string; output: string; error: string | null; duration_ms: number }
  | { type: 'step_start';         run_id: string; step: number; ts_ms: number }
  | { type: 'step_end';           run_id?: string; step: number; cost_usd: number; duration_ms: number; input_tokens: number; output_tokens: number; model?: string }
  | { type: 'processor_trigger';  run_id?: string; step: number; processor: string; hook: string; action: string; detail: Record<string, unknown> }
  | { type: 'task_context';       run_id?: string; system: string; tool_names: string[] }
  | { type: 'step_context';       run_id?: string; step: number; tool_names: string[]; message_count: number }
  | { type: 'compact';            run_id?: string; before_msgs: number; after_msgs: number; before_tokens: number; after_tokens: number }
  | { type: 'child_start';        parent_run_id: string; child_run_id: string; task: string }
  | { type: 'done';               exit_reason: string; steps: number; total_cost: number; total_input_tokens: number; total_output_tokens: number; passed: boolean | null; error?: string }
  | { type: 'error';              message: string }
