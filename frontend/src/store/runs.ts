import { create } from 'zustand'
import type {
  HarnessConfig,
  HarnessWorkspaceConfig,
  ChatMessage,
  MessageBlock,
  RunEvent,
  SlotConfig,
  TaskBlock,
  Attachment,
  SessionDisplayMessage,
  StepTrace,
  TimelineItem,
  StepInputContext,
  QueryContext,
  ProcessorTrigger,
} from '../api/types'
import { api } from '../api/client'
import { workspaceFromHarnessName } from '../lib/labWorkspace'

export type RunStatus = 'idle' | 'running' | 'done' | 'error'
const USER_INTERRUPTED_MESSAGE = 'user actively interrupted execution'

export interface RunStep {
  step:          number
  run_id:        string
  cost_usd:      number
  duration_ms:   number
  input_tokens:  number
  output_tokens: number
  ts_ms:         number   // ms since run start (from step_start event)
  tool_ms:       number   // total tool execution time this step
}

export interface RunResult {
  exit_reason:         string
  steps:               number
  total_cost:          number
  total_input_tokens:  number
  total_output_tokens: number
  passed:              boolean | null
}

export interface ChildRun {
  run_id:        string
  parent_run_id: string
  task:          string
  status:        'running' | 'done'
  messages:      ChatMessage[]
  steps:         RunStep[]
}

export interface RunInstance {
  id:              string               // client-side column id (not run_id)
  run_id:          string | null
  session_id:      string | null        // multi-turn session continuity
  harnessConfig:   HarnessConfig
  harnessName:     string
  workspaceAgentId: string
  workspaceProject: string
  status:          RunStatus
  messages:        ChatMessage[]        // root agent chat history
  steps:           RunStep[]
  result:          RunResult | null
  error:           string | null
  stopStream:      (() => void) | null
  children:        Record<string, ChildRun>  // child_run_id → ChildRun
  isReadOnly:      boolean             // true when showing historical session
  historySessionId: string | null      // the session_id being previewed
}

function cloneHarnessConfig(cfg: HarnessConfig): HarnessConfig {
  if (typeof structuredClone === 'function') return structuredClone(cfg)
  return JSON.parse(JSON.stringify(cfg)) as HarnessConfig
}

function makeRun(
  id: string,
  harnessConfig: HarnessConfig,
  harnessName = 'harness',
  workspace?: HarnessWorkspaceConfig,
): RunInstance {
  const ws = workspaceFromHarnessName(harnessName)
  const workspaceAgentId = (workspace?.agent_id || '').trim() || ws.agentId
  const workspaceProject = (workspace?.project || '').trim() || ws.project
  return {
    id, run_id: null, session_id: null, harnessConfig: cloneHarnessConfig(harnessConfig),
    harnessName,
    workspaceAgentId,
    workspaceProject,
    status: 'idle', messages: [], steps: [],
    result: null, error: null, stopStream: null,
    children: {},
    isReadOnly: false, historySessionId: null,
  }
}

interface RunsState {
  // Compare mode: list of column instances (1–4)
  columns:      RunInstance[]
  addColumn:    (harnessConfig: HarnessConfig, harnessName?: string, workspace?: HarnessWorkspaceConfig) => void
  removeColumn: (id: string) => void
  updateColumn: (id: string, patch: Partial<RunInstance>) => void
  resetColumn:  (id: string) => void
  resetSession: (id: string) => void  // clears messages + session_id ("New chat")
  clearAll:     () => void

  startRun: (
    columnId: string,
    task: string | TaskBlock[],
    successCriteria: string,
    providerConfig: Record<string, unknown>,
    slotConfig?: SlotConfig,
    options?: {
      harnessConfig?: HarnessConfig
      harnessName?: string
      agentId?: string
      project?: string
    },
  ) => Promise<void>
  stopRun: (columnId: string) => void

  /** Load historical messages into the column for read-only display. */
  loadHistorySession: (
    columnId:  string,
    sessionId: string,
    messages:  SessionDisplayMessage[],
  ) => void

  /** Switch the column from read-only history view back to active input. */
  resumeHistorySession: (columnId: string) => void
}

let _nextId = 1

export const useRunsStore = create<RunsState>((set, get) => ({
  columns: [],

  addColumn: (harnessConfig, harnessName, workspace) => {
    const id = String(_nextId++)
    set((s) => ({ columns: [...s.columns, makeRun(id, harnessConfig, harnessName ?? 'harness', workspace)] }))
  },

  removeColumn: (id) =>
    set((s) => ({ columns: s.columns.filter((c) => c.id !== id) })),

  updateColumn: (id, patch) =>
    set((s) => ({
      columns: s.columns.map((c) => c.id === id ? { ...c, ...patch } : c),
    })),

  resetColumn: (id) =>
    set((s) => ({
      columns: s.columns.map((c) =>
        c.id === id
          ? {
              ...makeRun(id, c.harnessConfig, c.harnessName),
              harnessConfig: cloneHarnessConfig(c.harnessConfig),
              harnessName: c.harnessName,
              workspaceAgentId: c.workspaceAgentId,
              workspaceProject: c.workspaceProject,
            }
          : c
      ),
    })),

  resetSession: (id) =>
    set((s) => ({
      columns: s.columns.map((c) =>
        c.id === id
          ? { ...c, messages: [], session_id: null, status: 'idle', result: null, error: null,
              isReadOnly: false, historySessionId: null }
          : c
      ),
    })),

  clearAll: () => {
    get().columns.forEach((c) => c.stopStream?.())
    set({ columns: [] })
  },

  startRun: async (columnId, task, successCriteria, providerConfig, slotConfig, options) => {
    const { updateColumn } = get()
    const col = get().columns.find((c) => c.id === columnId)
    if (!col) return
    const effectiveHarnessConfig = cloneHarnessConfig(options?.harnessConfig ?? col.harnessConfig)
    const effectiveHarnessName = (options?.harnessName ?? col.harnessName ?? 'harness').trim() || 'harness'
    const ws = workspaceFromHarnessName(effectiveHarnessName)
    const effectiveAgentId = options?.agentId ?? col.workspaceAgentId ?? ws.agentId
    const effectiveProject = options?.project ?? col.workspaceProject ?? ws.project

    // Derive display text and attachments from task (string or TaskBlock[])
    const userContent = typeof task === 'string'
      ? task
      : task.filter((b): b is { type: 'text'; text: string } => b.type === 'text').map((b) => b.text).join('\n')
    type ImageBlock = { type: 'image'; source: { type: 'base64'; media_type: string; data: string } }
    const userAttachments: Attachment[] | undefined = typeof task === 'string'
      ? undefined
      : (task.filter((b): b is ImageBlock => b.type === 'image')
           .map((b) => ({ type: 'image' as const, media_type: b.source.media_type, data: b.source.data })))

    // Append user message + streaming assistant placeholder
    const userMsg:  ChatMessage = {
      role:        'user',
      content:     userContent,
      attachments: userAttachments?.length ? userAttachments : undefined,
    }
    const assistMsg: ChatMessage = { role: 'assistant', content: '', blocks: [], streaming: true }
    updateColumn(columnId, {
      status:   'running',
      harnessConfig: effectiveHarnessConfig,
      harnessName: effectiveHarnessName,
      workspaceAgentId: effectiveAgentId,
      workspaceProject: effectiveProject,
      messages: [...col.messages, userMsg, assistMsg],
      steps:    [],
      result:   null,
      error:    null,
      children: {},
    })

    let runId: string
    let newSessionId: string
    try {
      const res = await api.startRun({
        harness_config:   effectiveHarnessConfig,
        task,
        provider_config:  providerConfig,
        success_criteria: successCriteria,
        session_id:       col.session_id ?? undefined,
        slot_config:      slotConfig,
        agent_id:         effectiveAgentId,
        project:          effectiveProject,
      })
      runId = res.run_id
      newSessionId = res.session_id
    } catch (err) {
      // Remove streaming placeholder on request failure
      set((s) => ({
        columns: s.columns.map((c) =>
          c.id === columnId
            ? { ...c, status: 'error', error: String(err),
                messages: c.messages.filter((m) => !m.streaming) }
            : c
        ),
      }))
      return
    }

    updateColumn(columnId, { run_id: runId, session_id: newSessionId })

    // ── Per-step accumulator for tool durations and ts_ms ──────────────────
    // Keyed by `${run_id}:${step_id}` so root and child steps don't collide.
    const pendingStepTs:   Record<string, number> = {}   // step key → ts_ms
    const pendingToolMs:   Record<string, number> = {}   // step key → accumulated tool ms

    // ── Per-step trace timeline ────────────────────────────────────────────
    const pendingTimeline:    Record<string, TimelineItem[]>    = {}  // step key → ordered items
    const pendingCurrentStep: Record<string, number>            = {}  // rid → latest step number
    const pendingTextBuf:     Record<string, string>            = {}  // rid → accumulated text
    const pendingThinkBuf:    Record<string, string>            = {}  // rid → accumulated thinking
    const pendingStepContext: Record<string, StepInputContext>  = {}  // step key → input context

    /** Flush text/thinking buffers as timeline blocks for the given run_id. */
    function flushTextBuffers(rid: string) {
      const step = pendingCurrentStep[rid]
      if (step === undefined) return
      const key = `${rid}:${step}`
      if (pendingThinkBuf[rid]) {
        pendingTimeline[key] = [...(pendingTimeline[key] ?? []),
          { kind: 'block', block: { type: 'thinking', content: pendingThinkBuf[rid] } }]
        delete pendingThinkBuf[rid]
      }
      if (pendingTextBuf[rid]) {
        pendingTimeline[key] = [...(pendingTimeline[key] ?? []),
          { kind: 'block', block: { type: 'text', content: pendingTextBuf[rid] } }]
        delete pendingTextBuf[rid]
      }
    }

    /** Append a StepTrace to the last assistant message in the root column. */
    function appendStepTrace(trace: StepTrace) {
      set((s) => ({
        columns: s.columns.map((c) => {
          if (c.id !== columnId) return c
          const msgs = [...c.messages]
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'assistant') {
              msgs[i] = { ...msgs[i], stepTraces: [...(msgs[i].stepTraces ?? []), trace] }
              return { ...c, messages: msgs }
            }
          }
          return c
        }),
      }))
    }

    /** Append a StepTrace to the last assistant message in a child run. */
    function appendChildStepTrace(childRunId: string, trace: StepTrace) {
      set((s) => ({
        columns: s.columns.map((c) => {
          if (c.id !== columnId) return c
          const child = c.children[childRunId]
          if (!child) return c
          const msgs = [...child.messages]
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'assistant') {
              msgs[i] = { ...msgs[i], stepTraces: [...(msgs[i].stepTraces ?? []), trace] }
              return {
                ...c,
                children: { ...c.children, [childRunId]: { ...child, messages: msgs } },
              }
            }
          }
          return c
        }),
      }))
    }

    /** Set or update query_context on the last user message in the root column. */
    function setQueryContext(patch: Partial<QueryContext>) {
      set((s) => ({
        columns: s.columns.map((c) => {
          if (c.id !== columnId) return c
          const msgs = [...c.messages]
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'user') {
              const existing = msgs[i].query_context ?? { system: '', tool_names: [], on_task_start_triggers: [], post_query_triggers: [] }
              msgs[i] = { ...msgs[i], query_context: { ...existing, ...patch } }
              return { ...c, messages: msgs }
            }
          }
          return c
        }),
      }))
    }

    /** Append an on_task_start trigger to the last user message's query_context. */
    function appendTaskStartTrigger(trigger: QueryContext['on_task_start_triggers'][number]) {
      set((s) => ({
        columns: s.columns.map((c) => {
          if (c.id !== columnId) return c
          const msgs = [...c.messages]
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'user') {
              const existing = msgs[i].query_context ?? { system: '', tool_names: [], on_task_start_triggers: [], post_query_triggers: [] }
              msgs[i] = { ...msgs[i], query_context: { ...existing, on_task_start_triggers: [...existing.on_task_start_triggers, trigger] } }
              return { ...c, messages: msgs }
            }
          }
          return c
        }),
      }))
    }

    /** Append query post-processing trigger (e.g. UserWrapper on step 0). */
    function appendQueryPostTrigger(trigger: ProcessorTrigger) {
      set((s) => ({
        columns: s.columns.map((c) => {
          if (c.id !== columnId) return c
          const msgs = [...c.messages]
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'user') {
              const existing = msgs[i].query_context ?? { system: '', tool_names: [], on_task_start_triggers: [], post_query_triggers: [] }
              const exists = (existing.post_query_triggers ?? []).some((t) =>
                t.processor === trigger.processor
                && t.hook === trigger.hook
                && t.action === trigger.action
                && JSON.stringify(t.detail ?? {}) === JSON.stringify(trigger.detail ?? {})
              )
              if (exists) return { ...c, messages: msgs }
              msgs[i] = {
                ...msgs[i],
                query_context: {
                  ...existing,
                  post_query_triggers: [...(existing.post_query_triggers ?? []), trigger],
                },
              }
              return { ...c, messages: msgs }
            }
          }
          return c
        }),
      }))
    }

    /** Ensure per-step input context exists before writing trigger details. */
    function ensureStepContext(stepKey: string) {
      if (!pendingStepContext[stepKey]) {
        pendingStepContext[stepKey] = {
          tool_names: [],
          message_count: 0,
          on_step_start_triggers: [],
        }
      }
    }

    // ── Root-agent message helpers ─────────────────────────────────────────

    function updateStreamingMsg(transform: (msg: ChatMessage) => ChatMessage) {
      set((s) => ({
        columns: s.columns.map((c) => {
          if (c.id !== columnId) return c
          const last = c.messages[c.messages.length - 1]
          if (!last?.streaming) return c
          const msgs = [...c.messages]
          msgs[msgs.length - 1] = transform(last)
          return { ...c, messages: msgs }
        }),
      }))
    }

    function pushBlock(block: MessageBlock) {
      updateStreamingMsg((msg) => ({ ...msg, blocks: [...(msg.blocks ?? []), block] }))
    }

    function appendThinking(msg: ChatMessage, delta: string): ChatMessage {
      const blocks = [...(msg.blocks ?? [])]
      const last = blocks[blocks.length - 1]
      if (last?.type === 'thinking') {
        blocks[blocks.length - 1] = { type: 'thinking', content: last.content + delta }
      } else {
        blocks.push({ type: 'thinking', content: delta })
      }
      return { ...msg, blocks }
    }

    // ── Child-agent message helpers ────────────────────────────────────────

    function updateChildStreamingMsg(childRunId: string, transform: (msg: ChatMessage) => ChatMessage) {
      set((s) => ({
        columns: s.columns.map((c) => {
          if (c.id !== columnId) return c
          const child = c.children[childRunId]
          if (!child) return c
          const last = child.messages[child.messages.length - 1]
          if (!last?.streaming) return c
          const msgs = [...child.messages]
          msgs[msgs.length - 1] = transform(last)
          return {
            ...c,
            children: { ...c.children, [childRunId]: { ...child, messages: msgs } },
          }
        }),
      }))
    }

    function pushChildBlock(childRunId: string, block: MessageBlock) {
      updateChildStreamingMsg(childRunId, (msg) => ({ ...msg, blocks: [...(msg.blocks ?? []), block] }))
    }

    // ── SSE event dispatch ─────────────────────────────────────────────────

    const stop = api.streamRun(runId, (event: RunEvent) => {
      // Determine whether this event belongs to root or a child.
      const evRunId = (event as { run_id?: string }).run_id
      const isChild = evRunId !== undefined && evRunId !== runId

      if (event.type === 'child_start') {
        // Create a new ChildRun entry and add a streaming placeholder.
        const placeholder: ChatMessage = { role: 'assistant', content: '', blocks: [], streaming: true }
        set((s) => ({
          columns: s.columns.map((c) => {
            if (c.id !== columnId) return c
            return {
              ...c,
              children: {
                ...c.children,
                [event.child_run_id]: {
                  run_id:        event.child_run_id,
                  parent_run_id: event.parent_run_id,
                  task:          event.task,
                  status:        'running',
                  messages:      [placeholder],
                  steps:         [],
                },
              },
            }
          }),
        }))
        return
      }

      if (event.type === 'step_start') {
        const key = `${event.run_id}:${event.step}`
        pendingStepTs[key]  = event.ts_ms
        pendingToolMs[key]  = 0
        pendingTimeline[key] = []
        pendingStepContext[key] = {
          tool_names: [],
          message_count: 0,
          // processor_trigger events (step_start hook) arrive before step_start in the
          // SSE stream (emitted inside pipe_all before tracer.on_event(StepStartEvent)).
          // Preserve any triggers already buffered so they are not overwritten.
          on_step_start_triggers: pendingStepContext[key]?.on_step_start_triggers ?? [],
        }
        pendingCurrentStep[event.run_id] = event.step
        return
      }

      if (event.type === 'thinking') {
        if (isChild) updateChildStreamingMsg(evRunId!, (msg) => appendThinking(msg, event.content))
        else         updateStreamingMsg((msg) => appendThinking(msg, event.content))
        // Accumulate for timeline
        const thRid = evRunId ?? runId
        pendingThinkBuf[thRid] = (pendingThinkBuf[thRid] ?? '') + event.content

      } else if (event.type === 'token') {
        const appendToken = (msg: ChatMessage): ChatMessage => {
          const blocks = [...(msg.blocks ?? [])]
          const last = blocks[blocks.length - 1]
          if (last?.type === 'text') {
            blocks[blocks.length - 1] = { type: 'text', content: last.content + event.content }
          } else {
            blocks.push({ type: 'text', content: event.content })
          }
          return { ...msg, content: msg.content + event.content, blocks }
        }
        if (isChild) updateChildStreamingMsg(evRunId!, appendToken)
        else         updateStreamingMsg(appendToken)
        // Accumulate for timeline
        const tokRid = evRunId ?? runId
        pendingTextBuf[tokRid] = (pendingTextBuf[tokRid] ?? '') + event.content

      } else if (event.type === 'tool_use') {
        const block: MessageBlock = { type: 'tool_use', id: event.id, name: event.name, input: event.input }
        if (isChild) pushChildBlock(evRunId!, block)
        else         pushBlock(block)
        // Flush text/thinking then push tool_use to timeline
        const tuRid = evRunId ?? runId
        flushTextBuffers(tuRid)
        const tuKey = `${tuRid}:${pendingCurrentStep[tuRid] ?? 0}`
        pendingTimeline[tuKey] = [...(pendingTimeline[tuKey] ?? []),
          { kind: 'block', block: { type: 'tool_use', id: event.id, name: event.name, input: event.input } }]

      } else if (event.type === 'tool_result') {
        const block: MessageBlock = { type: 'tool_result', id: event.id, name: event.name, output: event.output, error: event.error, duration_ms: event.duration_ms }
        if (isChild) pushChildBlock(evRunId!, block)
        else         pushBlock(block)
        // Accumulate tool_ms into the most recent open step for this run.
        const stepPrefix = `${evRunId ?? runId}:`
        const openKeys = Object.keys(pendingStepTs).filter(k => k.startsWith(stepPrefix))
        const openKey  = openKeys[openKeys.length - 1]
        if (openKey) pendingToolMs[openKey] = (pendingToolMs[openKey] ?? 0) + event.duration_ms
        // Push tool_result to timeline
        const trRid = evRunId ?? runId
        const trKey = `${trRid}:${pendingCurrentStep[trRid] ?? 0}`
        pendingTimeline[trKey] = [...(pendingTimeline[trKey] ?? []),
          { kind: 'block', block: { type: 'tool_result', id: event.id, name: event.name, output: event.output, error: event.error, duration_ms: event.duration_ms } }]

      } else if (event.type === 'task_context') {
        setQueryContext({ system: event.system, tool_names: event.tool_names })

      } else if (event.type === 'step_context') {
        const scRid = evRunId ?? runId
        const scKey = `${scRid}:${event.step}`
        if (pendingStepContext[scKey]) {
          pendingStepContext[scKey] = {
            ...pendingStepContext[scKey],
            tool_names:    event.tool_names,
            message_count: event.message_count,
          }
        }

      } else if (event.type === 'processor_trigger') {
        const ptRid = evRunId ?? runId
        const hook = event.hook
        const isTaskStart = hook === 'task_start' || hook === 'on_task_start'
        const isStepStart = hook === 'step_start' || hook === 'on_step_start'
        const isBeforeModel = hook === 'before_model' || hook === 'on_before_model' || hook === 'on_model_start'

        if (isTaskStart) {
          appendTaskStartTrigger({ processor: event.processor, hook: event.hook, action: event.action, detail: event.detail })
        } else if (isStepStart || isBeforeModel) {
          const ptKey = `${ptRid}:${event.step}`
          ensureStepContext(ptKey)
          pendingStepContext[ptKey] = {
            ...pendingStepContext[ptKey],
            on_step_start_triggers: [...pendingStepContext[ptKey].on_step_start_triggers,
              { processor: event.processor, hook: event.hook, action: event.action, detail: event.detail }],
          }
          // Query post-processing: only step 0 UserWrapper-like triggers
          if (
            event.step === 0
            && isStepStart
            && /userwrapper/i.test(event.processor)
          ) {
            appendQueryPostTrigger({ processor: event.processor, hook: event.hook, action: event.action, detail: event.detail })
          }
        } else {
          const ptKey = `${ptRid}:${event.step}`
          pendingTimeline[ptKey] = [...(pendingTimeline[ptKey] ?? []),
            { kind: 'processor', trigger: { processor: event.processor, hook: event.hook, action: event.action, detail: event.detail } }]
        }

      } else if (event.type === 'step_end') {
        const rid2 = evRunId ?? runId
        // step_start sends step_id (0-based); step_end sends step_id+1 (1-based).
        // Use step-1 to look up the pending accumulators created at step_start.
        const stepKey = `${rid2}:${event.step - 1}`
        const ts_ms   = pendingStepTs[stepKey]  ?? 0
        const tool_ms = pendingToolMs[stepKey]  ?? 0
        delete pendingStepTs[stepKey]
        delete pendingToolMs[stepKey]

        // Flush remaining text/thinking into the timeline before sealing.
        flushTextBuffers(rid2)

        const trace: StepTrace = {
          step:          event.step,
          model:         event.model ?? '',
          input_tokens:  event.input_tokens,
          output_tokens: event.output_tokens,
          duration_ms:   event.duration_ms,
          cost_usd:      event.cost_usd,
          timeline:      pendingTimeline[stepKey] ?? [],
          input:         pendingStepContext[stepKey],
        }
        delete pendingTimeline[stepKey]
        delete pendingStepContext[stepKey]

        const newStep: RunStep = {
          step:          event.step,
          run_id:        rid2,
          cost_usd:      event.cost_usd,
          duration_ms:   event.duration_ms,
          input_tokens:  event.input_tokens,
          output_tokens: event.output_tokens,
          ts_ms,
          tool_ms,
        }
        if (isChild) {
          set((s) => ({
            columns: s.columns.map((c) => {
              if (c.id !== columnId) return c
              const child = c.children[rid2]
              if (!child) return c
              return {
                ...c,
                children: { ...c.children, [rid2]: { ...child, steps: [...child.steps, newStep] } },
              }
            }),
          }))
          appendChildStepTrace(rid2, trace)
        } else {
          set((s) => ({
            columns: s.columns.map((c) =>
              c.id === columnId
                ? { ...c, steps: [...c.steps, newStep] }
                : c
            ),
          }))
          appendStepTrace(trace)
        }

      } else if (event.type === 'compact') {
        const systemMsg: ChatMessage = {
          role:    'system',
          content: `Context compacted · ${event.before_msgs}→${event.after_msgs} messages · ~${Math.round(event.before_tokens / 1000)}k→~${Math.round(event.after_tokens / 1000)}k tokens`,
        }
        if (!isChild) {
          set((s) => ({
            columns: s.columns.map((c) => {
              if (c.id !== columnId) return c
              const msgs = [...c.messages]
              const insertAt = msgs.length > 0 && msgs[msgs.length - 1].streaming
                ? msgs.length - 1
                : msgs.length
              msgs.splice(insertAt, 0, systemMsg)
              return { ...c, messages: msgs }
            }),
          }))
        }

      } else if (event.type === 'done') {
        const isRunError = event.exit_reason === 'error' && !!event.error
        set((s) => ({
          columns: s.columns.map((c) => {
            if (c.id !== columnId) return c
            const msgs = c.messages.map((m) => m.streaming ? { ...m, streaming: false } : m)
            // Also stop streaming on all children
            const children = Object.fromEntries(
              Object.entries(c.children).map(([k, ch]) => [k, {
                ...ch,
                status: 'done' as const,
                messages: ch.messages.map((m) => m.streaming ? { ...m, streaming: false } : m),
              }])
            )
            return {
              ...c,
              status:    isRunError ? 'error' : 'done',
              error:     isRunError ? event.error! : null,
              messages:  msgs,
              children,
              result:    isRunError ? null : {
                exit_reason:         event.exit_reason,
                steps:               event.steps,
                total_cost:          event.total_cost,
                total_input_tokens:  event.total_input_tokens  ?? 0,
                total_output_tokens: event.total_output_tokens ?? 0,
                passed:              event.passed,
              },
              stopStream: null,
            }
          }),
        }))

      } else if (event.type === 'error') {
        set((s) => ({
          columns: s.columns.map((c) => {
            if (c.id !== columnId) return c
            const msgs = c.messages.map((m) => m.streaming ? { ...m, streaming: false } : m)
            return { ...c, status: 'error', error: event.message, messages: msgs, stopStream: null }
          }),
        }))
      }
    })

    updateColumn(columnId, { stopStream: stop })
  },

  stopRun: (columnId) => {
    const col = get().columns.find((c) => c.id === columnId)
    if (!col) return
    if (col.status !== 'running') return
    if (col.run_id) {
      void api.cancelRun(col.run_id).catch(() => {
        // best-effort cancel; keep UI stable even if backend cancellation fails
      })
    }
    col.stopStream?.()
    set((s) => ({
      columns: s.columns.map((c) => {
        if (c.id !== columnId) return c
        const msgs = c.messages.map((m) => m.streaming ? { ...m, streaming: false } : m)
        const last = msgs[msgs.length - 1]

        if (last?.role === 'assistant' && (last.streaming || !last.content.trim())) {
          msgs[msgs.length - 1] = {
            ...last,
            streaming: false,
            content: USER_INTERRUPTED_MESSAGE,
            blocks: [{ type: 'text', content: USER_INTERRUPTED_MESSAGE }],
          }
        } else if (!(last?.role === 'assistant' && last.content.trim() === USER_INTERRUPTED_MESSAGE)) {
          msgs.push({
            role: 'assistant',
            content: USER_INTERRUPTED_MESSAGE,
            blocks: [{ type: 'text', content: USER_INTERRUPTED_MESSAGE }],
          })
        }

        return {
          ...c,
          status: 'done',
          stopStream: null,
          messages: msgs,
          result: c.result ?? {
            exit_reason: 'interrupted',
            steps: c.steps.length,
            total_cost: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            passed: null,
          },
        }
      }),
    }))
  },

  loadHistorySession: (columnId, sessionId, rawMsgs) => {
    const messages: ChatMessage[] = rawMsgs.map((m) => {
      if (m.role === 'user' || m.role === 'system') {
        // Build query_context from backend data if present.
        const qc = m.query_context
        const query_context: ChatMessage['query_context'] = qc
          ? {
              system:                 qc.system ?? '',
              tool_names:             qc.tool_names ?? [],
              on_task_start_triggers: qc.on_task_start_triggers ?? [],
              post_query_triggers:    qc.post_query_triggers ?? [],
            }
          : undefined
        const content = typeof m.content === 'string' ? m.content : Array.isArray(m.content)
          ? m.content.filter((b: any) => b?.type === 'text').map((b: any) => b.text ?? '').join('\n')
          : String(m.content ?? '')
        return { role: m.role as 'user' | 'system', content, query_context }
      }
      // assistant — build blocks
      const blocks: MessageBlock[] = []
      const aContent = typeof m.content === 'string' ? m.content : Array.isArray(m.content)
        ? m.content.filter((b: any) => b?.type === 'text').map((b: any) => b.text ?? '').join('\n')
        : String(m.content ?? '')
      if (aContent) blocks.push({ type: 'text', content: aContent })
      for (const tc of m.tool_calls) {
        blocks.push({ type: 'tool_use', id: tc.id, name: tc.name, input: {} })
        if (tc.output !== undefined) {
          blocks.push({
            type: 'tool_result',
            id: tc.id,
            name: tc.name,
            output: tc.output,
            error: null,
            duration_ms: 0,
          })
        }
      }
      return {
        role: 'assistant' as const,
        content: aContent,
        blocks,
        stepTraces: (m.step_traces?.length ?? 0) > 0 ? (m.step_traces as StepTrace[]) : undefined,
      }
    })

    set((s) => ({
      columns: s.columns.map((c) =>
        c.id === columnId
          ? {
              ...c,
              messages,
              steps:           [],
              result:          null,
              error:           null,
              status:          'idle',
              isReadOnly:      true,
              historySessionId: sessionId,
              session_id:      null,  // not yet resumed
            }
          : c
      ),
    }))
  },

  resumeHistorySession: (columnId) => {
    set((s) => ({
      columns: s.columns.map((c) =>
        c.id === columnId
          ? { ...c, isReadOnly: false, session_id: c.historySessionId }
          : c
      ),
    }))
  },
}))
