import { useUIStore } from '../store/ui'

type Dict = Record<string, string>

const en: Dict = {
  // ── Navigation ──────────────────────────────────────────────
  'nav.builder':  'Builder',
  'nav.compare':  'Compare',

  // ── Sidebar ─────────────────────────────────────────────────
  'sidebar.slot_config':      'Slot Config',
  'sidebar.presets':          'Presets',
  'sidebar.examples':         'Examples',
  'sidebar.custom':           'Custom',
  'sidebar.new_blank':        'New blank harness…',
  'sidebar.new_blank_hint':   'Create empty harness — configure all processors from scratch',
  'sidebar.save_current':     'Save current as custom…',
  'sidebar.save_current_hint':'Saves the current harness config as a new custom entry',
  'sidebar.import_yaml':      'Import from YAML…',
  'sidebar.save':             'Save',
  'sidebar.create':           'Create',
  'sidebar.duplicate':        'Duplicate',
  'sidebar.delete':           'Delete',
  'sidebar.harness_name_placeholder': 'Harness name…',
  'sidebar.import_yaml_parse_error':  'Could not parse YAML — make sure it is a valid HarnessX config file.',
  'sidebar.broken_processor_hint':    'One or more custom processors are missing from the backend. Delete this harness or re-import the processor.',
  'sidebar.default_agent':    'CLI Agent',
  'sidebar.default_agent_desc': 'Built-in universal agent',

  // ── Builder page ────────────────────────────────────────────
  'builder.empty.title':   'Select a preset or create a custom harness',
  'builder.empty.desc':    'Choose a preset from the sidebar to view and edit its configuration, then start chatting with it.',
  'builder.export_yaml':   'Export YAML',
  'builder.save_custom':   'Save as Custom',
  'builder.copy_example_to_custom': 'Copy Current Example To Custom (Editable)',
  'builder.start_chat':    'Start Chatting →',
  'builder.loading_dims':  'Loading dimensions…',

  // ── Status ───────────────────────────────────────────────────
  'status.idle':    'Idle',
  'status.running': 'Running',
  'status.done':    'Done',
  'status.error':   'Error',
  'status.passed':  '✓ Passed',
  'status.failed':  '✗ Failed',
  'status.stop':    'Stop',
  'status.new_chat':'New chat',
  'status.back':    'Back',

  // ── Chat ────────────────────────────────────────────────────
  'chat.placeholder':          'Describe a task… (Enter to send, Shift+Enter for newline)',
  'chat.compare_placeholder':  'Send a task to all columns… (Enter to send)',
  'chat.advanced':             'Advanced',
  'chat.success_criteria':     'Success criteria',
  'chat.success_placeholder':  'Optional: how to judge success',
  'chat.empty':                'start a conversation',
  'chat.no_model':             'No API key configured for the main model.',
  'chat.no_model_link':        'Configure in Model →',

  // ── Compare page ────────────────────────────────────────────
  'compare.add_column':  'Add column',
  'compare.config':      'Config',
  'compare.empty':       'Add a column to get started.',
  'compare.custom':      'Custom…',
  'compare.drop_hint':   'Drop',  // "Drop 'label' here"
  'compare.applied':     'Applied',
  'compare.columns_of':  'of',    // "2 of 4 columns"
  'compare.no_custom_harness': 'No saved custom harness.',
  'compare.no_custom_harness_hint': 'No saved custom harness. Please go to Builder and save one first.',

  // ── Vendor / Model config ────────────────────────────────────
  'vendor.model':           'Model',
  'vendor.api_key':         'API Key',
  'vendor.base_url':        'Base URL',
  'vendor.api_key_hint':    'Leave blank to use env variable',
  'vendor.base_url_hint':   'Leave blank to use default endpoint',
  'vendor.custom_model_id': 'Custom model ID…',
  'vendor.apply':           'Apply',

  // ── Sections ─────────────────────────────────────────────────
  'section.model_provider': 'Model Providers',
  'section.sandbox':        'Sandbox & Workspace',
  'section.tools':          'Tools',
  'section.skills':         'Skills',

  // ── Providers ────────────────────────────────────────────────
  'provider.add':        '+ Add provider',
  'provider.remove':     'Remove',
  'provider.name':       'Name',
  'provider.main_hint':  'Main provider (used as primary model)',
  'provider.name_hint':  'e.g. compact, judge, vision',
  'provider.model_label':'Model',

  // ── Model Registry ────────────────────────────────────────────
  'model.registry':         'Model Registry',
  'model.registry_desc':    'Define reusable model configs with credentials and capabilities.',
  'model.add':              '+ Add Model',
  'model.remove':           'Remove model',
  'model.display_name':     'Display Name',
  'model.vendor':           'Vendor',
  'model.model_id':         'Model ID',
  'model.backend_impl':     'Backend Impl',
  'model.extra_headers':    'Extra Headers',
  'model.extra_headers_hint':'One per line: Header-Name: value; JSON object is also supported.',
  'model.capabilities':     'Capabilities',
  'model.reasoning':        'Reasoning / Thinking',
  'model.extended_thinking':'Extended thinking',
  'model.thinking_budget_tokens': 'Thinking budget tokens',
  'model.reasoning_effort': 'Reasoning effort',
  'model.reasoning_effort.auto': 'Auto',
  'model.reasoning_effort.low': 'Low',
  'model.reasoning_effort.medium': 'Medium',
  'model.reasoning_effort.high': 'High',
  'model.reasoning_summary': 'Reasoning summary',
  'model.import_yaml':      'Import YAML',
  'model.export_yaml':      'Export YAML',
  'model.save':             'Save',
  'model.saving':           'Saving…',
  'model.save_success':     'Saved to ~/.harnessx/model_config.yaml',
  'model.save_error':       'Save failed',
  'model.import_success':   'Model config imported.',
  'model.import_error':     'Import failed — invalid model config YAML.',
  'model.no_models':        'No models defined yet.',
  'model.provider_models':  'models',
  'model.provider_add':     'Add under provider',
  'model.provider_add_hint':'Reuse this provider config; only model id/display name are required.',

  // ── Model Slots ───────────────────────────────────────────────
  'slot.section':           'Slot Configuration',
  'slot.section_desc':      'Assign registry models to named slots used by harness processors.',
  'slot.add':               '+ Add custom slot',
  'slot.remove':            'Remove slot',
  'slot.strategy':          'Strategy',
  'slot.strategy.primary':  'Primary',
  'slot.strategy.fallback': 'Fallback chain',
  'slot.strategy.round_robin': 'Round-robin',
  'slot.add_model':         '+ Add model',
  'slot.default':           'Default',
  'slot.set_default':       'Set default',
  'slot.drag_reorder':      'Drag to reorder',
  'slot.no_models':         'No models assigned',
  'slot.required':          'Required',

  // ── Sandbox / Workspace ──────────────────────────────────────
  'sandbox.workspace_dir':  'Workspace Directory',
  'sandbox.workspace_hint': 'Skills & system prompts are copied here on launch',

  // ── Tools ────────────────────────────────────────────────────
  'tools.enable_all':  'Enable all',
  'tools.disable_all': 'Disable all',

  // ── Skills ───────────────────────────────────────────────────
  'skills.enable':       'Enable skill loading',
  'skills.auto_inject':  'Matching skills are auto-injected per step:',
  'skills.available':    'Available built-in skills (enable to auto-load):',
  'skills.loading':      'Loading…',

  // ── Top bar ──────────────────────────────────────────────────
  'topbar.model':       'Model',
  'topbar.history':     'History',

  // ── Settings ─────────────────────────────────────────────────
  'settings.title':     'Settings',
  'settings.open':      'Settings',
  'settings.env':       'Environment',
  'settings.back':      'Back to Main',
  'settings.theme':     'Theme',
  'settings.light':     'Light',
  'settings.dark':      'Dark',
  'settings.language':  'Language',
  'settings.font_size': 'Font Size',
  'settings.thinking_post_stream': 'Thinking after stream',
  'settings.thinking.collapse': 'Collapse',
  'settings.thinking.keep': 'Keep open',

  // ── Settings pages ────────────────────────────────────────────
  'settings.page.model':     'Model',
  'settings.page.workspace': 'Workspace',
  'settings.page.tools':     'Tools',
  'settings.page.skills':    'Skills',
  'settings.page.plugins':   'Plugins',

  // ── MCP ───────────────────────────────────────────────────────
  'mcp.section':         'MCP Servers',
  'mcp.add':             'Add MCP Server',
  'mcp.edit':            'Edit',
  'mcp.delete':          'Delete',
  'mcp.preview':         'Preview tools',
  'mcp.no_servers':      'No MCP servers configured.',
  'mcp.transport':       'Transport',
  'mcp.command':         'Command',
  'mcp.url':             'URL',
  'mcp.env':             'Environment',
  'mcp.connecting':      'Connecting…',
  'mcp.tools_empty':     'No tools exposed by this server.',

  // ── Plugins ───────────────────────────────────────────────────
  'plugin.browse':            'Browse files',
  'plugin.remove':            'Remove',
  'plugin.import':            'Import Plugin',
  'plugin.import_hint':       'Enter path to a plugin directory (must contain plugin.json)',
  'plugin.no_plugins':        'No plugins found.',
  'plugin.tools':             'tools',
  'plugin.skills':            'skills',
  'plugin.mcp':               'MCP servers',
  'plugin.scan_dirs':         'Plugin Scan Directories',
  'plugin.scan_dirs_desc':    'Parent directories auto-scanned for plugins on startup.',
  'plugin.scan_add':          '+ Add scan directory',
  'plugin.scan_add_hint':     'Path to a directory containing plugin subdirectories',
  'plugin.scan_empty':        'No extra scan directories. Default: ~/.harnessx/plugins/',
  'plugin.install_hint':      'Auto-discovered from ~/.harnessx/plugins/ · Install via claude plugin install or drop plugin folders into a scan directory.',

  // ── Workspace ──────────────────────────────────────────────────
  'workspace.dir_local':      'Workspace Directory',
  'workspace.dir_remote':     'Remote Workspace Path',
  'workspace.dir_local_hint': 'Skills and system prompts are copied here on launch',
  'workspace.dir_remote_hint':'Working directory on the remote sandbox (e.g. /workspace)',
  'workspace.remote_url':     'Remote Sandbox URL',

  // ── File manager ──────────────────────────────────────────────
  'fs.edit':             'Edit file',
  'fs.save':             'Save',
  'fs.cancel':           'Cancel',
  'fs.read_only':        'Read-only file',
  'fs.empty_dir':        'Empty directory',
  'fs.enter_path':       'Enter a workspace path to browse files.',

}

const zh: Dict = {
  // ── Navigation ──────────────────────────────────────────────
  'nav.builder':  '构建器',
  'nav.compare':  '对比',

  // ── Sidebar ─────────────────────────────────────────────────
  'sidebar.slot_config':      '运行配置',
  'sidebar.presets':          '预设',
  'sidebar.examples':         '示例',
  'sidebar.custom':           '自定义',
  'sidebar.new_blank':        '新建空白 Harness…',
  'sidebar.new_blank_hint':   '从零开始配置所有处理器维度',
  'sidebar.save_current':     '将当前保存为自定义…',
  'sidebar.save_current_hint':'将当前配置另存为新的自定义条目',
  'sidebar.import_yaml':      '从 YAML 导入…',
  'sidebar.save':             '保存',
  'sidebar.create':           '创建',
  'sidebar.duplicate':        '复制',
  'sidebar.delete':           '删除',
  'sidebar.harness_name_placeholder': 'Harness 名称…',
  'sidebar.import_yaml_parse_error':  'YAML 解析失败，请确认是合法的 HarnessX 配置文件。',
  'sidebar.broken_processor_hint':    '一个或多个自定义处理器文件在后端已不存在，请删除此 Harness 或重新导入处理器。',
  'sidebar.default_agent':    'CLI Agent',
  'sidebar.default_agent_desc': '内置通用 Agent',

  // ── Builder page ────────────────────────────────────────────
  'builder.empty.title':   '选择预设或创建自定义 Harness',
  'builder.empty.desc':    '从侧边栏选择预设，查看并编辑配置，然后开始对话。',
  'builder.export_yaml':   '导出 YAML',
  'builder.save_custom':   '保存为自定义',
  'builder.copy_example_to_custom': '复制当前示例到自定义（可编辑）',
  'builder.start_chat':    '开始对话 →',
  'builder.loading_dims':  '正在加载维度…',

  // ── Status ───────────────────────────────────────────────────
  'status.idle':    '空闲',
  'status.running': '运行中',
  'status.done':    '完成',
  'status.error':   '错误',
  'status.passed':  '✓ 通过',
  'status.failed':  '✗ 失败',
  'status.stop':    '停止',
  'status.new_chat':'新对话',
  'status.back':    '返回',

  // ── Chat ────────────────────────────────────────────────────
  'chat.placeholder':          '描述任务… (Enter 发送，Shift+Enter 换行)',
  'chat.compare_placeholder':  '向所有列发送任务… (Enter 发送)',
  'chat.advanced':             '高级设置',
  'chat.success_criteria':     '成功标准',
  'chat.success_placeholder':  '可选：如何判断成功',
  'chat.empty':                '开始对话',
  'chat.no_model':             '主模型未配置 API 密钥。',
  'chat.no_model_link':        '前往 模型 配置 →',

  // ── Compare page ────────────────────────────────────────────
  'compare.add_column':  '添加列',
  'compare.config':      '配置',
  'compare.empty':       '添加列以开始对比。',
  'compare.custom':      '自定义…',
  'compare.drop_hint':   '放置',
  'compare.applied':     '已应用',
  'compare.columns_of':  '/',
  'compare.no_custom_harness': '没有已保存的自定义 Harness。',
  'compare.no_custom_harness_hint': '没有可用的自定义 Harness。请先到 Builder 页面保存一个自定义 Harness，再返回 Compare。',

  // ── Vendor / Model config ────────────────────────────────────
  'vendor.model':           '模型',
  'vendor.api_key':         'API 密钥',
  'vendor.base_url':        '基础 URL',
  'vendor.api_key_hint':    '留空以使用环境变量',
  'vendor.base_url_hint':   '留空以使用默认端点',
  'vendor.custom_model_id': '自定义模型 ID…',
  'vendor.apply':           '应用',

  // ── Sections ─────────────────────────────────────────────────
  'section.model_provider': '模型提供商',
  'section.sandbox':        '沙箱与工作目录',
  'section.tools':          '工具',
  'section.skills':         '技能',

  // ── Providers ────────────────────────────────────────────────
  'provider.add':        '+ 添加提供商',
  'provider.remove':     '移除',
  'provider.name':       '名称',
  'provider.main_hint':  '主提供商（用于主模型）',
  'provider.name_hint':  '例如 compact、judge、vision',
  'provider.model_label':'模型',

  // ── Model Registry ────────────────────────────────────────────
  'model.registry':         '模型注册表',
  'model.registry_desc':    '定义带有凭据和能力标签的可复用模型配置。',
  'model.add':              '+ 添加模型',
  'model.remove':           '移除模型',
  'model.display_name':     '显示名称',
  'model.vendor':           '供应商',
  'model.model_id':         '模型 ID',
  'model.backend_impl':     '后端实现',
  'model.extra_headers':    '额外请求头',
  'model.extra_headers_hint':'每行一个：Header-Name: value；也支持 JSON 对象。',
  'model.capabilities':     '能力标签',
  'model.reasoning':        '推理 / 思考',
  'model.extended_thinking':'扩展思考',
  'model.thinking_budget_tokens': '思考预算 tokens',
  'model.reasoning_effort': '推理强度',
  'model.reasoning_effort.auto': '自动',
  'model.reasoning_effort.low': '低',
  'model.reasoning_effort.medium': '中',
  'model.reasoning_effort.high': '高',
  'model.reasoning_summary': '推理摘要',
  'model.import_yaml':      '导入 YAML',
  'model.export_yaml':      '导出 YAML',
  'model.save':             '保存',
  'model.saving':           '保存中…',
  'model.save_success':     '已保存到 ~/.harnessx/model_config.yaml',
  'model.save_error':       '保存失败',
  'model.import_success':   '模型配置已导入。',
  'model.import_error':     '导入失败 — YAML 格式无效。',
  'model.no_models':        '尚未定义模型。',
  'model.provider_models':  '个模型',
  'model.provider_add':     '同 Provider 新增',
  'model.provider_add_hint':'复用该 Provider 配置，只需填写显示名和 model id。',

  // ── Model Slots ───────────────────────────────────────────────
  'slot.section':           '插槽配置',
  'slot.section_desc':      '将注册表中的模型分配给处理器使用的命名插槽。',
  'slot.add':               '+ 添加自定义插槽',
  'slot.remove':            '移除插槽',
  'slot.strategy':          '路由策略',
  'slot.strategy.primary':  '单一主模型',
  'slot.strategy.fallback': '故障转移链',
  'slot.strategy.round_robin': '轮询',
  'slot.add_model':         '+ 添加模型',
  'slot.default':           '默认',
  'slot.set_default':       '设为默认',
  'slot.drag_reorder':      '拖拽排序',
  'slot.no_models':         '尚未分配模型',
  'slot.required':          '必填',

  // ── Sandbox / Workspace ──────────────────────────────────────
  'sandbox.workspace_dir':  '工作目录',
  'sandbox.workspace_hint': '启动时将自动复制技能与系统提示到此目录',

  // ── Tools ────────────────────────────────────────────────────
  'tools.enable_all':  '启用全部',
  'tools.disable_all': '禁用全部',

  // ── Skills ───────────────────────────────────────────────────
  'skills.enable':       '启用技能加载',
  'skills.auto_inject':  '匹配的技能将在每步自动注入：',
  'skills.available':    '可用内置技能（启用以自动加载）：',
  'skills.loading':      '加载中…',

  // ── Top bar ──────────────────────────────────────────────────
  'topbar.model':       '模型',
  'topbar.history':     '历史',

  // ── Settings ─────────────────────────────────────────────────
  'settings.title':     '设置',
  'settings.open':      '基础设置',
  'settings.env':       '环境',
  'settings.back':      '返回主界面',
  'settings.theme':     '主题',
  'settings.light':     '浅色',
  'settings.dark':      '深色',
  'settings.language':  '语言',
  'settings.font_size': '字体大小',
  'settings.thinking_post_stream': '流式结束后 Thinking',
  'settings.thinking.collapse': '自动折叠',
  'settings.thinking.keep': '保持展开',

  // ── Settings pages ────────────────────────────────────────────
  'settings.page.model':     '模型',
  'settings.page.workspace': '工作区',
  'settings.page.tools':     '工具',
  'settings.page.skills':    '技能',
  'settings.page.plugins':   '插件',

  // ── MCP ───────────────────────────────────────────────────────
  'mcp.section':         'MCP 服务',
  'mcp.add':             '添加 MCP 服务',
  'mcp.edit':            '编辑',
  'mcp.delete':          '删除',
  'mcp.preview':         '查看工具',
  'mcp.no_servers':      '暂无 MCP 服务。',
  'mcp.transport':       '传输方式',
  'mcp.command':         '命令',
  'mcp.url':             'URL',
  'mcp.env':             '环境变量',
  'mcp.connecting':      '连接中…',
  'mcp.tools_empty':     '该服务未暴露任何工具。',

  // ── Plugins ───────────────────────────────────────────────────
  'plugin.browse':            '浏览文件',
  'plugin.remove':            '移除插件',
  'plugin.import':            '导入插件',
  'plugin.import_hint':       '输入插件目录路径（需包含 plugin.json）',
  'plugin.no_plugins':        '未发现插件。',
  'plugin.tools':             '工具',
  'plugin.skills':            '技能',
  'plugin.mcp':               'MCP 服务',
  'plugin.scan_dirs':         '插件扫描目录',
  'plugin.scan_dirs_desc':    '启动时自动扫描以下目录中的插件子目录。',
  'plugin.scan_add':          '+ 添加扫描目录',
  'plugin.scan_add_hint':     '包含插件子目录的父目录路径',
  'plugin.scan_empty':        '未配置额外扫描目录，默认扫描 ~/.harnessx/plugins/',
  'plugin.install_hint':      '自动发现 ~/.harnessx/plugins/ 中的插件 · 也可通过 claude plugin install 安装或将插件文件夹放入扫描目录。',

  // ── Workspace ──────────────────────────────────────────────────
  'workspace.dir_local':      '工作目录',
  'workspace.dir_remote':     '远程工作路径',
  'workspace.dir_local_hint': '启动时将自动复制技能与系统提示到此目录',
  'workspace.dir_remote_hint':'远程沙箱上的工作目录（如 /workspace）',
  'workspace.remote_url':     '远程沙箱 URL',

  // ── File manager ──────────────────────────────────────────────
  'fs.edit':             '编辑文件',
  'fs.save':             '保存',
  'fs.cancel':           '取消',
  'fs.read_only':        '只读文件',
  'fs.empty_dir':        '空目录',
  'fs.enter_path':       '请先设置工作区路径以浏览文件。',

}

const dicts: Record<string, Dict> = { en, zh }

export function useT() {
  const lang = useUIStore((s) => s.lang)
  return (key: string) => dicts[lang]?.[key] ?? dicts.en[key] ?? key
}
