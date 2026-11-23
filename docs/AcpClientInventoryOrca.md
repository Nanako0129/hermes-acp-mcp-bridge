# Orca 的 ACP client 實作盤點

- App：`/Applications/Orca.app`，bundle id `com.stablyai.orca`，版本 **1.4.265**（`Info.plist` 與 `package.json` 一致）
- 方法：只做靜態閱讀，沒有啟動 Orca，也沒有執行 bundle 內的任何程式。`app.asar` 本體只有 35 個檔（用 Python 解析 asar 抽出，npx 被 hook 擋下）；ACP 程式碼放在 **`Contents/Resources/app.asar.unpacked/`**，沒有包進 asar，所以直接原地讀取。
- 沒有用 `@agentclientprotocol/sdk` 或 `@zed-industries/agent-client-protocol`：JSON-RPC peer、zod schema、client 都是 Orca 自己寫的，打包時一起 bundle 進去。

證據檔縮寫（都在 `/Applications/Orca.app/Contents/Resources/app.asar.unpacked/out/main/chunks/` 底下）：
- **A** = `claude-stream-json-connection-DuHGc1qy.js`：ACP 協定層、各 agent 的 spec 與 dialect
- **B** = `structured-agent-session-host-assembly-7nH3EPCf.js`：session host，處理權限、重開與事件
- **M** = asar 內的 `out/main/index.js`：組裝 wiring
- `@N` 是檔案內的字元位移（檔案經過 minify，大多只有一行）

## 結論

**Orca 是完整的 ACP client，而且做得相當周全。** 它的「Chat」介面（native chat）會用 ACP 經 stdio 驅動多數 agent；「CLI」選項則是在終端機裡開原生 CLI。每個 agent 用哪一種，由使用者的「Chat | CLI」設定決定，也可以用 `--surface chat|terminal` 覆寫（`app.asar.unpacked/out/cli/specs/worktree-create.js`、`agent-launch.js`）。

走 ACP 的 agent（A @279127 的 `yw` 陣列）：

| agent | 指令 |
|---|---|
| antigravity | Orca 自行下載的 `agy_acp_server.par` 1.5.0 |
| hermes | `hermes acp` |
| kimi | `kimi acp` |
| copilot | `copilot --acp` |
| cursor | `cursor-agent ...` |
| grok | `grok agent [--always-approve] stdio` |
| opencode / opencode2 | `opencode acp` |
| omp | `omp acp` |

**不走 ACP 的 agent**：claude 透過 `@anthropic-ai/claude-agent-sdk` 和 stream-json 驅動，codex 透過自家 app-server transport，pi 透過 `jsonl-rpc`（A @294147、@311325、@313218）。

## 功能檢查表

| 功能 | Orca 的行為 | 證據 |
|---|---|---|
| transport | 每個 chat spawn 一個 child process，用 stdio 傳 NDJSON JSON-RPC。框架有行長上限，非 JSON 的行會交給 `onOtherLine`（用來抓登入連結），其餘丟棄並記錄 diagnostic | A @216961 前的 `Sy` 類別、`My` 類別 |
| `initialize` | 送出 `protocolVersion:1`，回應不是 1 就關閉連線並報 `-32602`。`initialize()` 會被 memoize，失敗時清掉以便重試 | A @218823：`this.call(\`initialize\`,{protocolVersion:1,clientCapabilities:{fs:{readTextFile:!1,writeTextFile:!1},terminal:!1,...}` |
| clientCapabilities | 基底一律是 **fs 讀／寫 false、terminal false**。各 dialect 再疊加：多數加 `elicitation:{form:{}}`；opencode 加 `_meta:{"opencode/child-session-updates":true}`；另一個 dialect（VC）加 `_meta:{parameterizedModelPicker:true}` | A @173845、@177618、@202530、@274251 |
| clientInfo | 預設 `{name:"orca",version:"1"}`。**Antigravity 版本低於 1.5.1 時改送 `{name:"zed",version:"0.0.0"}`**（偽裝成 Zed） | A @266224：`function oC(e){return e&&c.X(e)&&c.q(e,\`1.5.1\`)>=0?mv:{name:\`zed\`,version:\`0.0.0\`}}` |
| `authenticate` | 讀取 `initialize` 回傳的 `authMethods`。`session/new`／`load`／`resume` 回報需要驗證時，用 spec 選定的 `authMethodId` 呼叫 `authenticate`，再重試一次。另有獨立的 `authenticate(methodId)`。Antigravity 依設定選 `oauth-personal` 或 `oauth-business`（後者要填 GCP project 與 location） | A `Ey` @216961、`nS`／`iS` |
| `session/new` | 參數：`{cwd, mcpServers, _meta?, authMethodId...}`；`authMethodId` 會先剝掉再送出。回應 schema 接受 `modes`、`configOptions`、`models` | A `Ey` |
| `session/load` 與 `session/resume` | 有既有 sessionId 時：agent 宣告 `loadSession:true`，而且 spec 不偏好 resume，或沒宣告 `sessionCapabilities.resume` 時用 **load**；否則宣告了 `sessionCapabilities.resume` 就用 **resume**。兩者都沒有時報 `-32601 "ACP agent cannot load or resume this session"`。Antigravity 設 `resumePreference:"resume"`。load 失敗會退回 `session/new` | A `Ey`；B @820176 一帶（`a.start({cwd:t,mcpServers:i.mcpServers??[],sessionId:y.sessionId,...})` 失敗後走 `b=await a.start({cwd:i.cwd,...})`） |
| history replay | load／resume 時重放的 `session/update` 會在 `_meta` 加上 `isReplay:true`。`historyCatchUp`（`prompt-id` 或 `user-message-id`）用來去重 | B `NN`、`VM` |
| `session/prompt` | `{sessionId, prompt:[ContentBlock], _meta?}`，有 prompt 佇列；omp 啟用 `retryBusyPrompts`。prompt 會用到的 block：`text`（可帶 `_meta.displayText`）、`image`、`resource_link`（`@檔案` 提及會轉成 `file://` 連結，需 dialect 設 `linksFileMentions`）、`resource`。有大小與張數上限（20 MiB 等常數） | A @218823 的 `prompt()`；B 的 `cA` schema 與 `_P`／`gP`（"ACP prompt refused"） |
| `session/cancel` | 以 **notification** 送出 `{sessionId, _meta?}`，同時在本地取消佇列中的 prompt。session 尚未開始就 reject | A：`cancel(e={}){...this.prompts.cancel();...this.peer.notify(\`session/cancel\`,t)}` |
| `session/update` 處理 | 先用 zod 驗證外層 envelope；認不得的 update 會轉送，只記一次 diagnostic，**不會報錯**。會解析的類型：`agent_message_chunk`、`agent_thought_chunk`、`user_message_chunk`、`tool_call`、`tool_call_update`、`plan`、`available_commands_update`、`current_mode_update`、`config_option_update`、`usage_update`、`session_info_update`。各 dialect 另有 `normalizeToolUpdate`，修正各 agent 不一致的 rawInput／rawOutput | A `handleNotification`、`Fp`；B `PN` |
| `session/request_permission` | 不屬於目前 session 的請求一律回 `cancelled`。chat 還不能接受請求時也回 `cancelled`。**full-access 模式**下，spec 沒有 `permissionModeConfig` 或設了 `fullAccessAnswersRequests` 時，**自動選第一個 `kind:"allow_once"` 選項**；其他情況顯示 UI 卡片，使用者只能選 agent 提供的選項。handler 拋錯或選了不存在的選項時回 `cancelled` | A `handleRequest`、`xv`、`Sv`、`gv={outcome:{outcome:\`cancelled\`}}`；B `onPermission` @832280 前 |
| 權限模式 | 設定值 ask、accept-edits、full-access 在 session 開啟後透過 **`session/set_config_option`** 寫進 agent 的 mode config option，並驗證回傳的 `currentValue`，不符就中止。對照：Antigravity 為 `default`／`auto_edit`／`yolo`；Hermes 為 `ask`／`workspace_session`／`session`；Kimi 為 `default`／`yolo`；grok 改用啟動參數 `--always-approve` | B @820176：`await a.setConfigOption(r,n,void 0,({configOptions:e})=>{...if(...currentValue!==n)throw Error(\`Permission mode could not be confirmed\`)` |
| `session/set_model` | 有 model config option 時走 `session/set_config_option`，否則走 `session/set_model`。omp 改送 `/model X` 文字 prompt（`modelCommand`） | B `cN`；A `rg.modelCommand` |
| `session/set_mode` | 有實作 `setMode(modeId,_meta)`。實際用在哪些地方 **unverified**（權限改走 config option） | A `setMode` |
| `session/set_config_option` | 送出 `{configId, sessionId, value, _meta?}`，布林值會加 `type:"boolean"` | A `setConfigOption` |
| `fs/read_text_file`、`fs/write_text_file` | **不支援**：capability 宣告為 false。若 agent 仍然送來 `fs/*` 或 `terminal/*`，不產生任何 UI 事件（`wv=/^(?:fs\|terminal)\//`）。實際回給 agent 的 JSON-RPC 是錯誤還是空值 **unverified** | A @199114 `Ev` |
| `terminal/*` | **不支援**（同上） | 同上 |
| `elicitation/create` | 支援 form 與 url 兩種模式，可回 accept、decline、cancel。Antigravity 的 dialect 會把 "The agent has N questions" 轉成問題卡；另外把 `toolCallId` 以 `interaction_` 開頭的權限請求當成單選問題 | A `jv`、`Mv`；schema 在 A @352228 |
| 其他 agent→client 請求 | dialect 不認得的方法轉成 `provider.frame` 事件。連線層沒有 `onRequest` 時回 `-32601 "Unknown ACP client method"`。支援收到的 `$/cancel_request` | A `Ev`、`dispatch` |
| `session/close` | 用 request 送出，timeout 10 秒，逾時改走 cancel | A `My.closeSession` |
| `session/list`、`session/fork`、`session/delete` | 都有實作。fork 送 `mcpServers:[]`。delete 用在 model 與指令探測後清掉拋棄式 session（需 agent 宣告 `sessionCapabilities.delete`） | A `listSessions`、`forkSession`、`deleteSession`；B 指令探測 |
| extension 方法 | `requestExtension`（方法名必須以 `_` 開頭，timeout 30 秒）與 `requestSessionFreeExtension`。各 dialect 會攔截 extension notification，例如 `_opencode/session/child_update`、`_x.ai/session/prompt_complete` | A `requestExtension`、`dg`、`Vh` |
| `mcpServers` | 來源是 Orca 的 **bot 或資料夾工具**設定（`resolveSessionToolServers` 會依 workspace 綁定的 bot 解析），其他情況送空陣列 `[]`。轉換規則：URL 型轉成 `{type:"http",name,url,headers:[{name,value}]}`；stdio 型轉成 `{name,command,args,env:[...]}`（不帶 type）。**不會產生 `sse`**。命令探測與 fork 一律送 `[]` | A @413689 `function JI(e){return e.servers.map(e=>\`url\`in e?{type:\`http\`,...}:{name:e.name,command:e.command,args:e.args,env:qI(e.env)})}`；M `resolveSessionToolServers`；`orcad-retained-source-CIgRydIS.js` 的 `fin` |
| `_meta` 用法 | `prompt`、`cancel`、`set_mode`、`set_model`、`set_config_option`、`session/new` 都可以帶 `_meta`。Antigravity 在 `session/new` 帶 `_meta.antigravity = {disabledTools:["run_workflow"], systemPrompt:{append:<standing instructions>}}`（由 `oG` 深層合併，重開時也帶）。grok 在 probe session 帶 `_meta:{yoloMode:false,autoMode:false}` | A @266224 的 `sC.sessionMeta`、`N_`；A `oG`；A probe 段 |
| standing instructions | 依 agent 而異：Antigravity 放 `_meta.antigravity.systemPrompt.append`（agent 必須在 `agentCapabilities._meta` 宣告支援，否則不啟動）；omp 用 `--append-system-prompt=<file>`；opencode 用 `OPENCODE_CONFIG_CONTENT` 環境變數 | A `N_`、`F_`、`P_`、`KS` |
| process 生命週期 | **每個 chat 一個 process**（`jy`／`My`），一個 connection 只開一個 session。spawn 經 `@orca/process-host`，結束時先 graceful exit 再強制結束，並驗證整棵 process tree 已經退出。crash 時 `onClose`／`onExit` 會帶 stderr 尾段（8 KiB）的錯誤，並區分是預期關閉還是連線中途遺失（`lostBeforeExit`），交給上層重新 attach，用 load 或 resume 接回同一個 session。agent 列印登入連結時 Orca 會擷取並關閉連線。model 目錄探測另外 spawn 短命 process（one-shot，timeout 15 秒）。Hermes 設 `sharedToolProcess:true`（會剝掉 `ORCA_TERMINAL_HANDLE` 等環境變數）、`keepsEverySession:true`。process 跨 chat **重用 unverified**；從程式碼看是每個 chat 各自一個 | A `My`、`zy`／`Ry`、`dT`；B `IN` |
| 版本門檻 | 各 agent 有最低版本：hermes ≥ 0.23.0、kimi ≥ 2.5.0、omp ≥ 17.0.5、opencode 1.18.31 或 2.0.14 | A agent specs |
| Antigravity 特別處理 | 不用使用者安裝的 `agy`：Orca 從 `https://dl.google.com/agy-extensions/releases/macos/agy-acp-server-1.5.0-darwin-arm64.zip` 下載 `agy_acp_server.par` 與 `localharness_external`，比對 sha256 後放進 `agent-runtimes/antigravity`，並忽略使用者自訂的指令設定（`ignoreAgentCommandSetting`）。其他處理：獨立的 `GEMINI_HOME`；在私有暫存目錄執行；從 macOS `scutil --proxy` 注入 proxy 環境變數；把 `~/.gemini/GEMINI.md` 與 repo 內的指示檔當作額外指示；權限請求 `_meta["agy.security.warning"]` 會顯示成說明文字；"execute" 工具名稱另行正規化；不提供 `logout` 指令 | A `iS`…`uS`、`sC`、`JS`、`Fv`、`Gv`、`tL` |
| Hermes 特別處理 | 指令 `hermes acp`。另有 `compactCommand:"/compress"`、不提供 `compress` 與 `reset` 指令、`reopenMovesSession`、`modelDiscovery: unavailable`（理由：Hermes 只在無法刪除的 session 裡列出 model）、透過 SQLite 匯入對話紀錄、支援 `HERMES_HOME` 下的 profiles | A @279127 `hw`、`cw` |

## 給 Python client 對照用的重點

1. clientCapabilities 刻意保守：沒有 fs、沒有 terminal，只有 elicitation form。
2. 一定送 `mcpServers`，至少是 `[]`。HTTP 型帶 `type:"http"`；stdio 型不帶 type，env 是 `{name,value}` 陣列。
3. 有 sessionId 時：有 `loadSession` 就 load，有 `sessionCapabilities.resume` 就 resume；兩者都沒有時，重開流程退回 `session/new`。
4. 權限：預設交給 UI 人工選；full-access 時自動選 `allow_once`；非目前 session 或尚未就緒時回 `cancelled`。
5. 權限模式用 `session/set_config_option` 設定並驗證回傳值；`session/cancel` 是 notification；結束時送 `session/close`，10 秒逾時，之後確認 process tree 已收掉。

## Unverified

- `fs/*`、`terminal/*` 請求實際回給 agent 的 JSON-RPC 內容
- `session/set_mode` 的實際呼叫點
- process 會不會跨 chat 共用
- copilot、cursor、kimi 各自的完整 dialect 細節（只看了片段）

以上都沒有用執行中的 Orca 觀察過，全部來自靜態閱讀。
