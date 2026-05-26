import { useState, useRef, useEffect } from 'react'
import { 
  Send, Square, Trash2, Paperclip, Globe, Database, Bot, Zap, X, 
  CheckCircle, AlertCircle, Loader, FileX, Smile, Camera 
} from 'lucide-react'
import { useChat } from '../hooks/useChat'
import { uploadDocument, clearDocuments, getSources } from '../services/api'
import MessageBubble from './MessageBubble'
import StreamingIndicator from './StreamingIndicator'

const WELCOME_MESSAGE = {
  role: 'assistant',
  id: 0,
  content: `Hello! I'm your Advanced RAG Assistant.

I can help you with:
- **Document Q&A** — Attach a PDF, TXT, or DOCX using the 📎 button and ask questions about it
- **Web Search** — Get real-time information from the web
- **Hybrid Retrieval** — Combine document knowledge with web search

Click the **paperclip icon** inside the chat input box to upload a file, then start chatting!`,
  sources: [],
  streaming: false,
}

function UploadBadge({ file, status, message, onRemove }) {
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold border transition-all ${
      status === 'uploading' ? 'bg-rose-950/20 border-[#f03e3e]/40 text-rose-300' :
      status === 'done'      ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300' :
      status === 'error'     ? 'bg-red-950/20 border-red-500/30 text-red-300' :
                               'bg-[#1b1c21] border-[#2a2b30] text-gray-300'
    }`}>
      {status === 'uploading' && <Loader size={11} className="animate-spin text-[#f03e3e]" />}
      {status === 'done'      && <CheckCircle size={11} className="text-emerald-400" />}
      {status === 'error'     && <AlertCircle size={11} className="text-red-400" />}
      {status === 'pending'   && <Paperclip size={11} className="text-gray-400" />}
      <span className="max-w-[160px] truncate">{file.name}</span>
      {message && <span className="opacity-75 font-normal">· {message}</span>}
      {status !== 'uploading' && (
        <button onClick={onRemove} className="ml-1 hover:text-[#f03e3e] transition-colors">
          <X size={11} />
        </button>
      )}
    </div>
  )
}

export default function ChatWindow() {
  const { messages, isStreaming, statusText, sendMessage, cancelStream, clearHistory } = useChat()
  const [input, setInput]               = useState('')
  const [searchMode, setSearchMode]     = useState(null)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [indexedDocs, setIndexedDocs]   = useState([])
  const [clearing, setClearing]         = useState(false)
  const bottomRef    = useRef(null)
  const textareaRef  = useRef(null)
  const fileInputRef = useRef(null)

  const refreshSources = async () => {
    try {
      const sources = await getSources()
      setIndexedDocs(sources)
    } catch { /* non-critical */ }
  }

  useEffect(() => { refreshSources() }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isStreaming])

  // ── File handling ──────────────────────────────────────────────────────────
  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''

    const ALLOWED = ['.pdf', '.txt', '.docx', '.md']
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!ALLOWED.includes(ext)) {
      setUploadedFile({ file, status: 'error', message: 'Unsupported type' })
      return
    }
    if (file.size > 50 * 1024 * 1024) {
      setUploadedFile({ file, status: 'error', message: 'Max 50 MB' })
      return
    }

    setUploadedFile({ file, status: 'uploading', message: '' })
    try {
      const result = await uploadDocument(file)
      setUploadedFile({ file, status: 'done', message: `${result.chunks_indexed} chunks indexed` })
      refreshSources()
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Upload failed'
      setUploadedFile({ file, status: 'error', message: msg })
    }
  }

  // ── Clear all indexed documents ────────────────────────────────────────────
  const handleClearDocuments = async () => {
    const count = indexedDocs.length
    if (!window.confirm(`Remove all ${count} indexed document${count !== 1 ? 's' : ''}? You can upload new ones after.`)) return
    setClearing(true)
    try {
      await clearDocuments()
      setIndexedDocs([])
      setUploadedFile(null)
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to clear documents')
    } finally {
      setClearing(false)
    }
  }

  // ── Send ───────────────────────────────────────────────────────────────────
  const handleSend = () => {
    const text = input.trim()
    if (!text || isStreaming) return
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.focus()
      textareaRef.current.style.height = 'auto' // Reset height after send
    }
    sendMessage(text, { useWebSearch: searchMode })
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const allMessages = messages.length === 0 ? [WELCOME_MESSAGE] : messages

  return (
    <div className="flex flex-col h-screen overflow-hidden">

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-3.5 border-b border-[#2d2e35]/50 bg-[#16171b]/80 backdrop-blur-md shadow-lg z-10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-[#f03e3e] to-[#d93232] rounded-xl flex items-center justify-center shadow-md shadow-red-500/10">
            <Zap size={18} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-gray-100 tracking-wide">RAG Chatbot</h1>
            <p className="text-[10px] text-gray-500 font-medium">Hybrid Retrieval · Agentic Reasoning · Groq</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Indexed docs chip — shows count + clear button */}
          {indexedDocs.length > 0 && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[#1b1c21] rounded-xl border border-[#2d2e35]/80 shadow-inner">
              <Database size={12} className="text-[#f03e3e] flex-shrink-0" />
              <span className="text-[11px] font-semibold text-gray-300">
                {indexedDocs.length} doc{indexedDocs.length !== 1 ? 's' : ''}
              </span>
              <button
                onClick={handleClearDocuments}
                disabled={clearing}
                title="Clear all indexed documents"
                className="ml-1 text-gray-500 hover:text-[#f03e3e] transition-colors disabled:opacity-40"
              >
                {clearing
                  ? <Loader size={11} className="animate-spin" />
                  : <FileX size={11} />}
              </button>
            </div>
          )}

          {/* Clear conversation */}
          <button
            onClick={clearHistory}
            title="Clear conversation"
            className="p-2 text-gray-500 hover:text-red-400 hover:bg-[#1b1c21] rounded-xl border border-transparent hover:border-[#2d2e35]/60 transition-all duration-200"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </header>

      {/* ── Messages viewport ── */}
      <div className="flex-1 overflow-y-auto px-4 py-6 scrollbar-thin">
        <div className="max-w-2xl mx-auto space-y-5">
          {allMessages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          {isStreaming && statusText && <StreamingIndicator statusText={statusText} />}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* ── Input area ── */}
      <div className="border-t border-[#2d2e35]/40 bg-[#121316]/90 backdrop-blur-lg px-4 py-4 z-10">
        <div className="max-w-2xl mx-auto">

          {/* Search mode pills */}
          <div className="flex gap-1.5 mb-3.5">
            {[
              { value: false, label: 'Documents', Icon: Database },
              { value: null,  label: 'Auto',      Icon: Bot },
              { value: true,  label: 'Web Search', Icon: Globe },
            ].map(({ value, label, Icon }) => (
              <button
                key={String(value)}
                onClick={() => setSearchMode(value)}
                className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 ${
                  searchMode === value
                    ? 'bg-[#f03e3e] text-white shadow-sm hover:bg-[#e84343]'
                    : 'bg-[#1b1c21] text-gray-400 border border-[#2d2e35]/80 hover:bg-[#202126] hover:text-gray-300'
                }`}
              >
                <Icon size={12} /> {label}
              </button>
            ))}
          </div>

          {/* Uploaded file badge */}
          {uploadedFile && (
            <div className="mb-2">
              <UploadBadge
                {...uploadedFile}
                onRemove={() => setUploadedFile(null)}
              />
            </div>
          )}

          {/* Input container - matching reference image */}
          <div className="flex items-center gap-3">
            
            {/* Pill-shaped container for icons and text input */}
            <div className="flex-1 flex items-center gap-2 bg-[#16171b] border border-[#282930] rounded-full input-pill-shadow focus-within:border-[#f03e3e]/30 transition-all duration-300 px-4 py-1.5">
              
              {/* Dummy emoji button representing reference */}
              <button 
                type="button" 
                title="Emojis"
                className="text-[#f03e3e] hover:opacity-80 transition-opacity p-1.5"
              >
                <Smile size={18} />
              </button>
              
              {/* Attachment file input handler */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.docx,.md"
                onChange={handleFileChange}
                className="hidden"
              />

              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploadedFile?.status === 'uploading'}
                title="Attach file (PDF, TXT, DOCX, MD)"
                className={`text-[#f03e3e] hover:opacity-80 transition-opacity p-1.5 ${
                  uploadedFile?.status === 'uploading' ? 'opacity-30 cursor-not-allowed' : ''
                }`}
              >
                <Paperclip size={18} />
              </button>

              {/* Dummy camera button representing reference */}
              <button 
                type="button" 
                title="Camera"
                className="text-[#f03e3e] hover:opacity-80 transition-opacity p-1.5"
              >
                <Camera size={18} />
              </button>

              {/* Text Input area */}
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message..."
                rows={1}
                disabled={isStreaming}
                className="flex-1 bg-transparent text-gray-200 placeholder-gray-500 text-sm leading-normal resize-none outline-none py-1.5 px-1 max-h-24 overflow-y-auto"
                style={{ minHeight: '24px' }}
                onInput={(e) => {
                  e.target.style.height = 'auto'
                  e.target.style.height = `${e.target.scrollHeight}px`
                }}
              />
            </div>

            {/* Separate circular send/stop button on the right */}
            {isStreaming ? (
              <button
                onClick={cancelStream}
                className="flex-shrink-0 w-11 h-11 bg-[#f03e3e] hover:bg-[#e84343] rounded-full flex items-center justify-center transition-all duration-300 floating-btn-shadow hover:scale-105 active:scale-95 z-10"
                title="Stop generating"
              >
                <Square size={16} className="text-white fill-white" />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                className="flex-shrink-0 w-11 h-11 bg-[#f03e3e] hover:bg-[#e84343] disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:shadow-none rounded-full flex items-center justify-center transition-all duration-300 floating-btn-shadow hover:scale-105 active:scale-95 z-10"
                title="Send"
              >
                <Send size={16} className="text-white ml-0.5" />
              </button>
            )}
          </div>

          <p className="text-center text-[10px] text-gray-600 mt-2 font-medium tracking-wide">
            Attach PDF/TXT/DOCX/MD · Groq · FAISS · Tavily
          </p>
        </div>
      </div>
    </div>
  )
}
