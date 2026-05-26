import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertCircle } from 'lucide-react'
import SourceCitation from './SourceCitation'

function formatTime(timestamp) {
  if (!timestamp || timestamp === 0) return ''
  try {
    const date = new Date(timestamp)
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', hour12: true }).toLowerCase()
  } catch {
    return ''
  }
}

function UserBubble({ content, id }) {
  const time = formatTime(id)

  return (
    <div className="flex justify-end w-full mb-1">
      <div className="max-w-[80%] md:max-w-[70%] bg-gradient-to-br from-[#f03e3e] to-[#d93232] bubble-shadow-user rounded-2xl rounded-tr-sm px-6 py-2 text-sm text-white leading-relaxed">
        <div className="whitespace-pre-wrap">{content}</div>
        {time && (
          <div className="text-[10px] text-white/70 text-right mt-1.5 font-light tracking-wide select-none">
            {time}
          </div>
        )}
      </div>
    </div>
  )
}

function AssistantBubble({ content, sources, streaming, isError, id }) {
  const time = formatTime(id)

  return (
    <div className="flex justify-start w-full mb-1">
      <div className="max-w-[85%] md:max-w-[80%] min-w-[200px]">
        {/* Chat bubble body */}
        <div className={`bg-[#1b1c21] border border-[#2a2b30] bubble-shadow-assistant rounded-2xl rounded-tl-sm px-6 py-2 transition-all ${
          isError ? 'border-red-500/30 bg-red-950/10' : ''
        }`}>
          {isError && (
            <div className="flex items-center gap-2 text-red-400 font-medium text-xs mb-1.5">
              <AlertCircle size={14} />
              <span>Response Error</span>
            </div>
          )}

          {content ? (
            <div className={`prose-dark text-sm ${streaming ? 'cursor-blink' : ''}`}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
            </div>
          ) : (
            <div className="flex gap-1 py-2 items-center">
              <span className="w-1.5 h-1.5 bg-[#f03e3e] rounded-full animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 bg-[#f03e3e] rounded-full animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 bg-[#f03e3e] rounded-full animate-bounce [animation-delay:300ms]" />
            </div>
          )}

          {time && (
            <div className="text-[10px] text-gray-500 text-right mt-2 font-light tracking-wide select-none">
              {time}
            </div>
          )}
        </div>

        {/* Source citation outside bubble, matching the original design layout */}
        {!streaming && sources && sources.length > 0 && (
          <div className="px-1.5">
            <SourceCitation sources={sources} />
          </div>
        )}
      </div>
    </div>
  )
}

export default function MessageBubble({ message }) {
  if (message.role === 'user') {
    return <UserBubble content={message.content} id={message.id} />
  }
  return (
    <AssistantBubble
      content={message.content}
      sources={message.sources}
      streaming={message.streaming}
      isError={message.isError}
      id={message.id}
    />
  )
}
