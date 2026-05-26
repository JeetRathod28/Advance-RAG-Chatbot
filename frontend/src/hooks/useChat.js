import { useState, useCallback, useRef, useEffect } from 'react'
import { streamChat } from '../services/api'

const SESSION_KEY = 'rag_session_id'

function getOrCreateSessionId() {
  let id = sessionStorage.getItem(SESSION_KEY)
  if (!id) {
    id = `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
    sessionStorage.setItem(SESSION_KEY, id)
  }
  return id
}

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [statusText, setStatusText] = useState('')
  const cancelRef = useRef(null)
  const sessionId = useRef(getOrCreateSessionId())

  // Clean up on unmount
  useEffect(() => () => cancelRef.current?.cancel(), [])

  const sendMessage = useCallback((text, { useWebSearch = null } = {}) => {
    if (!text.trim() || isStreaming) return

    // Add user message immediately
    const userMsg = { role: 'user', content: text, id: Date.now() }
    setMessages((prev) => [...prev, userMsg])
    setIsStreaming(true)
    setStatusText('')

    // Placeholder for streaming assistant message
    const assistantId = Date.now() + 1
    setMessages((prev) => [
      ...prev,
      { role: 'assistant', content: '', sources: [], id: assistantId, streaming: true },
    ])

    cancelRef.current = streamChat(
      { sessionId: sessionId.current, message: text, useWebSearch },
      {
        onStatus: (msg) => setStatusText(msg),

        onToken: (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + token } : m
            )
          )
        },

        onSources: (sources) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, sources } : m
            )
          )
        },

        onDone: () => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, streaming: false } : m
            )
          )
          setIsStreaming(false)
          setStatusText('')
        },

        onError: (err) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: `Error: ${err}`, streaming: false, isError: true }
                : m
            )
          )
          setIsStreaming(false)
          setStatusText('')
        },
      }
    )
  }, [isStreaming])

  const cancelStream = useCallback(() => {
    cancelRef.current?.cancel()
    setIsStreaming(false)
    setStatusText('')
    // Mark last streaming message as done
    setMessages((prev) =>
      prev.map((m) => (m.streaming ? { ...m, streaming: false } : m))
    )
  }, [])

  const clearHistory = useCallback(() => {
    cancelRef.current?.cancel()
    setMessages([])
    setIsStreaming(false)
    setStatusText('')
    // New session
    const newId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
    sessionStorage.setItem(SESSION_KEY, newId)
    sessionId.current = newId
  }, [])

  return {
    messages,
    isStreaming,
    statusText,
    sendMessage,
    cancelStream,
    clearHistory,
    sessionId: sessionId.current,
  }
}
