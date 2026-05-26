import axios from 'axios'

const BASE_URL = '/api'

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
})

/**
 * Upload a document file. Returns UploadResponse.
 */
export async function uploadDocument(file, onProgress) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })
  return response.data
}

/**
 * Health check. Returns HealthResponse.
 */
export async function getHealth() {
  const response = await api.get('/health')
  return response.data
}

/**
 * List indexed sources.
 */
export async function getSources() {
  const response = await api.get('/sources')
  return response.data
}

/**
 * Clear all indexed documents from the vector store.
 */
export async function clearDocuments() {
  const response = await api.delete('/clear-documents')
  return response.data
}

/**
 * Open an SSE stream for a chat message.
 * Returns an object with a cancel() method.
 *
 * callbacks:
 *   onStatus(text)    — status messages ("Retrieved N passages")
 *   onToken(text)     — incremental tokens
 *   onSources(list)   — final sources array
 *   onDone()          — stream complete
 *   onError(msg)      — error
 */
export function streamChat({ sessionId, message, useWebSearch }, callbacks) {
  const url = new URL(`${window.location.origin}${BASE_URL}/chat/stream`)

  // We use fetch + ReadableStream instead of EventSource because
  // EventSource only supports GET; we need POST with a JSON body.
  const controller = new AbortController()

  ;(async () => {
    try {
      const res = await fetch(url.toString().replace('/chat/stream', '/chat/stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message,
          use_web_search: useWebSearch ?? null,
        }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
        callbacks.onError?.(err.detail || 'Request failed')
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const payload = JSON.parse(line.slice(6))
              switch (payload.type) {
                case 'status':
                  callbacks.onStatus?.(payload.content)
                  break
                case 'token':
                  callbacks.onToken?.(payload.content)
                  break
                case 'sources':
                  callbacks.onSources?.(payload.content)
                  break
                case 'done':
                  callbacks.onDone?.()
                  break
                case 'error':
                  callbacks.onError?.(payload.content)
                  break
              }
            } catch {
              // malformed JSON — skip
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        callbacks.onError?.(err.message || 'Stream error')
      }
    }
  })()

  return { cancel: () => controller.abort() }
}
