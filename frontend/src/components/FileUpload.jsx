import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, X, CheckCircle, AlertCircle, FileText } from 'lucide-react'
import { uploadDocument } from '../services/api'

const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'text/plain': ['.txt'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/markdown': ['.md'],
}

function UploadItem({ file, status, progress, message }) {
  return (
    <div className="flex items-center gap-3 p-3 bg-gray-700 rounded-lg">
      <FileText size={16} className="text-blue-400 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-200 truncate">{file.name}</p>
        {status === 'uploading' && (
          <div className="mt-1 bg-gray-600 rounded-full h-1">
            <div
              className="bg-blue-500 h-1 rounded-full transition-all duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
        {status === 'done' && (
          <p className="text-xs text-green-400 mt-0.5">{message}</p>
        )}
        {status === 'error' && (
          <p className="text-xs text-red-400 mt-0.5">{message}</p>
        )}
      </div>
      {status === 'done' && <CheckCircle size={16} className="text-green-400 flex-shrink-0" />}
      {status === 'error' && <AlertCircle size={16} className="text-red-400 flex-shrink-0" />}
    </div>
  )
}

export default function FileUpload({ onClose }) {
  const [uploads, setUploads] = useState([])

  const updateUpload = (id, patch) =>
    setUploads((prev) => prev.map((u) => (u.id === id ? { ...u, ...patch } : u)))

  const onDrop = useCallback(async (acceptedFiles) => {
    for (const file of acceptedFiles) {
      const id = `${Date.now()}_${file.name}`
      setUploads((prev) => [...prev, { id, file, status: 'uploading', progress: 0 }])

      try {
        const result = await uploadDocument(file, (pct) =>
          updateUpload(id, { progress: pct })
        )
        updateUpload(id, {
          status: 'done',
          message: `${result.chunks_indexed} chunks indexed`,
        })
      } catch (err) {
        const msg = err.response?.data?.detail || err.message || 'Upload failed'
        updateUpload(id, { status: 'error', message: msg })
      }
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: 50 * 1024 * 1024, // 50 MB
  })

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg bg-gray-800 rounded-2xl border border-gray-700 shadow-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-100">Upload Documents</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-gray-600 hover:border-gray-500 hover:bg-gray-700/30'
          }`}
        >
          <input {...getInputProps()} />
          <Upload size={32} className="mx-auto mb-3 text-gray-400" />
          {isDragActive ? (
            <p className="text-blue-400 font-medium">Drop files here</p>
          ) : (
            <>
              <p className="text-gray-300 font-medium mb-1">
                Drag & drop or click to upload
              </p>
              <p className="text-xs text-gray-500">PDF, TXT, DOCX, MD — max 50 MB each</p>
            </>
          )}
        </div>

        {uploads.length > 0 && (
          <div className="mt-4 flex flex-col gap-2 max-h-48 overflow-y-auto">
            {uploads.map((u) => (
              <UploadItem key={u.id} {...u} />
            ))}
          </div>
        )}

        <button
          onClick={onClose}
          className="mt-4 w-full py-2 text-sm text-gray-400 hover:text-gray-200 transition-colors"
        >
          Done
        </button>
      </div>
    </div>
  )
}
