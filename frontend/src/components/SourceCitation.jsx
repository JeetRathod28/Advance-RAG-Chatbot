import { useState } from 'react'
import { ChevronDown, ChevronUp, ExternalLink, FileText, Globe } from 'lucide-react'

function SourceCard({ source, index }) {
  const isWeb = source.source_type === 'web'
  const Icon = isWeb ? Globe : FileText

  return (
    <div className="source-card">
      <div className="flex-shrink-0 flex items-center justify-center w-6 h-6 bg-[#f03e3e] rounded-full text-xs font-bold text-white shadow-sm">
        {index}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5">
          <Icon size={13} className={isWeb ? 'text-emerald-400' : 'text-rose-400'} />
          <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full border ${
            isWeb 
              ? 'bg-emerald-950/40 text-emerald-400 border-emerald-500/20' 
              : 'bg-rose-950/40 text-rose-300 border-rose-500/20'
          }`}>
            {isWeb ? 'Web' : 'Document'}
          </span>
          {source.url && (
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="ml-auto text-gray-500 hover:text-rose-400 transition-colors"
            >
              <ExternalLink size={13} />
            </a>
          )}
        </div>
        <p className="text-xs font-semibold text-gray-200 truncate">
          {source.title || source.url || 'Unknown source'}
        </p>
        {source.snippet && (
          <p className="text-[11px] text-gray-400 mt-1 line-clamp-2 leading-relaxed">
            {source.snippet}
          </p>
        )}
        {source.score > 0 && (
          <div className="flex items-center gap-2 mt-2">
            <div className="flex-1 bg-gray-800 rounded-full h-1">
              <div
                className="bg-[#f03e3e] h-1 rounded-full shadow-sm"
                style={{ width: `${Math.min(source.score * 100, 100)}%` }}
              />
            </div>
            <span className="text-[10px] text-gray-500 font-medium">
              {(source.score * 100).toFixed(0)}% Match
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

export default function SourceCitation({ sources }) {
  const [expanded, setExpanded] = useState(false)

  if (!sources || sources.length === 0) return null

  const visible = expanded ? sources : sources.slice(0, 3)

  return (
    <div className="mt-3 border-t border-gray-800/40 pt-3">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-rose-400 transition-colors mb-2 font-medium"
      >
        {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        <span>{sources.length} Source{sources.length !== 1 ? 's' : ''} Referenced</span>
      </button>

      <div className="flex flex-col gap-2">
        {visible.map((src, i) => (
          <SourceCard key={i} source={src} index={i + 1} />
        ))}
      </div>

      {!expanded && sources.length > 3 && (
        <button
          onClick={() => setExpanded(true)}
          className="text-xs text-rose-400 hover:text-rose-300 font-medium mt-2 transition-colors block"
        >
          +{sources.length - 3} More Sources
        </button>
      )}
    </div>
  )
}
