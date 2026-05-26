export default function StreamingIndicator({ statusText }) {
  return (
    <div className="flex items-center gap-2 px-4 py-2 text-sm text-gray-400">
      <span className="flex gap-1.5">
        <span className="w-1.5 h-1.5 bg-[#f03e3e] rounded-full animate-bounce [animation-delay:0ms]" />
        <span className="w-1.5 h-1.5 bg-[#f03e3e] rounded-full animate-bounce [animation-delay:150ms]" />
        <span className="w-1.5 h-1.5 bg-[#f03e3e] rounded-full animate-bounce [animation-delay:300ms]" />
      </span>
      {statusText && <span className="text-gray-400/80 italic font-light tracking-wide">{statusText}</span>}
    </div>
  )
}
