import { useId, useState, type ReactNode } from 'react'
import { GLOSSARY } from './glossary'

/** A glossary word: hover, focus or tap it for a one-line definition. */
export function Term({ k, children }: { k: string; children?: ReactNode }) {
  const entry = GLOSSARY[k]
  const id = useId()
  const [open, setOpen] = useState(false)
  if (!entry) return <>{children}</>
  return (
    <span
      className="term"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        className="term-word"
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}
        data-term={k}
      >
        {children ?? entry.term}
      </button>
      {open && (
        <span role="tooltip" id={id} className="term-tip">
          <strong>{entry.term}.</strong> {entry.text}
        </span>
      )}
    </span>
  )
}
