import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { GLOSSARY } from './glossary'

/** A glossary word: hover, focus or tap it for a one-line definition. */
export function Term({ k, children }: { k: string; children?: ReactNode }) {
  const entry = GLOSSARY[k]
  const id = useId()
  const [open, setOpen] = useState(false)
  const tip = useRef<HTMLSpanElement>(null)

  useLayoutEffect(() => {
    if (!open) return
    const fit = () => {
      const el = tip.current
      if (!el) return
      el.style.transform = ''
      const r = el.getBoundingClientRect()
      const over = r.right - (document.documentElement.clientWidth - 8)
      if (over > 0) el.style.transform = `translateX(${-Math.max(0, Math.min(over, r.left - 8))}px)`
    }
    fit()
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [open])

  if (!entry) return <>{children}</>
  return (
    <span className="term" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
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
        <span role="tooltip" id={id} className="term-tip" ref={tip}>
          <strong>{entry.term}.</strong> {entry.text}
        </span>
      )}
    </span>
  )
}
