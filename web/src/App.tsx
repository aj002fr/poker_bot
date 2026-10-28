import { useEffect, useState } from 'react'
import { ApiFailure, api, loadCredentials, saveCredentials, useTableStream, type Credentials } from './api'
import { Lobby } from './screens/Lobby'
import { Results } from './screens/Results'
import { Setup, formatFailure } from './screens/Setup'
import { TableScreen } from './screens/TableScreen'

function useHashRoute(): [string, (path: string) => void] {
  const read = () => location.hash.replace(/^#/, '') || '/'
  const [route, setRoute] = useState(read)
  useEffect(() => {
    const onChange = () => setRoute(read())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return [route, (path: string) => (location.hash = path)]
}

function Header({ onHome }: { onHome: () => void }) {
  return (
    <header className="site-header">
      <button className="brand" onClick={onHome}>
        <span className="brand-mark" aria-hidden="true" />
        <span className="brand-name">Practice Arena</span>
      </button>
      <span className="practice-label" data-testid="practice-label">
        Practice assumptions — official kit not yet validated
      </span>
    </header>
  )
}

function JoinSeat({ tableId, seat, navigate }: { tableId: string; seat: number; navigate: (p: string) => void }) {
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    const existing = loadCredentials(tableId)
    if (existing.seatToken) {
      navigate(`/t/${tableId}`)
      return
    }
    api.join(tableId, seat).then(
      (joined) => {
        saveCredentials(tableId, { ...existing, seat: joined.seat, seatToken: joined.seat_token })
        navigate(`/t/${tableId}`)
      },
      (err) => setError(err instanceof ApiFailure ? formatFailure(err) : 'Could not reach the server.'),
    )
  }, [tableId, seat])
  return (
    <main className="panel narrow">
      {error ? (
        <>
          <p className="error">{error}</p>
          <a className="btn" href={`#/t/${tableId}`}>
            Watch the table instead
          </a>
        </>
      ) : (
        <p>Claiming seat {seat}…</p>
      )}
    </main>
  )
}

function errorTitle(error: ApiFailure): string {
  if (error.status === 404) return 'Table not found'
  if (error.code === 'table_halted') return 'Table halted'
  if (error.code === 'storage_unavailable') return 'Table paused'
  return 'Access denied'
}

function errorText(error: ApiFailure): string {
  if (error.status === 404) return 'No table with this link exists here. In-memory mode does not keep tables across a server restart.'
  if (error.code === 'table_halted')
    return 'Its stored record could not be reproduced exactly on recovery, so the table was stopped for diagnosis instead of being repaired.'
  if (error.code === 'storage_unavailable')
    return 'Storage is unavailable. Think time is paused, and the table resumes here automatically when storage is back.'
  return `${error.code}: ${error.message}`
}

function TableRoute({ tableId, navigate }: { tableId: string; navigate: (p: string) => void }) {
  const [creds, setCreds] = useState<Credentials>(() => loadCredentials(tableId))
  const token = creds.seatToken ?? null
  const { snapshot, status, error, accept } = useTableStream(tableId, token)

  if (error && !snapshot) {
    return (
      <main className="panel narrow">
        <h1 className="screen-title">{errorTitle(error)}</h1>
        <p className="lede">{errorText(error)}</p>
        <button className="btn btn-primary" onClick={() => navigate('/')}>
          Set up a new table
        </button>
      </main>
    )
  }
  if (!snapshot) {
    return (
      <main className="panel narrow" aria-busy="true">
        <p className="muted">Connecting to table…</p>
      </main>
    )
  }
  if (snapshot.status === 'waiting') return <Lobby snapshot={snapshot} creds={creds} onCreds={setCreds} />
  if (snapshot.status === 'complete') return <Results snapshot={snapshot} creds={creds} navigate={navigate} />
  return <TableScreen snapshot={snapshot} token={token} stream={status} onSnapshot={accept} />
}

export default function App() {
  const [route, navigate] = useHashRoute()
  const join = route.match(/^\/t\/([\w-]+)\/join\/(\d+)$/)
  const table = route.match(/^\/t\/([\w-]+)$/)
  return (
    <div className="app">
      <Header onHome={() => navigate('/')} />
      {join ? (
        <JoinSeat key={route} tableId={join[1]} seat={Number(join[2])} navigate={navigate} />
      ) : table ? (
        <TableRoute key={table[1]} tableId={table[1]} navigate={navigate} />
      ) : (
        <Setup navigate={navigate} />
      )}
    </div>
  )
}
