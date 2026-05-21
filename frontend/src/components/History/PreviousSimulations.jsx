import { useEffect, useState, useCallback } from 'react'
import { list, remove, rename, subscribe } from '../../history/store'
import { useHashRoute } from '../../hooks/useHashRoute'
import HistoryDetail from './HistoryDetail'
import './History.css'

function formatTs(ts) {
  try {
    return new Date(ts).toLocaleString(undefined, {
      year: '2-digit', month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    })
  } catch { return '' }
}

function faultBadgeClass(fc) {
  if (fc == null || fc === 0) return 'badge-normal'
  if (fc === 3)               return 'badge-critical'
  return 'badge-fault'
}

function useRuns() {
  const [runs, setRuns] = useState(() => list())
  useEffect(() => {
    const refresh = () => setRuns(list())
    refresh()
    const off = subscribe(refresh)
    const onStorage = (e) => { if (e.key && e.key.startsWith('dt:')) refresh() }
    window.addEventListener('storage', onStorage)
    return () => { off(); window.removeEventListener('storage', onStorage) }
  }, [])
  return runs
}

function RenameDialog({ initial, onCancel, onSave }) {
  const [value, setValue] = useState(initial)
  return (
    <div className="hist-dialog-backdrop" onClick={onCancel}>
      <div className="hist-dialog" onClick={e => e.stopPropagation()}>
        <div className="panel-title">Rename simulation</div>
        <input
          autoFocus
          className="hist-rename-input"
          value={value}
          maxLength={120}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter') onSave(value.trim())
            if (e.key === 'Escape') onCancel()
          }}
        />
        <div className="hist-dialog-row">
          <button className="hist-btn-secondary" onClick={onCancel}>Cancel</button>
          <button
            className="hist-btn-primary"
            disabled={!value.trim()}
            onClick={() => onSave(value.trim())}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  )
}

function ConfirmDelete({ name, onCancel, onConfirm }) {
  return (
    <div className="hist-dialog-backdrop" onClick={onCancel}>
      <div className="hist-dialog" onClick={e => e.stopPropagation()}>
        <div className="panel-title" style={{ color: 'var(--accent-red)' }}>Delete simulation</div>
        <p className="hist-dialog-text">
          Delete <b style={{ color: 'var(--text-primary)' }}>{name}</b>?
          This cannot be undone.
        </p>
        <div className="hist-dialog-row">
          <button className="hist-btn-secondary" onClick={onCancel}>Cancel</button>
          <button className="hist-btn-danger" onClick={onConfirm}>Delete</button>
        </div>
      </div>
    </div>
  )
}

function RunCard({ run, onOpen, onRename, onDelete }) {
  const finalFc = run.finalFault?.class
  const finalFn = run.finalFault?.name || 'Normal'
  return (
    <article className="hist-card panel">
      <header className="hist-card-header">
        <button className="hist-card-title" onClick={onOpen} title="Open details">
          {run.name}
        </button>
        <div className="hist-card-actions">
          <button className="hist-icon-btn" onClick={onRename}>Rename</button>
          <button className="hist-icon-btn danger" onClick={onDelete}>Delete</button>
        </div>
      </header>

      <div className="hist-card-meta">
        <span>{formatTs(run.createdAt)}</span>
        <span className="hist-sep">·</span>
        <span className="mono">{run.engineId}</span>
        <span className="hist-sep">·</span>
        <span>{run.cycleCount} cycles</span>
      </div>

      <div className="hist-card-row">
        <span className={`badge ${faultBadgeClass(finalFc)}`}>
          Final: {finalFn}
        </span>
        {run.autoCorrection && <span className="badge badge-normal">Auto on</span>}
        {!run.autoCorrection && <span className="badge badge-fault">Auto off</span>}
        {run.finalObservations?.converged && <span className="badge badge-approved">Converged</span>}
      </div>

      <div className="hist-card-numbers">
        <div><span className="hist-k">λ</span><span className="mono">{run.finalObservations?.lambdaCurrent?.toFixed(3) ?? '—'}</span></div>
        <div><span className="hist-k">CO</span><span className="mono">{run.finalObservations?.coCurrent?.toFixed(3) ?? '—'}</span></div>
        <div><span className="hist-k">HC</span><span className="mono">{run.finalObservations?.hcCurrent?.toFixed(3) ?? '—'}</span></div>
        <div><span className="hist-k">NOx</span><span className="mono">{run.finalObservations?.noxCurrent?.toFixed(3) ?? '—'}</span></div>
      </div>

      <button className="hist-card-open" onClick={onOpen}>View details</button>
    </article>
  )
}

export default function PreviousSimulations() {
  const { path, navigate } = useHashRoute()
  const runs = useRuns()
  const [renaming, setRenaming] = useState(null)   // { id, name } | null
  const [deleting, setDeleting] = useState(null)   // { id, name } | null

  const detailMatch = path.match(/^\/history\/(.+)$/)
  const detailId    = detailMatch ? decodeURIComponent(detailMatch[1]) : null

  const openRun  = useCallback(id => navigate(`/history/${encodeURIComponent(id)}`), [navigate])
  const backList = useCallback(()  => navigate('/history'), [navigate])

  if (detailId) {
    return (
      <HistoryDetail
        id={detailId}
        onBack={backList}
        onRename={(id, currentName) => setRenaming({ id, name: currentName })}
        onDelete={(id, currentName) => setDeleting({ id, name: currentName })}
      />
    )
  }

  return (
    <div className="hist-page">
      <header className="hist-page-header">
        <h2 className="hist-page-title">Previous simulations</h2>
        <span className="hist-page-meta">{runs.length} saved</span>
      </header>

      {runs.length === 0 ? (
        <div className="hist-empty panel">
          <svg className="hist-empty-icon" width="32" height="32" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 3h18v18H3z" /><path d="M3 9h18" /><path d="M9 21V9" />
          </svg>
          <p>No saved simulations yet.</p>
          <p className="text-muted">
            Press <b style={{ color: 'var(--text-primary)' }}>Start simulation</b> on the Dashboard.
            Runs are recorded automatically when they stop or converge.
          </p>
        </div>
      ) : (
        <div className="hist-grid">
          {runs.map(run => (
            <RunCard
              key={run.id}
              run={run}
              onOpen={()   => openRun(run.id)}
              onRename={() => setRenaming({ id: run.id, name: run.name })}
              onDelete={() => setDeleting({ id: run.id, name: run.name })}
            />
          ))}
        </div>
      )}

      {renaming && (
        <RenameDialog
          initial={renaming.name}
          onCancel={() => setRenaming(null)}
          onSave={(name) => { rename(renaming.id, name || renaming.name); setRenaming(null) }}
        />
      )}
      {deleting && (
        <ConfirmDelete
          name={deleting.name}
          onCancel={() => setDeleting(null)}
          onConfirm={() => {
            const id = deleting.id
            remove(id)
            setDeleting(null)
            if (detailId === id) backList()
          }}
        />
      )}
    </div>
  )
}
