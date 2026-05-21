// Persistent storage for completed simulation runs.
// Keeps a capped list (MAX_RUNS) of detailed run records in localStorage so
// the Previous Simulations page can list, rename, and delete them.
//
// IMPORTANT: the storage key is namespaced by the currently-logged-in user's
// id so that switching accounts on the same browser does not leak runs
// across users.  `dt:simulation_history:v1:<user_id>` for authenticated
// users, `dt:simulation_history:v1:anon` otherwise.

import { getStoredUser } from '../auth/tokenStorage'

const STORAGE_PREFIX = 'dt:simulation_history:v1'
const MAX_RUNS       = 100

function storageKey() {
  const u = getStoredUser()
  return `${STORAGE_PREFIX}:${u?.id || 'anon'}`
}

function readAll() {
  try {
    const raw = localStorage.getItem(storageKey())
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function writeAll(runs) {
  try {
    localStorage.setItem(storageKey(), JSON.stringify(runs))
    notify()
  } catch (e) {
    if (runs.length > 1) {
      writeAll(runs.slice(0, Math.max(1, Math.floor(runs.length / 2))))
    }
  }
}

const listeners = new Set()
function notify() { for (const fn of listeners) fn() }

export function subscribe(fn) {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export function list() {
  return readAll().sort((a, b) => b.createdAt - a.createdAt)
}

export function get(id) {
  return readAll().find(r => r.id === id) || null
}

export function save(run) {
  const all = readAll()
  const idx = all.findIndex(r => r.id === run.id)
  if (idx >= 0) all[idx] = run
  else all.unshift(run)
  if (all.length > MAX_RUNS) all.length = MAX_RUNS
  writeAll(all)
}

export function rename(id, name) {
  const all = readAll()
  const idx = all.findIndex(r => r.id === id)
  if (idx < 0) return false
  all[idx] = { ...all[idx], name }
  writeAll(all)
  return true
}

export function remove(id) {
  const all = readAll().filter(r => r.id !== id)
  writeAll(all)
}

export function clear() {
  writeAll([])
}

function nextRunOrdinal() {
  const all = readAll()
  const max = all.reduce((m, r) => Math.max(m, r.ordinal || 0), 0)
  return max + 1
}

export function buildRunRecord(state, { sessionStartIso, autoName } = {}) {
  const ordinal   = nextRunOrdinal()
  const createdAt = Date.now()
  const faultTag  = state.faultSnapshot?.faultName
    ? ` · ${state.faultSnapshot.faultName}`
    : ''
  const name = autoName
    ?? `Run #${ordinal} · ${state.engineId}${faultTag}`

  return {
    id:        cryptoId(),
    ordinal,
    name,
    createdAt,
    startedAt: sessionStartIso || new Date(createdAt).toISOString(),
    sessionId: state.sessionId,
    engineId:  state.engineId,
    cycleCount: state.cycleNumber,
    sliders:        { ...state.sliders },
    autoCorrection: state.autoCorrection,
    finalFault: {
      class:      state.currentFault.class,
      name:       state.currentFault.name,
      confidence: state.currentFault.confidence,
      raw:        state.rawFaultClass,
    },
    finalObservations: {
      lambdaCurrent:    state.lambdaCurrent,
      lambdaPredicted:  state.lambdaPredicted,
      coCurrent:        state.coCurrent,
      hcCurrent:        state.hcCurrent,
      noxCurrent:       state.noxCurrent,
      converged:        state.converged,
      lastFuelTrim:     state.lastFuelTrim,
      lastSparkAdv:     state.lastSparkAdv,
      stabilityLabel:     state.stabilityLabel,
      stabilityAgreement: state.stabilityAgreement,
    },
    lambdaHistory:    state.lambdaHistory,
    emissionsHistory: state.emissionsHistory,
    twinLog:          state.twinLog,
    shapFeatures:     state.shapFeatures,
    faultSnapshot:    state.faultSnapshot,
    healedSnapshot:   state.healedSnapshot,
    impactStats:      state.impactStats,
    runtimeConfig:    state.runtimeConfig,
  }
}

function cryptoId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return 'run-' + Math.random().toString(36).slice(2) + Date.now().toString(36)
}
