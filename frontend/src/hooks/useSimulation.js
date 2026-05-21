import { useRef, useCallback, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { simulateStep, getStatus, selectEngine } from '../api/client'
import { useSimulationContext } from '../context/SimulationContext'
import { buildRunRecord, save as saveRun } from '../history/store'

const CYCLE_INTERVAL_MS = 500   // 2 Hz update rate — visible to the human eye

export function useSimulation() {
  const { state, dispatch, setEngine } = useSimulationContext()
  const intervalRef   = useRef(null)
  const sessionIdRef  = useRef(null)
  const sessionStartRef = useRef(null)

  // Always-fresh ref so the interval closure never goes stale
  const stateRef = useRef(state)
  useEffect(() => { stateRef.current = state }, [state])

  // Persist the just-finished run to the history store.  No-op when the
  // session never ran a cycle (e.g. start->stop with no completed step).
  const persistCurrentRun = useCallback(() => {
    const s = stateRef.current
    if (!s || !s.sessionId || !s.cycleNumber) return
    try {
      const record = buildRunRecord(s, { sessionStartIso: sessionStartRef.current })
      saveRun(record)
    } catch (e) {
      // History persistence must never break the live simulator.
      console.warn('[history] failed to save run', e)
    }
  }, [])

  // ── Check backend health on mount ───────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const status = await getStatus()
        dispatch({ type: 'SET_BACKEND_READY', payload: status.models_loaded })
      } catch {
        dispatch({ type: 'SET_BACKEND_READY', payload: false })
      }
    }
    check()
    const healthInterval = setInterval(check, 10_000)
    return () => clearInterval(healthInterval)
  }, [dispatch])

  // ── Start simulation ─────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    if (stateRef.current.isRunning) return

    sessionIdRef.current   = uuidv4()
    sessionStartRef.current = new Date().toISOString()
    dispatch({ type: 'SET_SESSION_ID', payload: sessionIdRef.current })
    dispatch({ type: 'START' })

    try {
      await selectEngine(stateRef.current.engineId)
    } catch { /* non-fatal */ }

    intervalRef.current = setInterval(async () => {
      // Read from ref so we always get the latest React state — avoids
      // the stale-closure bug where fault_inject / lambda_val would never
      // update after the first cycle.
      const s = stateRef.current

      try {
        const payload = {
          session_id:      sessionIdRef.current,
          engine_id:       s.engineId,
          // When auto-correction is on don't override lambda — let the
          // backend healing logic drive the state window toward stoichiometric.
          lambda_val:      s.autoCorrection ? 0.0 : s.sliders.lambda,
          rpm:             s.sliders.rpm,
          load:            s.sliders.load,
          ignition_angle:  s.sliders.ignitionAngle,
          co_baseline:     s.sliders.coBaseline,
          hc_baseline:     s.sliders.hcBaseline,
          fault_inject:    s.faultInject,   // null after first cycle (reducer clears it)
          auto_correction: s.autoCorrection,
          cycle_number:    s.cycleNumber,
        }

        const result = await simulateStep(payload)
        dispatch({ type: 'CYCLE_COMPLETE', payload: result })

        if (result.converged && result.cycle_number > 20) {
          clearInterval(intervalRef.current)
          // Wait one tick for the reducer to commit the final cycle, then
          // snapshot the run so the persisted record includes it.
          setTimeout(() => {
            persistCurrentRun()
            dispatch({ type: 'STOP' })
          }, 0)
        }
      } catch (err) {
        dispatch({ type: 'ERROR', payload: err?.response?.data?.detail ?? err.message })
        if (stateRef.current.errorCount >= 2) {
          clearInterval(intervalRef.current)
          persistCurrentRun()
          dispatch({ type: 'STOP' })
        }
      }
    }, CYCLE_INTERVAL_MS)
  }, [dispatch, persistCurrentRun])

  // ── Stop simulation ──────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    persistCurrentRun()
    dispatch({ type: 'STOP' })
  }, [dispatch, persistCurrentRun])

  // ── Clean up on unmount ──────────────────────────────────────────────────────
  useEffect(() => () => {
    if (intervalRef.current) clearInterval(intervalRef.current)
  }, [])

  // ── Switch engine ────────────────────────────────────────────────────────────
  const switchEngine = useCallback(async (engineId) => {
    stop()
    setEngine(engineId)
    try {
      await selectEngine(engineId)
    } catch { /* non-fatal */ }
  }, [stop, setEngine])

  return { start, stop, switchEngine }
}
