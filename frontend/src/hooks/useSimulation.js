import { useRef, useCallback, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { simulateStep, getStatus, selectEngine } from '../api/client'
import { useSimulationContext } from '../context/SimulationContext'
import { buildRunRecord, save as saveRun } from '../history/store'

const CYCLE_INTERVAL_MS = 500   // 2 Hz update rate — visible to the human eye

export function useSimulation() {
  const { state, dispatch, setEngine } = useSimulationContext()
  // setInterval was firing /simulate every CYCLE_INTERVAL_MS regardless of
  // whether the previous request had returned.  On the deployed path one
  // round-trip can exceed the interval, so requests piled up against HF
  // Spaces' limited concurrency, hit the axios timeout, and the browser
  // showed them as pending → cancelled.  This ref drives a self-scheduling
  // loop instead — the next tick only fires after the previous one resolves.
  const runningRef    = useRef(false)
  const sessionIdRef  = useRef(null)
  const sessionStartRef = useRef(null)
  // Tracks the last sessionId we've already saved to history.  Prevents
  // duplicate saves when multiple in-flight /simulate ticks all resolve
  // after convergence and each queue their own persistCurrentRun.
  const persistedSessionRef = useRef(null)

  // Always-fresh ref so the loop closure never goes stale
  const stateRef = useRef(state)
  useEffect(() => { stateRef.current = state }, [state])

  // Persist the just-finished run to the history store.  Idempotent per
  // sessionId — convergence + slow network can queue many calls; only
  // the first one per session writes.
  const persistCurrentRun = useCallback(() => {
    const s = stateRef.current
    if (!s || !s.sessionId || !s.cycleNumber) return
    if (persistedSessionRef.current === s.sessionId) return
    persistedSessionRef.current = s.sessionId
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

  // ── Self-scheduling cycle loop ──────────────────────────────────────────────
  const tick = useCallback(async () => {
    if (!runningRef.current) return
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
      // Stopped while in-flight? Drop the result.
      if (!runningRef.current) return
      dispatch({ type: 'CYCLE_COMPLETE', payload: result })

      if (result.converged && result.cycle_number > 20) {
        runningRef.current = false
        // Wait one tick for the reducer to commit the final cycle, then
        // snapshot the run so the persisted record includes it.
        setTimeout(() => {
          persistCurrentRun()
          dispatch({ type: 'STOP' })
        }, 0)
        return
      }
    } catch (err) {
      if (!runningRef.current) return
      dispatch({ type: 'ERROR', payload: err?.response?.data?.detail ?? err.message })
      if (stateRef.current.errorCount >= 2) {
        runningRef.current = false
        persistCurrentRun()
        dispatch({ type: 'STOP' })
        return
      }
    }

    if (runningRef.current) {
      setTimeout(tick, CYCLE_INTERVAL_MS)
    }
  }, [dispatch, persistCurrentRun])

  // ── Start simulation ─────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    if (stateRef.current.isRunning || runningRef.current) return

    sessionIdRef.current      = uuidv4()
    sessionStartRef.current   = new Date().toISOString()
    persistedSessionRef.current = null    // allow this new session to be saved
    dispatch({ type: 'SET_SESSION_ID', payload: sessionIdRef.current })
    dispatch({ type: 'START' })

    try {
      await selectEngine(stateRef.current.engineId)
    } catch { /* non-fatal */ }

    runningRef.current = true
    tick()
  }, [dispatch, tick])

  // ── Stop simulation ──────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    runningRef.current = false
    persistCurrentRun()
    dispatch({ type: 'STOP' })
  }, [dispatch, persistCurrentRun])

  // ── Clean up on unmount ──────────────────────────────────────────────────────
  useEffect(() => () => {
    runningRef.current = false
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
