import { useRef, useCallback, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { simulateStep, getStatus, selectEngine } from '../api/client'
import { useSimulationContext } from '../context/SimulationContext'

const CYCLE_INTERVAL_MS = 500   // 2 Hz update rate — visible to the human eye

export function useSimulation() {
  const { state, dispatch, setEngine } = useSimulationContext()
  const intervalRef   = useRef(null)
  const sessionIdRef  = useRef(null)

  // Always-fresh ref so the interval closure never goes stale
  const stateRef = useRef(state)
  useEffect(() => { stateRef.current = state }, [state])

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

    sessionIdRef.current = uuidv4()
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
          dispatch({ type: 'STOP' })
        }
      } catch (err) {
        dispatch({ type: 'ERROR', payload: err?.response?.data?.detail ?? err.message })
        if (stateRef.current.errorCount >= 2) {
          clearInterval(intervalRef.current)
          dispatch({ type: 'STOP' })
        }
      }
    }, CYCLE_INTERVAL_MS)
  }, [dispatch])

  // ── Stop simulation ──────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    dispatch({ type: 'STOP' })
  }, [dispatch])

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
