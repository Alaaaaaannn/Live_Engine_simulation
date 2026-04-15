import { createContext, useContext, useReducer, useCallback } from 'react'

// ── Initial state ──────────────────────────────────────────────────────────────

const initialState = {
  // Session
  sessionId: null,
  isRunning: false,
  cycleNumber: 0,
  backendReady: false,

  // Engine
  engineId: 'gengine1',

  // Sliders (standardized values)
  sliders: {
    lambda:        0.0,
    rpm:           0.0,
    load:          0.0,
    ignitionAngle: 0.0,
    coBaseline:    0.0,
    hcBaseline:    0.0,
  },

  // Controls
  faultInject:    null,      // 'fault1' | 'fault2' | 'fault3' | null
  autoCorrection: true,

  // Current cycle result
  currentFault: { class: 0, name: 'Normal', confidence: 0 },
  lambdaCurrent:    0.0,
  lambdaPredicted:  0.0,
  coCurrent:        0.0,
  hcCurrent:        0.0,
  noxCurrent:       0.0,
  converged:        false,

  // Chart history (ring buffer: max 200 points)
  lambdaHistory:    [],   // [{cycle, actual, predicted}]
  emissionsHistory: [],   // [{cycle, co, hc, nox}]

  // Digital twin log (most recent first)
  twinLog: [],     // [{cycle, fuelTrim, sparkAdv, lambdaPred, approved}]

  // SHAP
  shapFeatures: null,   // [{feature, importance}] or null

  // Before/After healing comparison
  faultSnapshot: null,   // captured when fault first detected: {lambda, co, hc, nox, faultName, cycle}
  healedSnapshot: null,  // captured when system returns to Normal

  // Environmental impact accumulator (reset each START)
  impactStats: {
    coSavedG:           0,   // grams of CO avoided
    hcSavedMg:          0,   // mg of HC avoided
    noxSavedMg:         0,   // mg of NOx avoided
    fuelSavedMl:        0,   // ml of wasted fuel prevented (rich faults)
    catalystProtectedS: 0,   // seconds of catalyst overtemp exposure prevented
  },

  // Error
  errorMessage: null,
  errorCount: 0,
}

// ── Reducer ────────────────────────────────────────────────────────────────────

const MAX_HISTORY = 200

function reducer(state, action) {
  switch (action.type) {
    case 'SET_BACKEND_READY':
      return { ...state, backendReady: action.payload }

    case 'SET_SESSION_ID':
      return { ...state, sessionId: action.payload }

    case 'SET_SLIDER':
      return { ...state, sliders: { ...state.sliders, [action.key]: action.value } }

    case 'SET_FAULT_INJECT':
      return { ...state, faultInject: action.payload }

    case 'SET_AUTO_CORRECTION':
      return { ...state, autoCorrection: action.payload }

    case 'SET_ENGINE':
      return { ...state, engineId: action.payload, cycleNumber: 0,
               lambdaHistory: [], emissionsHistory: [], twinLog: [],
               shapFeatures: null, currentFault: initialState.currentFault }

    case 'START':
      return { ...state, isRunning: true, errorMessage: null, errorCount: 0,
               faultSnapshot: null, healedSnapshot: null,
               impactStats: initialState.impactStats }

    case 'STOP':
      return { ...state, isRunning: false, faultInject: null }

    case 'CYCLE_COMPLETE': {
      const d = action.payload
      const cycle = d.cycle_number

      // Update history (ring buffer)
      const lambdaEntry = { cycle, actual: d.lambda_current, predicted: d.lambda_predicted }
      const emissEntry  = { cycle, co: d.co_current, hc: d.hc_current, nox: d.nox_current }

      const newLambda = [...state.lambdaHistory, lambdaEntry].slice(-MAX_HISTORY)
      const newEmiss  = [...state.emissionsHistory, emissEntry].slice(-MAX_HISTORY)

      // DT log entry (prepend — newest first)
      const logEntry = {
        cycle,
        fuelTrim:    d.control_action.fuel_trim,
        sparkAdv:    d.control_action.spark_advance,
        lambdaPred:  d.twin.lambda_predicted,
        approved:    d.twin.approved,
        faultName:   d.fault_name,
      }
      const newLog = [logEntry, ...state.twinLog].slice(0, 50)

      const prevClass = state.currentFault.class
      const newClass  = d.fault_class

      // Capture fault snapshot the first time a fault is detected
      const newFaultSnapshot =
        prevClass === 0 && newClass !== 0
          ? { lambda: d.lambda_current, co: d.co_current, hc: d.hc_current,
              nox: d.nox_current, faultName: d.fault_name, cycle }
          : state.faultSnapshot

      // Capture healed snapshot once lambda is within ±0.15σ of stoichiometric
      const fullyHealed = Math.abs(d.lambda_current) < 0.15
      const newHealedSnapshot =
        state.healedSnapshot === null && fullyHealed && state.faultSnapshot !== null && newClass === 0
          ? { lambda: d.lambda_current, co: d.co_current, hc: d.hc_current,
              nox: d.nox_current, cycle }
          : state.healedSnapshot

      // Accumulate environmental impact while fault is active and being corrected
      const shouldAccumulate = d.fault_class !== 0 &&
                               state.autoCorrection &&
                               state.healedSnapshot === null
      const newImpact = shouldAccumulate ? {
        coSavedG:           state.impactStats.coSavedG    + Math.max(0, d.co_current)  * 0.12,
        hcSavedMg:          state.impactStats.hcSavedMg   + Math.max(0, d.hc_current)  * 60,
        noxSavedMg:         state.impactStats.noxSavedMg  + Math.max(0, d.nox_current) * 80,
        fuelSavedMl:        state.impactStats.fuelSavedMl + Math.max(0, -d.lambda_current) * 0.25,
        catalystProtectedS: state.impactStats.catalystProtectedS + 0.5,
      } : state.impactStats

      return {
        ...state,
        cycleNumber: cycle,
        currentFault: {
          class:      d.fault_class,
          name:       d.fault_name,
          confidence: d.fault_confidence,
        },
        lambdaCurrent:    d.lambda_current,
        lambdaPredicted:  d.lambda_predicted,
        coCurrent:        d.co_current,
        hcCurrent:        d.hc_current,
        noxCurrent:       d.nox_current,
        converged:        d.converged,
        lambdaHistory:    newLambda,
        emissionsHistory: newEmiss,
        twinLog:          newLog,
        shapFeatures:     d.shap_features ?? state.shapFeatures,
        faultInject:      null,
        faultSnapshot:    newFaultSnapshot,
        healedSnapshot:   newHealedSnapshot,
        impactStats:      newImpact,
        errorCount:       0,
      }
    }

    case 'ERROR':
      return {
        ...state,
        errorMessage: action.payload,
        errorCount: state.errorCount + 1,
        isRunning: state.errorCount + 1 < 3 ? state.isRunning : false,
      }

    case 'CLEAR_ERROR':
      return { ...state, errorMessage: null }

    default:
      return state
  }
}

// ── Context ────────────────────────────────────────────────────────────────────

const SimulationContext = createContext(null)

export function SimulationProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)

  const setSlider = useCallback((key, value) =>
    dispatch({ type: 'SET_SLIDER', key, value }), [])

  const setFaultInject = useCallback((v) =>
    dispatch({ type: 'SET_FAULT_INJECT', payload: v }), [])

  const setAutoCorrection = useCallback((v) =>
    dispatch({ type: 'SET_AUTO_CORRECTION', payload: v }), [])

  const setEngine = useCallback((v) =>
    dispatch({ type: 'SET_ENGINE', payload: v }), [])

  return (
    <SimulationContext.Provider value={{ state, dispatch, setSlider, setFaultInject, setAutoCorrection, setEngine }}>
      {children}
    </SimulationContext.Provider>
  )
}

export function useSimulationContext() {
  const ctx = useContext(SimulationContext)
  if (!ctx) throw new Error('useSimulationContext must be inside SimulationProvider')
  return ctx
}
