import axios from 'axios'
import { getStoredToken, clearStoredAuth } from '../auth/tokenStorage'

// In dev (vite proxy off) hit FastAPI directly.  In prod, hit the nginx
// /api/ route on the same origin.
const BASE_URL = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: BASE_URL,
  // /simulate against HF Spaces free tier (cold start + per-cycle compute +
  // Supabase write) can take several seconds.  5 s was triggering cancellations.
  timeout: 15_000,
  headers: { 'Content-Type': 'application/json' },
})

// Attach the bearer token (if present) to every request.
api.interceptors.request.use(cfg => {
  const tok = getStoredToken()
  if (tok) cfg.headers.Authorization = `Bearer ${tok}`
  return cfg
})

// On 401, drop the stale token so the app re-renders to the login screen.
api.interceptors.response.use(
  r => r,
  err => {
    if (err?.response?.status === 401) {
      clearStoredAuth()
      // Trigger a re-render so AuthProvider notices.  A storage event
      // would also work, but a manual location reload is simplest.
      if (!window.location.pathname.includes('login')) {
        window.dispatchEvent(new Event('dt:auth-expired'))
      }
    }
    return Promise.reject(err)
  },
)

// ── Endpoint wrappers ──────────────────────────────────────────────────────────

export const simulateStep = (payload) =>
  api.post('/simulate', payload).then(r => r.data)

export const classifyWindow = (sensorWindow) =>
  api.post('/classify', { sensor_window: sensorWindow }).then(r => r.data)

export const selectEngine = (engineId) =>
  api.post('/engine/select', { engine_id: engineId }).then(r => r.data)

export const injectFault = (sessionId, faultType) =>
  api.post('/fault/inject', { session_id: sessionId, fault_type: faultType }).then(r => r.data)

export const getStatus = () =>
  api.get('/status').then(r => r.data)

export const getRuntimeConfig = () =>
  api.get('/config/runtime').then(r => r.data)

export const postRuntimeConfig = (patch) =>
  api.post('/config/runtime', patch).then(r => r.data)

export default api
