import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:9004'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 5000,
  headers: { 'Content-Type': 'application/json' },
})

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

export default api
