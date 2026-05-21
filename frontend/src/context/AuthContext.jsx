import { createContext, useContext, useEffect, useState, useCallback } from 'react'
import * as authApi from '../api/auth'
import {
  getStoredToken, getStoredUser, setStoredAuth, clearStoredAuth,
} from '../auth/tokenStorage'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [token, setToken] = useState(getStoredToken)
  const [user,  setUser]  = useState(getStoredUser)
  const [bootstrapping, setBootstrapping] = useState(Boolean(token))

  // Validate token on mount — if /auth/me 401s the token is stale.
  useEffect(() => {
    if (!token) { setBootstrapping(false); return }
    authApi.me()
      .then(u => {
        setUser(u)
        setStoredAuth(token, u)
      })
      .catch(() => {
        clearStoredAuth()
        setToken(null); setUser(null)
      })
      .finally(() => setBootstrapping(false))
  }, [])  // run once

  const login = useCallback(async (email, password) => {
    const res = await authApi.login(email, password)
    const u = { id: res.user_id, email: res.email }
    setStoredAuth(res.access_token, u)
    setToken(res.access_token)
    setUser(u)
  }, [])

  const register = useCallback(async (email, password) => {
    const res = await authApi.register(email, password)
    const u = { id: res.user_id, email: res.email }
    setStoredAuth(res.access_token, u)
    setToken(res.access_token)
    setUser(u)
  }, [])

  const logout = useCallback(() => {
    clearStoredAuth()
    setToken(null)
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ token, user, bootstrapping, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
