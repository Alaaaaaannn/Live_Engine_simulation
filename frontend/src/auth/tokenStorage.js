// Plain localStorage wrapper for the auth token / cached user.
// Lives outside the React context so non-React modules (e.g. the axios
// client) can read it without pulling in React.

export const TOKEN_KEY = 'dt:auth_token:v1'
export const USER_KEY  = 'dt:auth_user:v1'

export function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function getStoredUser() {
  try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null') }
  catch { return null }
}

export function setStoredAuth(token, user) {
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export function clearStoredAuth() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}
