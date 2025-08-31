// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://shakespearegpt-production.up.railway.app'

export const API_ENDPOINTS = {
  generate: `${API_BASE_URL}/generate`,
  history: `${API_BASE_URL}/history`,
  stats: `${API_BASE_URL}/stats`
}

export default API_BASE_URL
