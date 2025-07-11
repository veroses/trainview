import axios from 'axios'

const api = axios.create({
    baseURL: "http://localhost:8000",
     timeout: 10000, // 10s timeout
  headers: {
    'Content-Type': 'application/json',
  },
})

export default api;