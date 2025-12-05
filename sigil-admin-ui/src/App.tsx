import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider } from './contexts/ThemeContext'
import { Layout } from './components/Layout'
import { StatusPage } from './pages/StatusPage'
import { IndexPage } from './pages/IndexPage'
import { VectorPage } from './pages/VectorPage'
import { LogsPage } from './pages/LogsPage'
import { ConfigPage } from './pages/ConfigPage'

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Navigate to="/status" replace />} />
            <Route path="/status" element={<StatusPage />} />
            <Route path="/index" element={<IndexPage />} />
            <Route path="/vector" element={<VectorPage />} />
            <Route path="/logs" element={<LogsPage />} />
            <Route path="/config" element={<ConfigPage />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App


