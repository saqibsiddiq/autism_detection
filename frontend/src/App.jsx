import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider, CssBaseline } from '@mui/material'
import theme from './theme'
import Layout from './components/Layout.jsx'
import Overview from './pages/Overview.jsx'
import FaceTest from './pages/FaceTest.jsx'
import Results from './pages/Results.jsx'
import Admin from './pages/Admin.jsx'

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Layout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/face" element={<FaceTest />} />
          <Route path="/results" element={<Results />} />
          <Route path="/admin" element={<Admin />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </ThemeProvider>
  )
}


