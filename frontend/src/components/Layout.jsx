import React from 'react'
import { AppBar, Toolbar, Typography, Container, Box, IconButton, Button, Stack } from '@mui/material'
import ScienceIcon from '@mui/icons-material/Science'
import { Link as RouterLink } from 'react-router-dom'

export default function Layout({ children }) {
  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Toolbar>
          <ScienceIcon color="primary" sx={{ mr: 1 }} />
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 700 }}>
            ASD Behavioral Analysis
          </Typography>
          <Stack direction="row" spacing={1}>
            <Button component={RouterLink} to="/" color="primary">Overview</Button>
            <Button component={RouterLink} to="/face" color="primary">Face Test</Button>
            <Button component={RouterLink} to="/results" color="primary">Results</Button>
            <Button component={RouterLink} to="/admin" color="secondary">Admin</Button>
          </Stack>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ py: 3 }}>
        {children}
      </Container>
    </Box>
  )
}


