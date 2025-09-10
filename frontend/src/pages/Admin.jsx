import React from 'react'
import client from '../api/client'
import { Card, CardContent, Typography, Grid } from '@mui/material'

export default function Admin() {
  const [stats, setStats] = React.useState(null)
  React.useEffect(() => {
    client.get('/admin/stats').then(({ data }) => setStats(data)).catch(() => setStats(null))
  }, [])

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>Admin Dashboard</Typography>
        {stats ? (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}><Typography>Total Users: {stats.total_users}</Typography></Grid>
            <Grid item xs={12} sm={6} md={3}><Typography>Total Assessments: {stats.total_assessments}</Typography></Grid>
            <Grid item xs={12} sm={6} md={3}><Typography>Completed: {stats.completed_assessments}</Typography></Grid>
            <Grid item xs={12} sm={6} md={3}><Typography>Completion Rate: {Math.round((stats.completion_rate || 0)*100)}%</Typography></Grid>
          </Grid>
        ) : <Typography variant="body2">No data yet.</Typography>}
      </CardContent>
    </Card>
  )
}


