import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Resolve SQLite DB path in project root
const dbPath = path.resolve(__dirname, '..', 'asddb.sqlite3');
const db = new Database(dbPath);

// Ensure tables exist (mirror of Python schema, simplified)
db.exec(`
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  session_id TEXT UNIQUE,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now')),
  age_group TEXT,
  consent_given INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS assessments (
  id INTEGER PRIMARY KEY,
  user_id INTEGER,
  assessment_type TEXT,
  status TEXT DEFAULT 'in_progress',
  started_at TEXT DEFAULT (datetime('now')),
  completed_at TEXT,
  total_duration INTEGER
);
CREATE TABLE IF NOT EXISTS gaze_data (
  id INTEGER PRIMARY KEY,
  assessment_id INTEGER,
  task_name TEXT,
  task_type TEXT,
  frame_number INTEGER,
  timestamp REAL,
  face_detected INTEGER,
  gaze_x REAL,
  gaze_y REAL,
  eye_contact_score REAL,
  fixation_duration REAL,
  saccade_amplitude REAL,
  social_attention_score REAL,
  recorded_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS assessment_results (
  id INTEGER PRIMARY KEY,
  assessment_id INTEGER,
  questionnaire_scores TEXT,
  gaze_metrics TEXT,
  ml_prediction TEXT,
  risk_assessment TEXT,
  recommendations TEXT,
  overall_score REAL,
  risk_level TEXT,
  confidence_score REAL,
  created_at TEXT DEFAULT (datetime('now'))
);
`);

const app = express();
app.use(cors());
app.use(helmet());
app.use(express.json({ limit: '2mb' }));
app.use(morgan('dev'));

// Health
app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

// Sessions / Users
app.post('/api/session', (req, res) => {
  const { sessionId, ageGroup, consentGiven } = req.body || {};
  if (!sessionId) return res.status(400).json({ error: 'sessionId required' });
  const upsert = db.prepare(
    `INSERT INTO users (session_id, age_group, consent_given) VALUES (?, ?, ?)
     ON CONFLICT(session_id) DO UPDATE SET updated_at = datetime('now')`
  );
  upsert.run(sessionId, ageGroup || null, consentGiven ? 1 : 0);
  const user = db.prepare('SELECT * FROM users WHERE session_id = ?').get(sessionId);
  res.json({ user });
});

// Assessments
app.post('/api/assessments', (req, res) => {
  const { userId, assessmentType } = req.body || {};
  if (!userId || !assessmentType) return res.status(400).json({ error: 'userId and assessmentType required' });
  const stmt = db.prepare(
    `INSERT INTO assessments (user_id, assessment_type) VALUES (?, ?)`
  );
  const info = stmt.run(userId, assessmentType);
  const assessment = db.prepare('SELECT * FROM assessments WHERE id = ?').get(info.lastInsertRowid);
  res.json({ assessment });
});

// Gaze data (aggregate or per-frame)
app.post('/api/assessments/:id/gaze', (req, res) => {
  const assessmentId = Number(req.params.id);
  const body = req.body;
  const insert = db.prepare(`INSERT INTO gaze_data (
    assessment_id, task_name, task_type, frame_number, timestamp, face_detected, gaze_x, gaze_y,
    eye_contact_score, fixation_duration, saccade_amplitude, social_attention_score
  ) VALUES (@assessment_id, @task_name, @task_type, @frame_number, @timestamp, @face_detected, @gaze_x, @gaze_y,
    @eye_contact_score, @fixation_duration, @saccade_amplitude, @social_attention_score)`);

  const toRow = (d, i = 0) => ({
    assessment_id: assessmentId,
    task_name: d.task_name || body.task_name || 'task',
    task_type: d.task_type || body.task_type || 'generic',
    frame_number: d.frame_number ?? i,
    timestamp: d.timestamp ?? 0,
    face_detected: d.face_detected ? 1 : 0,
    gaze_x: d.gaze_x ?? d.avg_gaze_x ?? 0,
    gaze_y: d.gaze_y ?? d.avg_gaze_y ?? 0,
    eye_contact_score: d.eye_contact_score ?? d.face_detection_rate ?? 0,
    fixation_duration: d.fixation_duration ?? d.avg_fixation_duration ?? 0,
    saccade_amplitude: d.saccade_amplitude ?? d.gaze_velocity_std ?? 0,
    social_attention_score: d.social_attention_score ?? d.social_attention_ratio ?? d.face_preference_ratio ?? 0
  });

  if (Array.isArray(body)) {
    const tx = db.transaction((rows) => {
      for (let i = 0; i < rows.length; i++) insert.run(toRow(rows[i], i));
    });
    tx(body);
    return res.json({ saved: body.length });
  } else if (typeof body === 'object') {
    insert.run(toRow(body, 0));
    return res.json({ saved: 1 });
  }
  res.status(400).json({ error: 'Invalid body' });
});

// Save results
app.post('/api/assessments/:id/results', (req, res) => {
  const assessmentId = Number(req.params.id);
  const { overall_scores, behavioral_patterns, meta, risk_indicators, recommendations } = req.body || {};
  const averageScore = overall_scores && Object.values(overall_scores).length
    ? (Object.values(overall_scores).reduce((a, b) => a + Number(b || 0), 0) / Object.values(overall_scores).length)
    : 0;

  const stmt = db.prepare(`INSERT INTO assessment_results (
    assessment_id, questionnaire_scores, gaze_metrics, ml_prediction, risk_assessment, recommendations,
    overall_score, risk_level, confidence_score
  ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`);

  stmt.run(
    assessmentId,
    JSON.stringify({}),
    JSON.stringify(behavioral_patterns || {}),
    JSON.stringify(meta || {}),
    JSON.stringify(risk_indicators || {}),
    JSON.stringify(recommendations || []),
    averageScore,
    (meta && meta.overall_risk_level) || 'unknown',
    (meta && meta.confidence_level) || 0
  );

  // Mark assessment completed
  const complete = db.prepare(`UPDATE assessments SET status='completed', completed_at=datetime('now') WHERE id = ?`);
  complete.run(assessmentId);

  res.json({ saved: true });
});

// Admin stats
app.get('/api/admin/stats', (_req, res) => {
  const totalUsers = db.prepare('SELECT COUNT(*) as c FROM users').get().c;
  const totalAssessments = db.prepare('SELECT COUNT(*) as c FROM assessments').get().c;
  const completedAssessments = db.prepare("SELECT COUNT(*) as c FROM assessments WHERE status='completed'").get().c;
  const riskRows = db.prepare('SELECT risk_level, COUNT(*) as c FROM assessment_results GROUP BY risk_level').all();
  const risk_distribution = {};
  for (const r of riskRows) if (r.risk_level) risk_distribution[r.risk_level] = r.c;
  res.json({ total_users: totalUsers, total_assessments: totalAssessments, completed_assessments: completedAssessments, completion_rate: totalAssessments ? completedAssessments/totalAssessments : 0, risk_distribution });
});

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => console.log(`API listening on http://localhost:${PORT}`));


