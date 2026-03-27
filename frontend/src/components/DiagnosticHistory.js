import React from 'react';
import './DiagnosticHistory.css';

const MOCK_HISTORY = [
  {
    id: 1,
    date: 'Dec 15, 2024',
    status: 'ok',
    title: 'Routine Periapical X-ray',
    description: 'No abnormalities detected. Routine follow-up in 6 months recommended.',
  },
  {
    id: 2,
    date: 'Oct 03, 2024',
    status: 'warning',
    title: 'Interproximal Caries - #14',
    description: 'Early-stage caries identified between teeth #14 and #15. Treatment planned.',
  },
  {
    id: 3,
    date: 'Jul 22, 2024',
    status: 'ok',
    title: 'Post-Treatment Review',
    description: 'Composite restoration on #19 verified. Margins intact, no recurrence.',
  },
  {
    id: 4,
    date: 'Apr 10, 2024',
    status: 'warning',
    title: 'Occlusal Caries - #19',
    description: 'Moderate caries on occlusal surface of #19. Restoration recommended.',
  },
];

const CheckIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

const WarningIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

export default function DiagnosticHistory() {
  return (
    <section className="diagnostic-history" aria-label="Patient diagnostic history">
      <div className="diagnostic-history-header">
        <div className="diagnostic-history-header-left">
          <div className="diagnostic-history-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
          </div>
          <span className="diagnostic-history-title">Patient Diagnostic History</span>
        </div>
        <button className="diagnostic-history-view-all" aria-label="View all patient records">
          View All Records
        </button>
      </div>

      <div className="diagnostic-history-scroll" role="list" aria-label="Diagnostic history entries">
        {MOCK_HISTORY.map((entry) => (
          <article
            key={entry.id}
            className={`diagnostic-card ${entry.status}`}
            role="listitem"
          >
            <div className="diagnostic-card-top">
              <span className="diagnostic-card-date">{entry.date}</span>
              <span className={`diagnostic-card-status ${entry.status}`}>
                {entry.status === 'ok' ? <CheckIcon /> : <WarningIcon />}
              </span>
            </div>
            <div className="diagnostic-card-title">{entry.title}</div>
            <div className="diagnostic-card-desc">{entry.description}</div>
          </article>
        ))}
      </div>
    </section>
  );
}
