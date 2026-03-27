import React from 'react';
import './Header.css';

export default function Header() {
  return (
    <header className="header" role="banner">
      <div className="header-left">
        <span className="header-patient-id">Patient ID: #PX-99281</span>
        <span className="header-status-pill active">
          <span className="header-status-dot" aria-hidden="true" />
          CARIESDETECT AI ACTIVE
        </span>
      </div>

      <div className="header-center">
        <div className="header-search">
          <svg className="header-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="search"
            placeholder="Search findings or patients..."
            aria-label="Search findings or patients"
          />
        </div>
      </div>

      <div className="header-right">
        <button className="header-icon-btn" aria-label="Notifications">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 0 1-3.46 0" />
          </svg>
          <span className="header-notification-badge" aria-hidden="true" />
        </button>

        <button className="header-icon-btn" aria-label="Help">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        </button>

        <div className="header-avatar" aria-label="User profile">AT</div>
      </div>
    </header>
  );
}
