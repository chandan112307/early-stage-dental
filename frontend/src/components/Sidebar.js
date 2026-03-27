import React from 'react';
import './Sidebar.css';

const NAV_ITEMS = [
  {
    id: 'upload',
    label: 'Upload X-ray',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    ),
  },
];

export default function Sidebar({ activeNav, onNavChange, isOpen, onToggle }) {
  return (
    <>
      <button
        className="sidebar-hamburger"
        onClick={onToggle}
        aria-label="Toggle navigation menu"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      <div
        className={`sidebar-overlay${isOpen ? ' visible' : ''}`}
        onClick={onToggle}
        aria-hidden="true"
      />

      <nav className={`sidebar${isOpen ? ' open' : ''}`} aria-label="Main navigation">
        <div className="sidebar-brand">
          <div className="sidebar-brand-title">
            <span className="sidebar-brand-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </span>
            CariesDetect AI
          </div>
        </div>

        <div className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`sidebar-nav-item${activeNav === item.id ? ' active' : ''}`}
              onClick={() => onNavChange(item.id)}
              aria-current={activeNav === item.id ? 'page' : undefined}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      </nav>
    </>
  );
}
