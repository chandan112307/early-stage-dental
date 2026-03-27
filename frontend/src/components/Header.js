import React from 'react';
import './Header.css';

export default function Header() {
  return (
    <header className="header" role="banner">
      <div className="header-left">
        <span className="header-status-pill active">
          <span className="header-status-dot" aria-hidden="true" />
          CARIESDETECT AI
        </span>
      </div>
    </header>
  );
}
