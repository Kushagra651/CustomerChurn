CREATE DATABASE IF NOT EXISTS pulsedirects;
USE pulsedirects;
 
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS customers;
 
-- ─────────────────────────────────────────────────────────────
-- CUSTOMERS — core demographic and service information
-- ─────────────────────────────────────────────────────────────
CREATE TABLE customers (
    customer_id     VARCHAR(20) PRIMARY KEY,   -- original CustomerID from Excel
    city            VARCHAR(100),
    zip_code        INT,
    latitude        FLOAT,
    longitude       FLOAT,
    gender          VARCHAR(10),
    senior_citizen  INT DEFAULT 0,
    partner         VARCHAR(5),
    dependents      VARCHAR(5),
    tenure_months   INT,
    phone_service   VARCHAR(5),
    multiple_lines  VARCHAR(30),
    internet_service    VARCHAR(30),
    online_security     VARCHAR(30),
    online_backup       VARCHAR(30),
    device_protection   VARCHAR(30),
    tech_support        VARCHAR(30),
    streaming_tv        VARCHAR(30),
    streaming_movies    VARCHAR(30),
    contract            VARCHAR(30),
    churn_value     INT DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
 
-- ─────────────────────────────────────────────────────────────
-- PAYMENTS — billing and payment information per customer
-- ─────────────────────────────────────────────────────────────
CREATE TABLE payments (
    payment_id          INT AUTO_INCREMENT PRIMARY KEY,
    customer_id         VARCHAR(20),
    paperless_billing   VARCHAR(5),
    payment_method      VARCHAR(50),
    monthly_charges     FLOAT,
    total_charges       FLOAT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
 
-- ─────────────────────────────────────────────────────────────
-- SUPPORT TICKETS — simulated support interaction data
-- Generated during seeding since Excel has no ticket data.
-- Adds analytical value to the project without fabricating
-- customer-level churn signals.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE support_tickets (
    ticket_id       INT AUTO_INCREMENT PRIMARY KEY,
    customer_id     VARCHAR(20),
    issue_type      VARCHAR(50),
    severity        VARCHAR(20),
    resolution_days INT,
    status          VARCHAR(20),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
 