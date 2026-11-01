-- Opponent styles. `setup` holds the table's style mode and which seats are
-- hidden; it is private and leaves only through the manager's projections.
-- Bot reasoning notes are reviewer-only and kept apart from the event log,
-- which replay must regenerate byte for byte.

ALTER TABLE arena_tables ADD COLUMN setup jsonb;

CREATE TABLE bot_notes (
    table_id     text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    decision_id  text NOT NULL,
    seat         integer NOT NULL,
    note         text NOT NULL,
    PRIMARY KEY (table_id, decision_id)
);
