-- Durable table state (spec 9.2). Everything here is private: decks, seeds,
-- hole cards and state hashes live in these rows and leave only through
-- arena.table.projections.

CREATE TABLE arena_tables (
    table_id         text PRIMARY KEY,
    config           jsonb NOT NULL,
    config_hash      text NOT NULL,
    deck_seed        text,
    bot_seed         text NOT NULL,
    host_token_hash  text NOT NULL UNIQUE,
    status           text NOT NULL CHECK (status IN ('waiting', 'running', 'complete', 'halted')),
    version          bigint NOT NULL CHECK (version >= 0),
    state_version    bigint NOT NULL DEFAULT 0,
    event_count      bigint NOT NULL DEFAULT 0,
    scheduler        jsonb,
    scheduler_at     timestamptz,
    halted_reason    text,
    owner_id         text,
    lease_until      timestamptz,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE participants (
    table_id       text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    seat           integer NOT NULL,
    kind           text NOT NULL CHECK (kind IN ('human', 'bot')),
    policy         text,
    artifact_hash  text,
    claimed        boolean NOT NULL DEFAULT false,
    token_hash     text UNIQUE,
    PRIMARY KEY (table_id, seat)
);

CREATE TABLE hands (
    table_id      text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    hand_id       text NOT NULL,
    hand_number   integer NOT NULL,
    start_seq     bigint NOT NULL,
    start_hash    text NOT NULL,
    end_seq       bigint,
    final_stacks  jsonb,
    PRIMARY KEY (table_id, hand_id),
    UNIQUE (table_id, hand_number)
);

CREATE TABLE events (
    table_id        text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    seq             bigint NOT NULL CHECK (seq >= 0),
    hand_id         text,
    schema_version  integer NOT NULL,
    type            text NOT NULL,
    data            jsonb NOT NULL,
    PRIMARY KEY (table_id, seq)
);

CREATE TABLE action_receipts (
    table_id     text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    action_id    text NOT NULL,
    seat         integer NOT NULL,
    decision_id  text NOT NULL,
    fingerprint  text NOT NULL,
    body         jsonb NOT NULL,
    version      bigint NOT NULL,
    PRIMARY KEY (table_id, action_id),
    UNIQUE (table_id, decision_id)
);

CREATE TABLE bot_checkpoints (
    table_id       text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    seat           integer NOT NULL,
    policy         text NOT NULL,
    artifact_hash  text NOT NULL,
    rng_state      jsonb NOT NULL,
    decisions      integer NOT NULL,
    version        bigint NOT NULL,
    PRIMARY KEY (table_id, seat)
);

CREATE TABLE snapshots (
    table_id     text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    hand_number  integer NOT NULL,
    seq          bigint NOT NULL,
    start_hash   text NOT NULL,
    point        jsonb NOT NULL,
    engine       jsonb NOT NULL,
    PRIMARY KEY (table_id, hand_number)
);

CREATE TABLE awards (
    table_id     text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    seat         integer NOT NULL,
    positions    jsonb NOT NULL,
    points       double precision NOT NULL,
    method       text NOT NULL,
    hand_number  integer NOT NULL,
    PRIMARY KEY (table_id, seat)
);

CREATE TABLE table_pauses (
    pause_id      bigserial PRIMARY KEY,
    table_id      text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    reason        text NOT NULL,
    decision_id   text,
    remaining_ms  integer,
    paused_at     timestamptz NOT NULL,
    resumed_at    timestamptz
);
CREATE INDEX table_pauses_open ON table_pauses (table_id) WHERE resumed_at IS NULL;

CREATE TABLE outbox (
    outbox_id     bigserial PRIMARY KEY,
    table_id      text NOT NULL REFERENCES arena_tables ON DELETE CASCADE,
    version       bigint NOT NULL,
    kind          text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    published_at  timestamptz,
    UNIQUE (table_id, version)
);
CREATE INDEX outbox_unpublished ON outbox (table_id, version) WHERE published_at IS NULL;
