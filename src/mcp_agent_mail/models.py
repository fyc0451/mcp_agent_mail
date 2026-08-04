"""SQLModel data models representing agents, messages, projects, and file reservations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, Index, UniqueConstraint
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def _utcnow_naive() -> datetime:
    """Return current UTC time as a naive datetime for SQLite compatibility.

    SQLite stores datetimes without timezone info. Using naive UTC datetimes
    throughout ensures consistent comparisons and avoids 'can't compare
    offset-naive and offset-aware datetimes' errors in SQLAlchemy ORM evaluator.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Project(SQLModel, table=True):
    __tablename__ = "projects"

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(index=True, unique=True, max_length=255)
    human_key: str = Field(max_length=255, index=True)
    created_at: datetime = Field(default_factory=_utcnow_naive)
    archived_at: Optional[datetime] = Field(default=None)

class Product(SQLModel, table=True):
    """Logical grouping across multiple repositories for product-wide inbox/search and threads."""

    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("product_uid", name="uq_product_uid"), UniqueConstraint("name", name="uq_product_name"))

    id: Optional[int] = Field(default=None, primary_key=True)
    product_uid: str = Field(index=True, max_length=64)
    name: str = Field(index=True, max_length=255)
    created_at: datetime = Field(default_factory=_utcnow_naive)

class ProductProjectLink(SQLModel, table=True):
    """Associates a Project with a Product (many-to-many via link table)."""

    __tablename__ = "product_project_links"
    __table_args__ = (
        UniqueConstraint("product_id", "project_id", name="uq_product_project"),
        Index("idx_product_project", "product_id", "project_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    product_id: int = Field(foreign_key="products.id", index=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    created_at: datetime = Field(default_factory=_utcnow_naive)


class Agent(SQLModel, table=True):
    __tablename__ = "agents"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_agent_project_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    name: str = Field(index=True, max_length=128)
    program: str = Field(max_length=128)
    model: str = Field(max_length=128)
    task_description: str = Field(default="", max_length=2048)
    inception_ts: datetime = Field(default_factory=_utcnow_naive)
    last_active_ts: datetime = Field(default_factory=_utcnow_naive)
    attachments_policy: str = Field(default="auto", max_length=16)
    contact_policy: str = Field(default="auto", max_length=16)  # open | auto | contacts_only | block_all
    registration_token: Optional[str] = Field(default=None, max_length=64, index=True)
    retired_at: Optional[datetime] = Field(default=None)


class MessageRecipient(SQLModel, table=True):
    __tablename__ = "message_recipients"
    __table_args__ = (
        Index("idx_message_recipients_agent_message", "agent_id", "message_id"),
    )

    message_id: int = Field(foreign_key="messages.id", primary_key=True)
    agent_id: int = Field(foreign_key="agents.id", primary_key=True)
    kind: str = Field(max_length=8, default="to")
    read_ts: Optional[datetime] = Field(default=None)
    ack_ts: Optional[datetime] = Field(default=None)


class Message(SQLModel, table=True):
    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_messages_project_created", "project_id", "created_ts"),
        Index("idx_messages_project_sender_created", "project_id", "sender_id", "created_ts"),
        Index("idx_messages_project_topic", "project_id", "topic"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    sender_id: int = Field(foreign_key="agents.id", index=True)
    thread_id: Optional[str] = Field(default=None, index=True, max_length=128)
    # Direct parent→child reply edge (the specific message this one replies to),
    # distinct from `thread_id` which groups a whole conversation. Nullable: a
    # top-level message replies to nothing. (#188)
    reply_to: Optional[int] = Field(default=None, foreign_key="messages.id", index=True)
    topic: Optional[str] = Field(default=None, max_length=64)
    subject: str = Field(max_length=512)
    body_md: str
    importance: str = Field(default="normal", max_length=16)
    ack_required: bool = Field(default=False)
    created_ts: datetime = Field(default_factory=_utcnow_naive)
    attachments: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, server_default="[]"),
    )


class FileReservation(SQLModel, table=True):
    __tablename__ = "file_reservations"
    __table_args__ = (
        Index("idx_file_reservations_project_released_expires", "project_id", "released_ts", "expires_ts"),
        Index("idx_file_reservations_project_agent_released", "project_id", "agent_id", "released_ts"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    # Nullable so a reservation can outlive its owning agent — when the agent
    # row is deleted (manual cleanup, project hygiene, etc.) the reservation
    # becomes "orphaned" and must still be discoverable so it can be
    # auto-released by the staleness sweeper instead of pinning the path
    # forever. (#161)
    agent_id: Optional[int] = Field(default=None, foreign_key="agents.id", index=True)
    path_pattern: str = Field(max_length=512)
    exclusive: bool = Field(default=True)
    reason: str = Field(default="", max_length=512)
    created_ts: datetime = Field(default_factory=_utcnow_naive)
    expires_ts: datetime
    released_ts: Optional[datetime] = None


class AgentLink(SQLModel, table=True):
    """Directed contact link request from agent A to agent B.

    When approved, messages may be sent cross-project between A and B.
    """

    __tablename__ = "agent_links"
    __table_args__ = (UniqueConstraint("a_project_id", "a_agent_id", "b_project_id", "b_agent_id", name="uq_agentlink_pair"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    a_project_id: int = Field(foreign_key="projects.id", index=True)
    a_agent_id: int = Field(foreign_key="agents.id", index=True)
    b_project_id: int = Field(foreign_key="projects.id", index=True)
    b_agent_id: int = Field(foreign_key="agents.id", index=True)
    status: str = Field(default="pending", max_length=16)  # pending | approved | blocked
    reason: str = Field(default="", max_length=512)
    created_ts: datetime = Field(default_factory=_utcnow_naive)
    updated_ts: datetime = Field(default_factory=_utcnow_naive)
    expires_ts: Optional[datetime] = None


class WindowIdentity(SQLModel, table=True):
    """Persistent window-based agent identity tied to a tmux/terminal window.

    Agents that share the same window_uuid within a project share a persistent
    identity that survives session restarts, eliminating per-session registration
    overhead and enabling tracking of which window/pane is doing what.
    """

    __tablename__ = "window_identities"
    __table_args__ = (
        UniqueConstraint("project_id", "window_uuid", name="uq_window_identity_project_uuid"),
        Index("idx_window_identities_project_active", "project_id", "expires_ts"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    window_uuid: str = Field(max_length=64, index=True)
    display_name: str = Field(max_length=128)
    created_ts: datetime = Field(default_factory=_utcnow_naive)
    last_active_ts: datetime = Field(default_factory=_utcnow_naive)
    expires_ts: Optional[datetime] = Field(default=None)


class MessageSummary(SQLModel, table=True):
    """Stored on-demand project-wide message summary."""

    __tablename__ = "message_summaries"
    __table_args__ = (
        Index("idx_summaries_project_end", "project_id", "end_ts"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    summary_text: str
    start_ts: datetime
    end_ts: datetime
    source_message_count: int = Field(default=0)
    source_thread_ids: str = Field(default="[]")  # JSON array of thread IDs
    llm_model: Optional[str] = Field(default=None, max_length=128)
    cost_usd: Optional[float] = Field(default=None)
    created_ts: datetime = Field(default_factory=_utcnow_naive)


class Channel(SQLModel, table=True):
    """Project-scoped public channel (the ``channel:<slug>`` recipient).

    A channel belongs to exactly one project (project-scoped). Agents subscribe
    via :class:`ChannelSubscription`, and may subscribe to channels of other
    projects. Channel identity within a project is the ``name``, unique per
    project.
    """

    __tablename__ = "channels"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_channel_project_name"),
        Index("idx_channels_project_created", "project_id", "created_ts"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    name: str = Field(max_length=128)
    created_ts: datetime = Field(default_factory=_utcnow_naive)


class ChannelSubscription(SQLModel, table=True):
    """Cross-project channel subscription (server-side subscriber table).

    Associates an agent with a channel so that the agent receives the
    channel's fanout. An agent defaults to its registration project but may
    subscribe to channels of other projects; the subscription is uniquely
    identified by ``(channel_id, agent_id)``.
    """

    __tablename__ = "channel_subscriptions"
    __table_args__ = (
        UniqueConstraint("channel_id", "agent_id", name="uq_channel_subscription"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    channel_id: int = Field(foreign_key="channels.id")
    agent_id: int = Field(foreign_key="agents.id", index=True)
    created_ts: datetime = Field(default_factory=_utcnow_naive)


class ChannelMessage(SQLModel, table=True):
    """A message posted to a channel (channel history, "blackboard").

    Independent from :class:`Message`: channel messages are not delivered to
    per-recipient ``message_recipients`` rows. Subscribers pull what they have
    not yet read via a per-agent read cursor (:class:`ChannelReadCursor`);
    the monotonically increasing ``id`` is the ordering / cursor key.

    The sender may be an agent from another project (cross-project subscriber
    posting to a channel it subscribes to), so ``sender_id`` is a plain FK to
    ``agents.id`` without a project coupling.
    """

    __tablename__ = "channel_messages"
    __table_args__ = (
        Index("idx_channel_messages_channel_created", "channel_id", "created_ts"),
        Index("idx_channel_messages_channel_id", "channel_id", "id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    channel_id: int = Field(foreign_key="channels.id")
    sender_id: int = Field(foreign_key="agents.id")
    subject: str = Field(max_length=512)
    body_md: str
    importance: str = Field(default="normal", max_length=16)
    created_ts: datetime = Field(default_factory=_utcnow_naive)
    attachments: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, server_default="[]"),
    )


class ChannelReadCursor(SQLModel, table=True):
    """Per-agent read cursor for a channel (uniquely keyed by channel+agent).

    ``last_read_message_id`` is NULL until the agent has read any channel
    message (NULL means "no messages read yet", i.e. cursor at the start).
    A NULL initial value avoids referencing a non-existent ``channel_messages``
    row under FK enforcement (a cursor at 0 would be an orphan reference).

    The FK only pins the column type/domain to ``channel_messages.id``;
    asserting that a non-NULL ``last_read_message_id`` belongs to the SAME
    channel as this cursor row is left to the advancing tool (a plain FK cannot
    express cross-row consistency in SQLite), noted here for that tool.
    """

    __tablename__ = "channel_read_cursors"
    __table_args__ = (
        UniqueConstraint("channel_id", "agent_id", name="uq_channel_read_cursor"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    channel_id: int = Field(foreign_key="channels.id")
    agent_id: int = Field(foreign_key="agents.id", index=True)
    last_read_message_id: Optional[int] = Field(
        default=None,
        foreign_key="channel_messages.id",
    )
    created_ts: datetime = Field(default_factory=_utcnow_naive)
    updated_ts: datetime = Field(default_factory=_utcnow_naive)


class MentionDelivery(SQLModel, table=True):
    """Delivery mapping for an ``@mention`` inside a channel message.

    Records that ``mentioned_agent_id`` was delivered a durable receipt
    (``receipt_message_id``) for the channel message ``source_channel_message_id``.
    Only successful deliveries are stored, so ``receipt_message_id`` is
    non-nullable.

    The mapping is unique per (source channel message, mentioned agent), but
    multiple rows may reference the SAME receipt message: when several
    mentioned agents live in the same project, one ``messages`` receipt row
    addressed to multiple recipients is reused, so the duplicate delivery is
    recorded once per mapping rather than once per recipient row.
    """

    __tablename__ = "mention_deliveries"
    __table_args__ = (
        UniqueConstraint("source_channel_message_id", "mentioned_agent_id", name="uq_mention_delivery"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    source_channel_message_id: int = Field(foreign_key="channel_messages.id")
    mentioned_agent_id: int = Field(foreign_key="agents.id", index=True)
    receipt_message_id: int = Field(foreign_key="messages.id")
    created_ts: datetime = Field(default_factory=_utcnow_naive)


class ProjectSiblingSuggestion(SQLModel, table=True):
    """LLM-ranked sibling project suggestion (undirected pair)."""

    __tablename__ = "project_sibling_suggestions"
    __table_args__ = (UniqueConstraint("project_a_id", "project_b_id", name="uq_project_sibling_pair"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    project_a_id: int = Field(foreign_key="projects.id", index=True)
    project_b_id: int = Field(foreign_key="projects.id", index=True)
    score: float = Field(default=0.0)
    status: str = Field(default="suggested", max_length=16)  # suggested | confirmed | dismissed
    rationale: str = Field(default="", max_length=4096)
    created_ts: datetime = Field(default_factory=_utcnow_naive)
    evaluated_ts: datetime = Field(default_factory=_utcnow_naive)
    confirmed_ts: Optional[datetime] = Field(default=None)
    dismissed_ts: Optional[datetime] = Field(default=None)
