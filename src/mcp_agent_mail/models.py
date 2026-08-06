"""SQLModel data models representing agents, messages, projects, and file reservations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, Index, UniqueConstraint, text
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


class TeamProject(SQLModel, table=True):
    """User-visible logical group backed by an opaque Hub routing project.

    ``routing_project_id`` points at the existing Agent Mail message space, but
    that Project uses a server-generated opaque key and never a client path.
    Technical Agent Mail projects are therefore invisible to the Team API.
    """

    __tablename__ = "team_projects"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_team_project_slug"),
        UniqueConstraint("routing_project_id", name="uq_team_project_routing"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(index=True, max_length=128)
    name: str = Field(max_length=255)
    routing_project_id: int = Field(foreign_key="projects.id", index=True)
    created_at: datetime = Field(default_factory=_utcnow_naive)
    archived_at: Optional[datetime] = Field(default=None)


class TeamProjectAgentBinding(SQLModel, table=True):
    """Explicit binding of an existing Agent identity to a TeamProject.

    Binding only references an existing agent id — it never accepts or scans
    local paths. Unbinding keeps the row with ``status="unbound"`` as history;
    re-binding revives the same row (idempotent per team_project+agent pair).
    """

    __tablename__ = "team_project_agent_bindings"
    __table_args__ = (
        UniqueConstraint("team_project_id", "agent_id", name="uq_tpab_team_agent"),
        Index("ix_tpab_agent_status", "agent_id", "status"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    team_project_id: int = Field(foreign_key="team_projects.id", index=True)
    agent_id: int = Field(foreign_key="agents.id", index=True)
    status: str = Field(default="active", max_length=16)  # active | unbound
    bound_by_human_id: Optional[int] = Field(default=None, foreign_key="humans.id")
    created_at: datetime = Field(default_factory=_utcnow_naive)
    updated_at: datetime = Field(default_factory=_utcnow_naive)


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
    # M3a identity: owning human (nullable so pre-M3a agents stay unowned).
    # The owner's project need not match this agent's project_id — a human is a
    # global identity, while default_agent_id lives on ProjectHumanMembership.
    owner_id: Optional[int] = Field(default=None, foreign_key="humans.id", index=True)


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


class HubAuditEvent(SQLModel, table=True):
    """Append-only, field-whitelisted audit event for Hub channel actions.

    Audit rows intentionally store only trusted identifiers and controlled
    outcome/reason values. Message text, registration tokens, exception text,
    and arbitrary client metadata do not belong in this table.

    Reference ids are durable tombstones: current queries never inner-join
    agents/messages, so explicit hard deletion may leave an id that no longer
    resolves while preserving the historical event. If SQLite FK enforcement
    is enabled later, these nullable references must use ON DELETE SET NULL.
    """

    __tablename__ = "hub_audit_events"
    __table_args__ = (
        Index("idx_hub_audit_events_project_id", "project_id", "id"),
        Index("idx_hub_audit_events_actor_id", "actor_agent_id", "id"),
        Index("idx_hub_audit_events_source", "source_type", "source_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    actor_agent_id: Optional[int] = Field(default=None, foreign_key="agents.id")
    event_type: str = Field(max_length=64)
    source_type: str = Field(max_length=32)
    source_id: int
    outcome: str = Field(max_length=32)
    reason: Optional[str] = Field(default=None, max_length=32)
    target_project_id: Optional[int] = Field(default=None, foreign_key="projects.id")
    target_agent_id: Optional[int] = Field(default=None, foreign_key="agents.id")
    related_message_id: Optional[int] = Field(default=None, foreign_key="messages.id")
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


class Human(SQLModel, table=True):
    """M3a global human identity.

    ``subject`` is the stable opaque auth identity; ``id`` is its local integer
    primary key. ``display_name`` may repeat across humans (it is only for
    display; @-mention disambiguation is handled by membership handles).
    """

    __tablename__ = "humans"

    id: Optional[int] = Field(default=None, primary_key=True)
    subject: str = Field(index=True, unique=True, max_length=255)
    display_name: str = Field(max_length=255)
    created_at: datetime = Field(default_factory=_utcnow_naive)


class ProjectHumanMembership(SQLModel, table=True):
    """M3a membership of a global human within a project (per-project role).

    ``mention_handle`` is unique per project and must not collide with an
    active agent name in the same project. ``default_agent_id`` is the human's
    default delivery target within THIS project (may be NULL) and must belong to
    the same project and to this human.

    Note: a plain unique constraint cannot express the cross-row default-agent
    invariant (same project + owned by this human); the service layer enforces it.
    """

    __tablename__ = "project_human_memberships"
    __table_args__ = (
        UniqueConstraint("project_id", "human_id", name="uq_phm_project_human"),
        UniqueConstraint("project_id", "mention_handle", name="uq_phm_project_mention_handle"),
        # human_id/project_id/default_agent_id 索引由 Field(index=True) 生成,
        # 不在此重复声明。
        # Database-level case-insensitive uniqueness for mention_handle within a
        # project: the plain unique constraint is case-sensitive, and human
        # handles must be unique ignoring case (lead audit point 2).
        # NOTE: must reference the COLUMN expression lower(mention_handle),
        # not func.lower('mention_handle') which would compile to the string
        # constant lower('mention_handle') and reject every second membership.
        Index(
            "uq_phm_project_handle_ci",
            "project_id",
            text("lower(mention_handle)"),
            unique=True,
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    human_id: int = Field(foreign_key="humans.id", index=True)
    mention_handle: str = Field(max_length=128)
    role: str = Field(default="member", max_length=32)  # member | admin
    status: str = Field(default="active", max_length=16)  # active | invited | removed
    default_agent_id: Optional[int] = Field(
        default=None, foreign_key="agents.id", index=True
    )
    created_at: datetime = Field(default_factory=_utcnow_naive)
    updated_at: datetime = Field(default_factory=_utcnow_naive)


class HumanInboxItem(SQLModel, table=True):
    """M3a durable human inbox entry (人工收件箱).

    Created when an @<mention_handle> channel mention targets a human whose
    membership has no usable default agent. ``message_id`` points at the
    receipt Message row (no agent recipients); humans read via /hub/api/inbox
    with their JWT principal. Delivery is idempotent per source channel
    message and human.
    """

    __tablename__ = "human_inbox_items"
    __table_args__ = (
        UniqueConstraint("message_id", "human_id", name="uq_hii_message_human"),
        UniqueConstraint(
            "source_channel_message_id", "human_id", name="uq_hii_source_human"
        ),
        Index("ix_hii_human_unread", "human_id", "read_ts"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    human_id: int = Field(foreign_key="humans.id", index=True)
    message_id: int = Field(foreign_key="messages.id", index=True)
    source_channel_message_id: Optional[int] = Field(
        default=None, foreign_key="channel_messages.id", index=True
    )
    kind: str = Field(default="mention", max_length=16)
    read_ts: Optional[datetime] = Field(default=None)
    created_ts: datetime = Field(default_factory=_utcnow_naive)
