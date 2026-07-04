"""Tests for auth session invalidation (Phase 1.1)."""
import pytest
from datetime import datetime, timedelta, timezone

from app.services.auth_service import AuthService
from app.models.user import User, Session as DBSession


class FakeDBSession:
    """Minimal in-memory SQLAlchemy-style session stub."""

    def __init__(self, users=None, sessions=None):
        self._users = users or []
        self._sessions = sessions or []
        self.deleted = []
        self.committed = 0

    def query(self, model):
        return FakeQuery(model, self)

    def add(self, obj):
        if isinstance(obj, User):
            self._users.append(obj)
        elif isinstance(obj, DBSession):
            self._sessions.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)
        if isinstance(obj, DBSession) and obj in self._sessions:
            self._sessions.remove(obj)

    def commit(self):
        self.committed += 1

    def refresh(self, obj):
        pass


class FakeQuery:
    def __init__(self, model, db):
        self.model = model
        self.db = db
        self._filters = []

    def filter(self, *conditions):
        self._filters.extend(conditions)
        return self

    def first(self):
        if self.model is DBSession:
            for s in self.db._sessions:
                if all(self._eval(cond, s) for cond in self._filters):
                    return s
        elif self.model is User:
            for u in self.db._users:
                if all(self._eval(cond, u) for cond in self._filters):
                    return u
        return None

    def all(self):
        if self.model is DBSession:
            return [s for s in self.db._sessions if all(self._eval(cond, s) for cond in self._filters)]
        return []

    @staticmethod
    def _eval(cond, obj):
        # Very limited SQLAlchemy binary-expression evaluator sufficient for these tests.
        try:
            left = cond.left.key
            right = cond.right.value
            if left == "token":
                return getattr(obj, "token") == right
            if left == "user_id":
                return getattr(obj, "user_id") == right
            if left == "email":
                return getattr(obj, "email") == right
            return True
        except Exception:
            return True


def _make_user(id="user-1", email="test@example.com"):
    user = User()
    user.id = id
    user.email = email
    user.password = "hashed"
    return user


def _make_session(user_id, token="tok-1"):
    sess = DBSession()
    sess.id = "sess-1"
    sess.user_id = user_id
    sess.token = token
    sess.expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    return sess


def test_delete_session_removes_matching_token():
    user = _make_user()
    session = _make_session(user.id, token="valid-token")
    db = FakeDBSession(users=[user], sessions=[session])

    result = AuthService.delete_session(db, "valid-token")

    assert result is True
    assert session in db.deleted
    assert db.committed == 1


def test_delete_session_returns_false_when_token_missing():
    db = FakeDBSession()

    result = AuthService.delete_session(db, "missing-token")

    assert result is False
    assert db.deleted == []
    assert db.committed == 0


def test_revoke_user_sessions_deletes_all_sessions_for_user():
    user = _make_user()
    sess1 = _make_session(user.id, token="tok-1")
    sess2 = _make_session(user.id, token="tok-2")
    other_user_session = _make_session("user-2", token="tok-3")
    db = FakeDBSession(users=[user], sessions=[sess1, sess2, other_user_session])

    count = AuthService.revoke_user_sessions(db, user.id)

    assert count == 2
    assert sess1 in db.deleted
    assert sess2 in db.deleted
    assert other_user_session not in db.deleted
    assert db.committed == 1


def test_reset_password_revokes_existing_sessions():
    user = _make_user()
    old_session = _make_session(user.id, token="old-token")
    # Minimal password reset token stub; real schema would enforce column types.
    from app.models.user import PasswordResetToken
    reset_token = PasswordResetToken()
    reset_token.email = user.email
    reset_token.token = "reset-1"
    reset_token.used = False
    reset_token.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    db = FakeDBSession(users=[user], sessions=[old_session])
    db._password_reset_tokens = [reset_token]

    # Monkey-patch the query so reset_password finds the token.
    original_query = db.query

    def patched_query(model):
        if model is PasswordResetToken:
            q = FakeQuery(model, db)
            q.first = lambda: reset_token
            q.all = lambda: [reset_token]
            return q
        return original_query(model)

    db.query = patched_query

    result = AuthService.reset_password(db, "reset-1", "new-password-123")

    assert result is True
    assert old_session in db.deleted
    # reset_password now commits twice: once after revoking sessions, once at
    # the end. The exact count is an implementation detail; both commits are
    # idempotent here.
    assert db.committed >= 1
