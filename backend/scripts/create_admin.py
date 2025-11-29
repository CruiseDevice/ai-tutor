#!/usr/bin/env python3
"""
Script to create or upgrade a user to admin role.

Usage:
    python scripts/create_admin.py <email> <password> [--super-admin]

Examples:
    python scripts/create_admin.py admin@example.com password123
    python scripts/create_admin.py admin@example.com password123 --super-admin
"""

import sys
import os

# Add parent directory to path to import app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models.user import User, UserRole
from app.core.security import get_password_hash


def create_admin_user(email: str, password: str, super_admin: bool = False):
    """Create or upgrade a user to admin role."""
    db = SessionLocal()

    try:
        # Check if user exists
        existing = db.query(User).filter(User.email == email).first()

        role = UserRole.SUPER_ADMIN if super_admin else UserRole.ADMIN
        role_name = "super admin" if super_admin else "admin"

        if existing:
            # Upgrade to admin
            existing.role = role
            db.commit()
            print(f"✓ Upgraded {email} to {role_name}")
        else:
            # Create new admin user
            hashed_password = get_password_hash(password)
            admin = User(
                email=email,
                password=hashed_password,
                role=role
            )
            db.add(admin)
            db.commit()
            print(f"✓ Created {role_name} user: {email}")

    except Exception as e:
        print(f"✗ Error: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/create_admin.py <email> <password> [--super-admin]")
        print("\nExamples:")
        print("  python scripts/create_admin.py admin@example.com password123")
        print("  python scripts/create_admin.py admin@example.com password123 --super-admin")
        sys.exit(1)

    email = sys.argv[1]
    password = sys.argv[2]
    super_admin = "--super-admin" in sys.argv

    create_admin_user(email, password, super_admin)
